# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Implementation of the DualBind model for binding affinity prediction."""

from typing import Any, List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from chemprop.features.featurization import BatchMolGraph
import pytorch_lightning as pl
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef, r2_score

from .fann import FANN
from .mpn import MPNEncoder
from .loss import MTLLoss


class DualBind(pl.LightningModule):
    def __init__(self, cfg_model: DictConfig):
        """
        Initialization of the DualBind model.

        Args:
            cfg_model (DictConfig): model configurations.
        """
        super(DualBind, self).__init__()
        self.cfg_model = cfg_model
        self.mse_loss = nn.MSELoss()
        self.loss_ratio = cfg_model.loss_ratio
        self.threshold = cfg_model.threshold
        self.lr = cfg_model.optimizer.lr
        self.anneal_rate = cfg_model.optimizer.anneal_rate
        self.max_residue_atoms = cfg_model.max_residue_atoms
        self.eps_scaling = cfg_model.eps_scaling

        self.mpn = MPNEncoder(cfg_model)
        self.encoder = FANN(cfg_model)

        self.binder_output = nn.Sequential(
            nn.Linear(cfg_model.hidden_size, cfg_model.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg_model.hidden_size, cfg_model.hidden_size),
        )
        self.target_output = nn.Sequential(
            nn.Linear(cfg_model.hidden_size, cfg_model.hidden_size),
            nn.SiLU(),
        )

        # intermediate output storage
        self._validation_outputs = []
        self._test_outputs = []
        
        if cfg_model.use_mtl_loss:
            self.mtl_loss = MTLLoss(num_tasks=2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.anneal_rate)

        if self.cfg_model.use_mtl_loss:
            # sets log_sigma as learnable
            optimizer.param_groups[0]['params'].append(self.mtl_loss.log_sigma)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler},
        }


    def forward(
        self,
        binder: Tuple[torch.Tensor, torch.Tensor, BatchMolGraph, torch.Tensor],
        target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        The forward pass used for inference.

        Args:
            binder (Tuple[torch.Tensor, torch.Tensor, BatchMolGraph, torch.Tensor]): the batched ligand info. The first tensor is the ligand atom coordinates. The second tensor represents atom type. The third list is a batched molecular graph. The fourth tensor is a mask for indicating ligand atoms.
            target (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): the batched target pocket info. The first tensor is the target atom coordinates. The second tensor is a one-hot residue embedding. The third tensor represents all target atoms in the pocket.

        Returns:
            torch.Tensor: the predicted energy.
        """
        bind_X, mol_batch, bind_A = binder
        tgt_X, tgt_S, tgt_A = target
        bind_S = self.mpn(mol_batch)

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_A = bind_A * (bind_X.norm(dim=-1) > 1e-4).long()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()

        mask_2D = (bind_A > 0).float().view(B, N * self.max_residue_atoms, 1) * (tgt_A > 0).float().view(
            B, 1, M * self.max_residue_atoms
        )
        dist = (
            bind_X.view(B, N * self.max_residue_atoms, 1, 3) - tgt_X.view(B, 1, M * self.max_residue_atoms, 3)
        ).norm(
            dim=-1
        )  # [B,N*self.max_residue_atoms,M*self.max_residue_atoms]
        mask_2D = mask_2D * (dist < self.threshold).float()

        h = self.encoder(
            (bind_X, bind_S, bind_A),
            (tgt_X, tgt_S, tgt_A),
        )  # [B,N+M,self.max_residue_atoms,H]

        bind_h = self.binder_output(h[:, :N]).view(B, N * self.max_residue_atoms, -1)
        tgt_h = self.target_output(h[:, N:]).view(B, M * self.max_residue_atoms, -1)
        energy = torch.matmul(bind_h, tgt_h.transpose(1, 2))  # [B,N*self.max_residue_atoms,M*self.max_residue_atoms]
        return (energy * mask_2D).sum(dim=(1, 2))  # [B]

    def _step(
        self,
        binder: Tuple[torch.Tensor, torch.Tensor, BatchMolGraph, torch.Tensor],
        target: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        affinity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Define the forward of the energy-based model, and compute the losses. This method does not support adaptive batching where some samples have affinity label whereas some don't.

        Args:
            binder (Tuple[torch.Tensor, torch.Tensor, BatchMolGraph, torch.Tensor]): the batched ligand info. The first tensor is the ligand atom coordinates. The second tensor represents atom type. The third list is a batched molecular graph. The fourth tensor is a mask for indicating ligand atoms.
            target (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): the batched target pocket info. The first tensor is the target atom coordinates. The second tensor is a one-hot residue embedding. The third tensor represents all target atoms in the pocket.
            affinity (torch.Tensor): the affinity labels

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the denoising score matching loss value, and the mse loss value
        """
        true_X, mol_batch, bind_A = binder
        tgt_X, tgt_S, tgt_A = target
        bind_S = self.mpn(mol_batch) # [B, N, hidden_size]

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()
        atom_mask = (bind_A > 0).float().unsqueeze(-1)

        # Crystal structures
        mask_2D = (bind_A > 0).float().view(B, N * self.max_residue_atoms, 1) * (tgt_A > 0).float().view(
            B, 1, M * self.max_residue_atoms
        )
        dist = (
            true_X.view(B, N * self.max_residue_atoms, 1, 3) - tgt_X.view(B, 1, M * self.max_residue_atoms, 3)
        ).norm(
            dim=-1
        )  # [B,N*self.max_residue_atoms,M*self.max_residue_atoms]
        mask_2D = mask_2D * (dist < self.threshold).float()

        h = self.encoder(
                (true_X, bind_S, bind_A), 
                (tgt_X, tgt_S, tgt_A), 
        ) # [B,N+M,self.max_residue_atoms,H]

        bind_h = self.binder_output(h[:, :N]).view(B, N * self.max_residue_atoms, -1)
        tgt_h = self.target_output(h[:, N:]).view(B, M * self.max_residue_atoms, -1)
        if affinity is None:
            loss_mse = torch.tensor(0.0, device=true_X.device)
        else:
            crystal_energy = torch.matmul(bind_h, tgt_h.transpose(1,2))  # [B,N*14,M*14]
            crystal_energy = (crystal_energy * mask_2D).sum(dim=(1,2))  # [B]
            loss_mse = self.mse_loss(crystal_energy, affinity)

        # Perturb
        eps = np.random.uniform(0.1, 1.0, size=B)
        eps = torch.tensor(eps, dtype=torch.float, device=true_X.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        hat_t = torch.randn_like(true_X).to(true_X.device) * eps
        bind_X = true_X + hat_t * atom_mask
        bind_X = bind_X.requires_grad_()

        # Get the contact map for perturbed structure
        mask_2D = (bind_A > 0).float().view(B, N * self.max_residue_atoms, 1) * (tgt_A > 0).float().view(
            B, 1, M * self.max_residue_atoms
        )
        dist = (
            bind_X.view(B, N * self.max_residue_atoms, 1, 3) - tgt_X.view(B, 1, M * self.max_residue_atoms, 3)
        ).norm(
            dim=-1
        )  # [B,N*self.max_residue_atoms,M*self.max_residue_atoms]
        mask_2D = mask_2D * (dist < self.threshold).float()

        # Compute the energy
        h = self.encoder(
            (bind_X, bind_S, bind_A),
            (tgt_X, tgt_S, tgt_A),
        )  # [B,N+M,self.max_residue_atoms,H]
        bind_h = self.binder_output(h[:, :N]).view(B, N * self.max_residue_atoms, -1)
        tgt_h = self.target_output(h[:, N:]).view(B, M * self.max_residue_atoms, -1)
        perturbed_energy = torch.matmul(bind_h, tgt_h.transpose(1, 2))  # [B,N*self.max_residue_atoms,M*self.max_residue_atoms]
        perturbed_energy = (perturbed_energy * mask_2D).sum(dim=(1, 2))  # [B]

        # Compute the DSM loss
        f_bind = torch.autograd.grad(perturbed_energy.sum(), bind_X, create_graph=True, retain_graph=True)[0]
        if self.eps_scaling:
            loss = F.mse_loss(hat_t / eps, f_bind * eps, reduction='none')
        else:
            loss = F.mse_loss(hat_t / eps**2, f_bind, reduction='none')
        loss_dsm = (loss * atom_mask).sum() / atom_mask.sum()
        return loss_dsm, loss_mse

    def training_step(
        self,
        batch, batch_idx: int
    ) -> torch.Tensor:
        binder, target, affinity = batch
        loss_dsm, loss_mse = self._step(
            binder=binder,
            target=target,
            affinity=affinity,
            )
        
        if self.cfg_model.use_mtl_loss:
            loss = self.mtl_loss(loss_dsm, loss_mse)
            self.log_dict({
                'train/loss_dsm': loss_dsm,
                'train/loss_mse': loss_mse,
                'train/loss': loss,
                'precision_dsm': self.mtl_loss.get_precisions()[0],
                'precision_mse': self.mtl_loss.get_precisions()[1]
                },
                prog_bar=True,
                on_step=True,
            )
        else:
            loss = self.loss_ratio * loss_dsm + loss_mse
            self.log_dict({
                'train/loss_dsm': loss_dsm,
                'train/loss_mse': loss_mse,
                'train/loss': loss,
                },
                prog_bar=True,
                on_step=True,
            )
        return loss

    def _validation_step(
        self,
        batch,
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        binder, target, affinity = batch
        with torch.enable_grad():  # DualBind model requires grad on energy function
            loss_dsm, loss_mse = self._step(
                binder=binder,
                target=target,
                affinity=affinity,
                )
        loss = self.loss_ratio * loss_dsm + loss_mse
        energy = self.forward(binder, target)

        outputs = {
            'loss_dsm': loss_dsm,
            'loss_mse': loss_mse,
            'loss': loss,
            'energy': energy,
            'affinity': affinity,
        }
        outputs = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in outputs.items()}
        return outputs

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        outputs = self._validation_step(batch, batch_idx)
        self._validation_outputs.append(outputs)
        return outputs
    
    def _test_step(
        self,
        batch,
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        binder, target, affinity = batch
        energy = self.forward(binder, target)

        outputs = {
            'energy': energy,
            'affinity': affinity,
        }
        outputs = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in outputs.items()}
        return outputs

    def test_step(
        self,
        batch,
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        outputs = self._test_step(batch, batch_idx)
        self._test_outputs.append(outputs)
        return outputs
    
    def predict_step(
        self,
        batch,
        batch_idx: int
    ) -> torch.Tensor:
        binder, target, _ = batch  # skip affinity
        output = self.forward(binder, target)
        return output

    def _on_epoch_end(self, outputs: List[Dict[str, Any]], stage: str) -> None:
        outputs_metrics: Dict[str, torch.Tensor] = {}
        energy: List[torch.Tensor] = []
        affinity: List[torch.Tensor] = []

        for output in outputs:
            if 'energy' in output:
                energy.append(output.pop('energy'))
            if 'affinity' in output:
                affinity.append(output.pop('affinity'))
            for k, v in output.items():
                if k in outputs_metrics:
                    outputs_metrics[k].append(v)
                else:
                    outputs_metrics[k] = [v]

        outputs_metrics = {f'{stage}/{k}': torch.stack(v).mean() for k, v in outputs_metrics.items()}

        if affinity[0] is not None:  # ASSUMPTION: either all affinity entries are None or tensor
            energy = torch.cat(energy, dim=-1).flatten()
            affinity = torch.cat(affinity, dim=-1).flatten()
            outputs_metrics[f'{stage}/src'] = spearman_corrcoef(energy, affinity)
            outputs_metrics[f'{stage}/pcc'] = pearson_corrcoef(energy, affinity)
            outputs_metrics[f'{stage}/r2'] = r2_score(energy, affinity)
            rmse = torch.sqrt(F.mse_loss(energy, affinity))
            outputs_metrics[f'{stage}/rmse'] = rmse
            
            # Adjust predictions for non-binders
            non_binder_mask = affinity == -3.0
            energy = torch.where(non_binder_mask & (energy >= -3.0), torch.tensor(-3.0, device=energy.device), energy)

            # Calculate Spearman and Pearson correlation coefficients
            outputs_metrics[f'{stage}/qualified_src'] = spearman_corrcoef(energy, affinity)
            outputs_metrics[f'{stage}/qualified_pcc'] = pearson_corrcoef(energy, affinity)
            outputs_metrics[f'{stage}/qualified_r2'] = r2_score(energy, affinity)
            qualified_rmse = torch.sqrt(F.mse_loss(energy, affinity))
            outputs_metrics[f'{stage}/qualified_rmse'] = qualified_rmse

        self.log_dict(outputs_metrics, on_epoch=True, on_step=False, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        outputs: List[Dict[str, Any]] = self.all_gather(self._validation_outputs) if self.trainer.world_size > 1 else self._validation_outputs
        self._on_epoch_end(outputs=outputs, stage='val')
        self._validation_outputs.clear()

    def on_test_epoch_end(self) -> None:
        outputs: List[Dict[str, Any]] = self.all_gather(self._test_outputs) if self.trainer.world_size > 1 else self._test_outputs
        self._on_epoch_end(outputs=outputs, stage='test')
        self._test_outputs.clear()
