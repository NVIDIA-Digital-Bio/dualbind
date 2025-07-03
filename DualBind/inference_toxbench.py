# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Entry point to DualBind inference on ToxBench dataset.

Typical usage example:
    cd DualBind
    python inference_toxbench.py

Notes:
    See conf/inference_toxbench.yaml for hyperparameters and settings for this script.
"""

from pathlib import Path

import pandas as pd
import hydra
from omegaconf.omegaconf import OmegaConf, DictConfig
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef, r2_score
import pytorch_lightning as pl

from data.dataset import ToxBenchDataset
from model.dualbind_model import DualBind


def multiply(*args): # enable global_batch_size calculation
    product = 1
    for arg in args:
        product *= arg
    return product

def filepath_to_name(filepath: str):  # enable output csv naming
    return Path(filepath).stem

def gather_predictions(predictions):
    """Gather predictions from all GPUs if using multiple devices."""
    if dist.is_initialized():
        gathered_preds = [torch.zeros_like(predictions) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_preds, predictions)
        return torch.cat(gathered_preds)
    return predictions


OmegaConf.register_new_resolver("multiply", multiply)
OmegaConf.register_new_resolver("filepath_to_name", filepath_to_name)


@hydra.main(config_path='conf', config_name='inference_toxbench')
def main(cfg: DictConfig) -> None:
    """
    This is the main function conducting data loading and model inference for DualBind.
    """
    print("\n\n************** Experiment Configuration ***********")
    print(f"\n{OmegaConf.to_yaml(cfg)}")

    if cfg.prediction_csv is None:
        raise ValueError('Must provide prediction_csv in config.')
    if cfg.ckpt_path_list is None:
        raise ValueError('Must provide ckpt_path_list in config.')
    
    if cfg.trainer.num_nodes * cfg.trainer.devices > 1:
        raise ValueError('Only single device is supported for inference')

    if cfg.seed:
        pl.seed_everything(cfg.seed, workers=True)

    print("************** Loading Dataset ***********")
    predict_dataset = ToxBenchDataset(
        csv_file=cfg.data.raw_data_path_csv,
        aa_size=cfg.model.aa_size,
        max_residue_atoms=cfg.model.max_residue_atoms,
        prefix_path=cfg.data.data_prefix_path,
        split='test'
    )
    predict_loader = DataLoader(
        predict_dataset, cfg.data.batch_size, collate_fn=predict_dataset.pl_collate_fn, shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    print("************** Collecting Ground Truth Labels ***********")
    all_targets = []
    for batch in predict_loader:
        _, _, ground_truth = batch
        all_targets.append(ground_truth)
    all_targets = torch.cat(all_targets)

    ensemble_predictions = []
    for ckpt in cfg.ckpt_path_list:
        model = DualBind(cfg.model)
        print("************** Setup Trainer for Inference ***********")
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=cfg.trainer.devices,
            num_nodes=cfg.trainer.num_nodes,
            strategy=cfg.trainer.strategy,
            logger=None,
            precision=cfg.trainer.precision,
        )

        print(f"************** Starting Inference for Checkpoint: {ckpt} ***********")
        predictions = trainer.predict(
                model=model,
                dataloaders=predict_loader,
                return_predictions=True,
                ckpt_path=ckpt,
            )

        predictions = torch.cat(predictions)
        # Gather predictions from all GPUs (if running in distributed mode) 
        # [TODO] fix the bug raised when using multiple devices
        predictions = gather_predictions(predictions)
        ensemble_predictions.append(predictions)
        

    # Average predictions across checkpoints
    all_preds = torch.stack(ensemble_predictions).mean(dim=0).cpu()
    all_targets = all_targets.cpu()
    assert all_preds.shape == all_targets.shape, "Mismatch between predictions and targets"

    print("************** Computing Metrics (Qulify Predictions as Needed) ***********")
    outputs_metrics = {}
    outputs_metrics['src'] = spearman_corrcoef(all_preds, all_targets).item()
    outputs_metrics['pcc'] = pearson_corrcoef(all_preds, all_targets).item()
    outputs_metrics['r2'] = r2_score(all_preds, all_targets).item()
    rmse = torch.sqrt(F.mse_loss(all_preds, all_targets))
    outputs_metrics['rmse'] = rmse.item()

    # Qulify predictions for non-binders
    all_raw_preds = all_preds
    non_binder_mask = all_targets == -3.0
    all_preds = torch.where(non_binder_mask & (all_preds >= -3.0), torch.tensor(-3.0, device=all_preds.device), all_preds)

    outputs_metrics['qualified_src'] = spearman_corrcoef(all_preds, all_targets).item()
    outputs_metrics['qualified_pcc'] = pearson_corrcoef(all_preds, all_targets).item()
    outputs_metrics['qualified_r2'] = r2_score(all_preds, all_targets).item()
    qualified_rmse = torch.sqrt(F.mse_loss(all_preds, all_targets))
    outputs_metrics['qualified_rmse'] = qualified_rmse.item()

    all_raw_preds_np = all_raw_preds.tolist()
    all_preds_np = all_preds.tolist()
    all_targets_np = all_targets.tolist()
    predictions_df = pd.DataFrame({
        'raw predictions': all_raw_preds_np,
        'predictions': all_preds_np,
        'ground_truth': all_targets_np
    })

    metrics_df = pd.DataFrame(outputs_metrics, index=[0])
    combined_df = pd.concat([predictions_df, metrics_df], ignore_index=True)
    combined_df.to_csv(cfg.prediction_csv, index=False)
    print(f"Predictions and ground truth saved to {cfg.prediction_csv}")

if __name__ == '__main__':
    main()
