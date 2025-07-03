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
Entry point to DualBind training on the ToxBench dataset.

Typical usage example:
    cd DualBind
    python train_toxbench.py

Notes:
    See conf/train_toxbench.yaml for hyperparameters and settings for this script.
"""

import os
import hydra
from omegaconf.omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data.dataset import ToxBenchDataset
from model.dualbind_model import DualBind

# enable multiplication in train.yaml for global_batch_size calculation
def multiply(*args):
    product = 1
    for arg in args:
        product *= arg
    return product

OmegaConf.register_new_resolver("multiply", multiply)

class CustomModelCheckpoint(ModelCheckpoint):
    def format_checkpoint_name(self, *args, **kwargs):
        # Get the full path (including directories and filename) from the parent class
        full_path = super().format_checkpoint_name(*args, **kwargs)

        # Find the last occurrence of the path separator and replace it with '_'
        sanitized_full_path = full_path[::-1].replace(os.path.sep[::-1], '_', 1)[::-1]
        
        return sanitized_full_path

@hydra.main(config_path='conf', config_name='train_toxbench', version_base="1.1")
def main(cfg: DictConfig) -> None:
    """
    This is the main function conducting data loading and model training for DualBind training with ToxBench data.
    """
    print("\n\n************** Experiment Configuration ***********")
    print(f"\n{OmegaConf.to_yaml(cfg)}")

    if cfg.seed:
        pl.seed_everything(cfg.seed, workers=True)

    # Data loading
    print("************** Loading Dataset ***********")
    train_dataset = ToxBenchDataset(
        csv_file=cfg.data.raw_data_path_csv,
        aa_size=cfg.model.aa_size,
        max_residue_atoms=cfg.model.max_residue_atoms,
        prefix_path=cfg.data.data_prefix_path,
        split='train'
    )
    train_loader = DataLoader(
        train_dataset, cfg.data.train.batch_size, collate_fn=train_dataset.pl_collate_fn, shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_dataset = ToxBenchDataset(
        csv_file=cfg.data.raw_data_path_csv,
        aa_size=cfg.model.aa_size,
        max_residue_atoms=cfg.model.max_residue_atoms,
        prefix_path=cfg.data.data_prefix_path,
        split='val'
    )
    val_loader = DataLoader(
        val_dataset, cfg.data.val.batch_size, collate_fn=val_dataset.pl_collate_fn, shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    test_dataset = ToxBenchDataset(
        csv_file=cfg.data.raw_data_path_csv,
        aa_size=cfg.model.aa_size,
        max_residue_atoms=cfg.model.max_residue_atoms,
        prefix_path=cfg.data.data_prefix_path,
        split='test'
    )
    test_loader = DataLoader(
        test_dataset, cfg.data.val.batch_size, collate_fn=test_dataset.pl_collate_fn, shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    
    print(f"# Training data: {len(train_dataset)}")
    print(f"# Validation data: {len(val_dataset)}")
    print(f"# Testing data: {len(test_dataset)}")
    # Setup model and potentially use DataParallel
    print("************** Setup Model ***********")
    model = DualBind(cfg.model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'# Model parameters: {total_params}')

    # Setup training
    print("************** Setup Training ***********")
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        strategy=cfg.trainer.strategy,
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        default_root_dir=cfg.trainer.default_root_dir,
        enable_checkpointing=True,
        deterministic=cfg.trainer.deterministic,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        logger=WandbLogger(
            project=cfg.trainer.wandb_logger.project,
            name=cfg.trainer.wandb_logger.name,
        ) if cfg.trainer.wandb_logger.create_wandb_logger else None,
        callbacks=[
            CustomModelCheckpoint(
                filename=f'best-model-{{epoch}}-{{{cfg.trainer.callbacks.model_checkpoint.monitor}:.4f}}',
                **cfg.trainer.callbacks.model_checkpoint,
            ),
            EarlyStopping(**cfg.trainer.callbacks.early_stopping),
        ]
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.ckpt_path,
    )
    trainer.test(model=model,
                 dataloaders=test_loader,
                 ckpt_path="best"
                 )

if __name__ == '__main__':
    main()
