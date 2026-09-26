#!/usr/bin/env python3
"""
OTT3R Training Script

Usage:
    python ott3r/train.py --config configs/ott3r_base.yaml
"""

import os
import sys
import time
import psutil
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger
import yaml
from tqdm import tqdm
import numpy as np

# Add the project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "pi3"))

from ott3r.student.distillation_module import OTT3RLitModule
from ott3r.data.cached_sample_dataset import CachedSampleDataset, create_cached_samples_collate_fn
from torch.utils.data import DataLoader


class MemoryTracker:
    """Track GPU and system memory usage throughout training."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.start_time = time.time()
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_file, 'w') as f:
            f.write("timestamp,elapsed_sec,gpu_memory_gb,gpu_reserved_gb,system_memory_gb,step\n")

    def log_memory(self, step: int = 0):
        """Log current memory usage."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory = info.used / 1024**3
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            except ImportError:
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        else:
            gpu_memory = gpu_reserved = 0.0

        system_memory = psutil.virtual_memory().used / 1024**3

        with open(self.log_file, 'a') as f:
            f.write(f"{current_time},{elapsed:.1f},{gpu_memory:.2f},{gpu_reserved:.2f},{system_memory:.2f},{step}\n")

        return {
            'gpu_memory_gb': gpu_memory,
            'gpu_reserved_gb': gpu_reserved,
            'system_memory_gb': system_memory,
            'elapsed_sec': elapsed
        }


class MemoryCallback(pl.Callback):
    """Lightning callback for memory tracking."""

    def __init__(self, tracker: MemoryTracker, alert_threshold_gb: float = 20.0):
        self.tracker = tracker
        self.alert_threshold = alert_threshold_gb
        self.step_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step_count += 1

        if self.step_count % 100 == 0:
            memory_info = self.tracker.log_memory(self.step_count)

            if memory_info['gpu_memory_gb'] > self.alert_threshold:
                print(f"\n  HIGH MEMORY USAGE: {memory_info['gpu_memory_gb']:.1f}GB GPU memory")


def create_datamodule(config: Dict[str, Any]) -> pl.LightningDataModule:
    """Create data module from teacher cache."""
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    model_config = config.get('model', {})

    class Pi3CachedDataModule(pl.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.cache_dir = data_config.get('cache_dir', 'caches/teacher_cache')
            self.batch_size = training_config.get('batch_size', 1)
            self.num_workers = data_config.get('num_workers', 8)
            self.num_views = model_config.get('num_views', 20)

        def setup(self, stage: Optional[str] = None):
            if stage == 'fit' or stage is None:
                self.train_dataset = CachedSampleDataset(
                    teacher_cache_dir=self.cache_dir,
                    image_size=512,
                    num_views=self.num_views,
                    max_samples=None,
                )

        def train_dataloader(self):
            pin_memory = data_config.get('pin_memory', True)
            persistent_workers = data_config.get('persistent_workers', True) and self.num_workers > 0
            prefetch_factor = data_config.get('prefetch_factor', 2) if self.num_workers > 0 else None

            print(f"Creating DataLoader: workers={self.num_workers}, batch={self.batch_size}, "
                  f"views={self.num_views}, pin_memory={pin_memory}")

            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                collate_fn=create_cached_samples_collate_fn(),
                drop_last=True
            )

    return Pi3CachedDataModule()


def create_callbacks(config: Dict[str, Any], memory_tracker: MemoryTracker):
    """Create callbacks with monitoring."""
    callbacks = []

    # Memory tracking
    memory_callback = MemoryCallback(
        memory_tracker,
        alert_threshold_gb=config.get('memory', {}).get('max_memory_gb', 40.0)
    )
    callbacks.append(memory_callback)

    # Model checkpointing - periodic saves based on config
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.get('output_dir', 'checkpoints/ott3r'),
        filename='ott3r-{epoch:02d}-{step:06d}-{train/loss:.4f}',
        monitor='train/loss',
        mode='min',
        save_top_k=config.get('checkpointing', {}).get('save_top_k', 5),
        save_last=True,
        every_n_epochs=config.get('checkpointing', {}).get('save_every_n_epochs', 10),
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )
    callbacks.append(checkpoint_callback)

    # Separate callback to save last.ckpt every epoch (for resume)
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=config.get('output_dir', 'checkpoints/ott3r'),
        save_top_k=0,  # Don't save top-k, only last
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )
    callbacks.append(last_checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Early stopping (optional)
    if config.get('training', {}).get('early_stop_patience'):
        early_stop = EarlyStopping(
            monitor='train/loss',
            patience=config['training']['early_stop_patience'],
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop)

    return callbacks


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='OTT3R Training')
    parser.add_argument('--config', type=str, default='configs/ott3r_base.yaml',
                       help='Training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint (restores optimizer/scheduler state)')
    parser.add_argument('--finetune-from', type=str, default=None,
                       help='Initialize weights from checkpoint for fine-tuning (fresh optimizer/scheduler)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Setup only, no training')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("OTT3R TRAINING")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Output: {config.get('output_dir', 'checkpoints/ott3r')}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print()

    # Set random seed
    pl.seed_everything(config.get('seed', 42))

    # Setup logging
    log_dir = Path(config.get('logging', {}).get('log_dir', 'logs/ott3r'))
    log_dir.mkdir(parents=True, exist_ok=True)
    memory_tracker = MemoryTracker(log_dir / 'memory_usage.csv')

    # Create data module
    print("Creating data module...")
    try:
        data_module = create_datamodule(config)
        data_module.setup('fit')

        train_loader = data_module.train_dataloader()
        data_size = len(train_loader.dataset)
        print(f"Training data size: {data_size} samples")

    except Exception as e:
        print(f"Error creating data module: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Calculate total training steps
    training_config = config.get('training', {})
    batch_size = training_config.get('batch_size', 1)
    accumulate_grad_batches = training_config.get('accumulate_grad_batches', 1)
    max_epochs = training_config.get('max_epochs', 100)

    steps_per_epoch = data_size / (batch_size * accumulate_grad_batches)
    calculated_max_steps = int(steps_per_epoch * max_epochs)
    print(f"Calculated max_steps: {calculated_max_steps:,} ({steps_per_epoch:.0f} steps/epoch × {max_epochs} epochs)")

    # Create model
    print("Creating student model...")
    model_config = config.get('model', {})
    loss_config = config.get('loss', {})
    memory_config = config.get('memory', {})
    finetuning_config = config.get('finetuning', {})

    # Use finetuning-specific LR/warmup if --finetune-from is provided and config exists
    if args.finetune_from and finetuning_config:
        effective_lr = finetuning_config.get('learning_rate', training_config.get('learning_rate', 1e-4))
        effective_warmup = finetuning_config.get('warmup_steps', training_config.get('warmup_steps', 1000))
        print(f"Fine-tuning config: lr={effective_lr}, warmup_steps={effective_warmup}")
    else:
        effective_lr = training_config.get('learning_rate', 1e-4)
        effective_warmup = training_config.get('warmup_steps', 1000)

    lit_module = OTT3RLitModule(
        # Loss config
        include_normal_loss=loss_config.get('include_normal_loss', True),
        include_camera_loss=loss_config.get('include_camera_loss', True),
        include_conf_loss=loss_config.get('include_conf_loss', True),
        camera_loss_weight=loss_config.get('camera_loss_weight', 0.1),
        camera_baseline_weighting=loss_config.get('camera_baseline_weighting', False),
        conf_loss_weight=loss_config.get('conf_loss_weight', 0.05),
        conf_loss_type=loss_config.get('conf_loss_type', 'bce'),
        # Training config (use effective values for fine-tuning)
        learning_rate=effective_lr,
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_steps=effective_warmup,
        max_steps=calculated_max_steps,
        min_lr=training_config.get('min_lr', 1e-6),
        # Model config
        enc_embed_dim=model_config.get('enc_embed_dim', 384),
        dec_embed_dim=model_config.get('dec_embed_dim', 384),
        dec_depth=model_config.get('dec_depth', 8),
        dec_num_heads=model_config.get('dec_num_heads', 6),
        num_register_tokens=model_config.get('num_register_tokens', 5),
        load_pretrained_encoder=model_config.get('load_pretrained_encoder', True),
        encoder_type=model_config.get('encoder_type', 'dune'),
        freeze_encoder=model_config.get('freeze_encoder', False),
        # Task decoder config (points, conf)
        task_dec_dim=model_config.get('task_dec_dim', 384),
        task_dec_depth=model_config.get('task_dec_depth', 3),
        task_dec_heads=model_config.get('task_dec_heads', 6),
        # Camera decoder config
        camera_dec_dim=model_config.get('camera_dec_dim', 384),
        camera_dec_heads=model_config.get('camera_dec_heads', 6),
        camera_dec_depth=model_config.get('camera_dec_depth', 3),
        camera_out_dim=model_config.get('camera_out_dim', 256),
        # Memory config
        teacher_cache_dir=config.get('data', {}).get('cache_dir', 'caches/teacher_cache'),
        memory_cleanup_freq=memory_config.get('memory_cleanup_freq', 50),
        log_memory_usage=memory_config.get('log_memory_usage', True),
        gpu_memory_threshold=memory_config.get('gpu_memory_threshold', 95.0),
    )

    # Print model statistics
    model_stats = lit_module.student.get_model_stats()

    # Teacher params (hardcoded - π³ model)
    teacher_total = 958_696_732
    teacher_encoder = 304_371_712

    print(f"\n{'='*60}")
    print(f"MODEL PARAMETERS")
    print(f"{'='*60}")
    print(f"{'Component':<20} {'Teacher (π³)':<18} {'Student':<18}")
    print(f"{'-'*60}")
    print(f"{'Encoder':<20} {teacher_encoder/1e6:>14.2f}M   {model_stats['component_breakdown']['encoder']:>14.2f}M")
    print(f"{'Main Decoder':<20} {453.55:>14.2f}M   {model_stats['component_breakdown']['main_decoder']:>14.2f}M")
    print(f"{'Point Decoder':<20} {66.13:>14.2f}M   {model_stats['component_breakdown']['point_decoder']:>14.2f}M")
    print(f"{'Conf Decoder':<20} {66.13:>14.2f}M   {model_stats['component_breakdown']['conf_decoder']:>14.2f}M")
    print(f"{'Camera Decoder':<20} {65.60:>14.2f}M   {model_stats['component_breakdown']['camera_decoder']:>14.2f}M")
    print(f"{'-'*60}")
    print(f"{'TOTAL':<20} {teacher_total/1e6:>14.2f}M   {model_stats['total_parameters_M']:>14.2f}M")
    print(f"{'Compression':<20} {'':>14}   {teacher_total/model_stats['total_parameters']:>13.1f}x")
    print(f"{'='*60}")
    print(f"Trainable: {model_stats['trainable_parameters']:,} ({model_stats['trainable_ratio']:.1%})")

    # Load weights for fine-tuning (fresh optimizer/scheduler, only model weights)
    if args.finetune_from:
        print(f"\n{'='*60}")
        print("FINE-TUNING MODE")
        print(f"{'='*60}")
        print(f"Loading weights from: {args.finetune_from}")

        finetune_ckpt = torch.load(args.finetune_from, map_location='cpu', weights_only=False)

        # Extract only student model weights from checkpoint
        if 'state_dict' in finetune_ckpt:
            # Lightning checkpoint format
            state_dict = {}
            for k, v in finetune_ckpt['state_dict'].items():
                if k.startswith('student.'):
                    new_key = k.replace('student.', '', 1)
                    state_dict[new_key] = v

            if not state_dict:
                print("WARNING: No 'student.*' keys found in checkpoint, trying direct load...")
                state_dict = finetune_ckpt['state_dict']
        else:
            # Direct state dict
            state_dict = finetune_ckpt

        # Load weights into student model
        missing, unexpected = lit_module.student.load_state_dict(state_dict, strict=False)

        print(f"Source checkpoint epoch: {finetune_ckpt.get('epoch', 'unknown')}")
        print(f"Loaded {len(state_dict)} weight tensors")
        if missing:
            print(f"Missing keys: {len(missing)} (expected for fresh training)")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
        print(f"Optimizer/scheduler: FRESH (not restored)")
        print(f"Starting from epoch: 0")
        print(f"{'='*60}")

    if args.dry_run:
        print("\nDry run complete - setup successful!")
        return 0

    # Create logger
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name=config.get('logging', {}).get('experiment_name', 'ott3r')
    )

    # Create trainer
    num_gpus = training_config.get('num_gpus', 1)
    gpu_id = training_config.get('gpu_id', 0)

    if num_gpus > 1:
        os.environ['NCCL_P2P_DISABLE'] = '1'

    strategy = 'ddp_find_unused_parameters_true' if num_gpus > 1 else 'auto'
    devices = [gpu_id] if num_gpus == 1 else num_gpus

    print(f"\nTraining configuration:")
    print(f"  GPUs: {devices}")
    print(f"  Strategy: {strategy}")
    print(f"  Precision: {training_config.get('precision', 'bf16-mixed')}")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=devices,
        strategy=strategy,
        precision=training_config.get('precision', 'bf16-mixed'),
        gradient_clip_val=training_config.get('max_grad_norm', 1.0),
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=training_config.get('log_every_n_steps', 50),
        enable_checkpointing=True,
        enable_progress_bar=True,
        callbacks=create_callbacks(config, memory_tracker),
        logger=csv_logger,
        deterministic=False,
        benchmark=True,
    )

    # Start training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    start_time = time.time()
    memory_tracker.log_memory(0)

    try:
        if args.resume:
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
            print(f"RESUMING from checkpoint:")
            print(f"  Checkpoint: {args.resume}")
            print(f"  Epoch: {ckpt.get('epoch', 'Unknown')}")
            print(f"  Global step: {ckpt.get('global_step', 'Unknown')}")
            print()

        trainer.fit(
            model=lit_module,
            datamodule=data_module,
            ckpt_path=args.resume
        )

        end_time = time.time()
        training_duration = end_time - start_time

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED!")
        print("=" * 80)
        print(f"Total training time: {training_duration:.1f}s ({training_duration/60:.1f}m)")

        if trainer.current_epoch > 0:
            print(f"Average time per epoch: {training_duration/trainer.current_epoch:.1f}s")

        # Save final model
        final_path = Path(config.get('output_dir', 'checkpoints/ott3r')) / 'final_model.ckpt'
        trainer.save_checkpoint(final_path)
        print(f"Final model saved: {final_path}")

        final_memory = memory_tracker.log_memory(-1)
        print(f"Final GPU memory: {final_memory['gpu_memory_gb']:.1f}GB")
        print(f"Memory log: {memory_tracker.log_file}")
        print(f"Training logs: {csv_logger.log_dir}")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
