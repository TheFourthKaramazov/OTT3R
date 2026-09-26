"""
OTT3R Training Module

PyTorch Lightning module for knowledge distillation training.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
import lightning.pytorch as pl

# Add external pi3 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "external" / "pi3"))

from ott3r.student.model import OTT3R
from ott3r.student.loss import OTT3RLoss


class OTT3RLitModule(pl.LightningModule):
    """Lightning module for OTT3R knowledge distillation training."""

    def __init__(
        self,
        # Loss config
        include_normal_loss: bool = True,
        include_camera_loss: bool = True,
        include_conf_loss: bool = True,
        camera_loss_weight: float = 0.1,
        camera_baseline_weighting: bool = False,
        conf_loss_weight: float = 0.05,
        conf_loss_type: str = "bce",
        # Training config
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        min_lr: float = 1e-6,
        # Model config
        enc_embed_dim: int = 384,
        dec_embed_dim: int = 384,
        dec_depth: int = 8,
        dec_num_heads: int = 6,
        num_register_tokens: int = 5,
        load_pretrained_encoder: bool = True,
        # Task decoder config (points, conf)
        task_dec_dim: int = 384,
        task_dec_depth: int = 3,
        task_dec_heads: int = 6,
        # Camera decoder config
        camera_dec_dim: int = 384,
        camera_dec_heads: int = 6,
        camera_dec_depth: int = 3,
        camera_out_dim: int = 256,
        # Encoder config
        encoder_type: str = "dune",  # "dune" or "dinov2"
        freeze_encoder: bool = False,  # Freeze encoder for fine-tuning
        # Memory management
        teacher_cache_dir: str = "caches/teacher_cache",
        memory_cleanup_freq: int = 50,
        log_memory_usage: bool = True,
        gpu_memory_threshold: float = 95.0,
    ):
        super().__init__()

        # Store config
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.min_lr = float(min_lr)

        self.teacher_cache_dir = teacher_cache_dir
        self.memory_cleanup_freq = memory_cleanup_freq
        self.log_memory_usage = log_memory_usage
        self.gpu_memory_threshold = gpu_memory_threshold

        # Save hyperparameters
        self.save_hyperparameters()

        # Build student model
        self.student = OTT3R(
            enc_embed_dim=enc_embed_dim,
            dec_embed_dim=dec_embed_dim,
            dec_depth=dec_depth,
            dec_num_heads=dec_num_heads,
            num_register_tokens=num_register_tokens,
            load_pretrained_encoder=load_pretrained_encoder,
            encoder_type=encoder_type,
            freeze_encoder=freeze_encoder,
            # Task decoder config (points, conf)
            task_dec_dim=task_dec_dim,
            task_dec_depth=task_dec_depth,
            task_dec_heads=task_dec_heads,
            # Camera decoder config
            camera_dec_dim=camera_dec_dim,
            camera_dec_heads=camera_dec_heads,
            camera_dec_depth=camera_dec_depth,
            camera_out_dim=camera_out_dim,
        )

        # Build loss function
        self.distill_loss = OTT3RLoss(
            include_normal_loss=include_normal_loss,
            include_camera_loss=include_camera_loss,
            include_conf_loss=include_conf_loss,
            camera_loss_weight=camera_loss_weight,
            camera_baseline_weighting=camera_baseline_weighting,
            conf_loss_weight=conf_loss_weight,
            conf_loss_type=conf_loss_type,
        )

        # Memory management
        self._step_count = 0

        # Throughput tracking
        self._epoch_start_time = None
        self._epoch_samples = 0

    def configure_optimizers(self):
        """Configure optimizer with warmup + cosine annealing."""
        trainable_params = [p for p in self.student.parameters() if p.requires_grad]

        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )

        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")

        # Warmup + Cosine Annealing
        if self.warmup_steps > 0:
            def warmup_lambda(step):
                if step < self.warmup_steps:
                    return float(step) / float(max(1, self.warmup_steps))
                return 1.0

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_steps - self.warmup_steps,
                eta_min=self.min_lr
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps]
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_steps,
                eta_min=self.min_lr
            )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def on_train_epoch_start(self):
        """Track epoch start time for throughput calculation."""
        self._epoch_start_time = time.time()
        self._epoch_samples = 0

    def on_train_epoch_end(self):
        """Log throughput and clean up memory."""
        import gc
        import psutil

        # Log throughput
        if self._epoch_start_time is not None:
            epoch_time = time.time() - self._epoch_start_time

            if self._epoch_samples > 0 and epoch_time > 0:
                samples_per_sec = self._epoch_samples / epoch_time
                self.log('train/samples_per_sec', samples_per_sec, sync_dist=True)
                self.log('train/epoch_time_min', epoch_time / 60, sync_dist=True)

                if self.trainer.is_global_zero:
                    print(f"\nEpoch {self.current_epoch} throughput: {samples_per_sec:.2f} samples/sec ({epoch_time/60:.1f} min)")

        # Memory cleanup
        if hasattr(self, '_step_count'):
            system_memory = psutil.virtual_memory()
            print(f"\n[Epoch {self.current_epoch} End] Pre-cleanup RAM: {system_memory.used/1024**3:.1f}GB ({system_memory.percent:.1f}%)")

        for _ in range(3):
            gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect(2)

        if hasattr(self, '_step_count'):
            system_memory = psutil.virtual_memory()
            print(f"[Epoch {self.current_epoch} End] Post-cleanup RAM: {system_memory.used/1024**3:.1f}GB ({system_memory.percent:.1f}%)\n")

        if self.log_memory_usage:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"Epoch end - GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            self.log("memory/epoch_end_allocated_gb", memory_allocated, on_epoch=True)
            self.log("memory/epoch_end_reserved_gb", memory_reserved, on_epoch=True)

    def forward(self, images):
        """Forward pass through student model."""
        return self.student(images)

    def training_step(self, batch, batch_idx):
        # Validate batch format
        if not isinstance(batch, dict) or 'images' not in batch:
            raise ValueError(f"Expected batch with 'images' key, got: {batch.keys() if isinstance(batch, dict) else type(batch)}")

        if 'teacher_data' not in batch:
            raise ValueError("Expected batch with 'teacher_data' key from CachedSampleDataset")

        images = batch['images']  # [B, N, C, H, W]
        teacher_data = batch['teacher_data']  # Dict with batched teacher outputs
        masks = batch['masks']  # [B, N, H, W]

        batch_size = images.shape[0]

        # Student forward pass - model takes [B, N, C, H, W] directly
        student_output = self.forward(images)

        # Format student predictions for loss
        pred = {
            'local_points': student_output['local_points'],  # [B, N, H, W, 3]
            'conf': student_output['conf'],  # [B, N, H, W, 1]
            'camera_poses': student_output['camera_poses'],  # [B, N, 4, 4]
            'points': student_output['points'],  # [B, N, H, W, 3] - global (for reference)
        }

        # Format teacher targets for loss
        target = {
            'local_points': teacher_data['local_points'],  # [B, N, H, W, 3]
            'conf': teacher_data['conf'],  # [B, N, H, W, 1]
            'camera_poses': teacher_data['camera_poses'],  # [B, N, 4, 4]
        }

        # Compute distillation loss
        loss, loss_details = self.distill_loss(pred, target, masks)

        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for key, value in loss_details.items():
            if isinstance(value, torch.Tensor):
                self.log(f'train/{key}', value, on_step=True, on_epoch=True)

        # Memory management
        self._step_count += 1

        # Print loss components every N steps (only on rank 0)
        log_interval = 50
        if self._step_count % log_interval == 0 and self.trainer.is_global_zero:
            loss_str = f"[Step {self._step_count}] loss={loss.item():.4f}"
            for key, value in loss_details.items():
                if isinstance(value, torch.Tensor):
                    loss_str += f" | {key}={value.item():.4f}"
            print(loss_str)
        self._monitor_memory()

        # Track samples for throughput
        self._epoch_samples += batch_size

        return loss

    def _monitor_memory(self):
        """Monitor memory usage and trigger cleanup if needed."""
        if self._step_count % 100 != 0:
            return

        try:
            import pynvml
            import psutil

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_gb = info.used / 1024**3
            total_gpu_gb = info.total / 1024**3
            gpu_utilization = (info.used / info.total) * 100

            system_memory = psutil.virtual_memory()
            system_ram_gb = (system_memory.total - system_memory.available) / 1024**3
            system_ram_percent = system_memory.percent

            cpu_percent = psutil.cpu_percent(interval=None)

            print(f"[Step {self._step_count}] GPU: {gpu_memory_gb:.1f}GB ({gpu_utilization:.1f}%), "
                  f"RAM: {system_ram_gb:.1f}GB ({system_ram_percent:.1f}%), CPU: {cpu_percent:.1f}%")

            # Safety checks
            if system_ram_percent > 85.0:
                print(f"\nSTOPPING: HIGH SYSTEM RAM: {system_ram_gb:.1f}GB ({system_ram_percent:.1f}%)")
                self.trainer.should_stop = True
                return

            if gpu_utilization > self.gpu_memory_threshold:
                print(f"\nSTOPPING: HIGH GPU MEMORY: {gpu_memory_gb:.1f}GB / {total_gpu_gb:.1f}GB ({gpu_utilization:.1f}%)")
                raise RuntimeError(f"GPU memory exceeded threshold: {gpu_utilization:.1f}%")

        except ImportError:
            # Fallback if pynvml not available
            pytorch_memory_gb = torch.cuda.memory_allocated() / 1024**3
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            pytorch_utilization = (pytorch_memory_gb / total_memory_gb) * 100

            if pytorch_utilization > self.gpu_memory_threshold:
                raise RuntimeError(f"GPU memory exceeded threshold: {pytorch_utilization:.1f}%")

        # Cleanup at specified frequency
        if self._step_count % self.memory_cleanup_freq == 0:
            torch.cuda.empty_cache()

    @classmethod
    def load_for_inference(cls, checkpoint_path: str):
        """Load model from checkpoint for inference."""
        model = cls.load_from_checkpoint(checkpoint_path)
        model.eval()
        return model


