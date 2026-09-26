"""
OTT3R Teacher Cache Generator

Generates supervision signals from π³ teacher model:
- xyz_local: Local 3D point maps [N, H, W, 3]
- conf: Confidence scores [N, H, W, 1]
- camera_poses: SE(3) poses [N, 4, 4]

Usage:
    python ott3r/teacher/ott3r_generator.py --manifest processed_data/images/manifest.csv
    python ott3r/teacher/ott3r_generator.py --num-gpus 2 --num-workers 4
"""

import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import time
import pandas as pd
import sys
import os
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import datetime

# Add external pi3 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "external" / "pi3"))

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pi3.models.pi3 import Pi3
from pi3.utils.geometry import depth_edge
from ott3r.teacher.rle_helpers import encode_rle, decode_rle

# Optional: open3d for PLY export
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SampleInfo:
    """Information about a single sample to process."""
    dataset: str
    scene_id: str
    sample_idx: int
    num_samples: int
    total_views: int
    image_paths: List[str]
    cache_dir: Path
    use_strided: bool
    hard_ratio: Optional[float]


# =============================================================================
# Dataset for Parallel Loading
# =============================================================================

class SampleDataset(Dataset):
    """Dataset that loads images for samples in parallel."""

    def __init__(self, samples: List[SampleInfo], target_size=(224, 518)):
        self.samples = samples
        self.target_h, self.target_w = target_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and preprocess images
        imgs, load_success = self._load_images(sample.image_paths)

        return {
            'idx': idx,
            'sample': sample,
            'imgs': imgs,
            'load_success': load_success
        }

    def _load_images(self, image_paths):
        """Load images for a sample."""
        tensor_list = []

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                # Resize to target (W, H)
                resized = img.resize((self.target_w, self.target_h), Image.Resampling.LANCZOS)
                tensor = self.to_tensor(resized)  # [3, H, W] in [0, 1]
                tensor_list.append(tensor)
            except Exception as e:
                # Return None on failure
                return None, False

        if not tensor_list:
            return None, False

        imgs = torch.stack(tensor_list, dim=0)  # [N, 3, H, W]
        return imgs, True


def collate_samples(batch):
    """Custom collate that handles variable sample info."""
    return batch[0]  # We process one sample at a time


# =============================================================================
# Sampling Functions (unchanged from export_pi3.py)
# =============================================================================

def sample_views_temporally(group, max_views=20, seed=42, sample_id=0):
    """Sample exactly max_views (20) CONSECUTIVE views from group."""
    total_views = len(group)

    if 'sample_id' in group.columns:
        group_sorted = group.sort_values('sample_id').reset_index(drop=True)
    else:
        group_sorted = group.reset_index(drop=True)

    np.random.seed(seed + sample_id)

    if total_views >= max_views:
        start_idx = sample_id * max_views
        if start_idx + max_views > total_views:
            start_idx = max(0, total_views - max_views)
        indices = np.arange(start_idx, start_idx + max_views)
    else:
        indices = np.random.choice(total_views, max_views, replace=True)
        indices = np.sort(indices)

    sampled = group_sorted.iloc[indices].reset_index(drop=True)
    return sampled


def sample_views_strided(group, max_views=20, seed=42, sample_id=0):
    """Sample views spread across entire sequence (wide-baseline)."""
    total_views = len(group)

    if 'sample_id' in group.columns:
        group_sorted = group.sort_values('sample_id').reset_index(drop=True)
    else:
        group_sorted = group.reset_index(drop=True)

    np.random.seed(seed + sample_id * 1000 + 12345)

    if total_views <= max_views:
        indices = np.sort(np.random.choice(total_views, max_views, replace=True))
    else:
        base = np.linspace(0, total_views - 1, max_views)
        bin_width = (total_views - 1) / (max_views - 1) if max_views > 1 else 0
        jitter = np.random.uniform(-0.35 * bin_width, 0.35 * bin_width, max_views)
        indices = np.round(base + jitter).astype(int)
        indices = np.clip(indices, 0, total_views - 1)

        for retry in range(10):
            unique_indices = np.unique(indices)
            if len(unique_indices) >= max_views:
                indices = np.sort(unique_indices)[:max_views]
                break
            jitter = np.random.uniform(-0.4 * bin_width, 0.4 * bin_width, max_views)
            indices = np.round(base + jitter).astype(int)
            indices = np.clip(indices, 0, total_views - 1)

        indices = np.sort(indices)[:max_views]

        if len(indices) < max_views:
            missing = max_views - len(indices)
            extra = np.linspace(0, total_views - 1, missing + 2)[1:-1].astype(int)
            indices = np.sort(np.concatenate([indices, extra]))[:max_views]

    sampled = group_sorted.iloc[indices].reset_index(drop=True)
    return sampled


def determine_samples_per_scene(total_views, max_samples=100, required_views_per_sample=20):
    """Calculate how many 20-view samples can be created from a scene."""
    if total_views < 2:
        return 0
    num_samples = int(total_views / required_views_per_sample)
    return min(max_samples, num_samples)


# =============================================================================
# Sample Enumeration
# =============================================================================

def enumerate_all_samples(
    df: pd.DataFrame,
    base_cache_dir: Path,
    sampling_strategy: str,
    hard_ratio: float,
    max_samples_per_scene: int,
    max_views_per_sample: int,
    resume: bool
) -> List[SampleInfo]:
    """Pre-enumerate all samples to process."""
    samples = []
    skipped = 0

    for dataset_name, dataset_group in df.groupby("dataset"):
        dataset_cache_dir = base_cache_dir / dataset_name

        for scene_id, full_group in dataset_group.groupby("scene_id"):
            total_views = len(full_group)
            num_samples = determine_samples_per_scene(
                total_views, max_samples_per_scene, max_views_per_sample
            )

            if num_samples == 0:
                continue

            for sample_idx in range(num_samples):
                # Determine cache directory
                if num_samples > 1:
                    scene_cache_dir = dataset_cache_dir / f"{scene_id}_sample{sample_idx:02d}"
                else:
                    scene_cache_dir = dataset_cache_dir / scene_id

                # Skip if already exists and resuming
                if resume and (scene_cache_dir / "consolidated.npz").exists():
                    skipped += 1
                    continue

                # Determine sampling method
                if sampling_strategy == 'mixed':
                    use_strided = math.floor((sample_idx + 1) * hard_ratio) > math.floor(sample_idx * hard_ratio)
                elif sampling_strategy == 'strided':
                    use_strided = True
                else:
                    use_strided = False

                # Sample views
                if use_strided:
                    group = sample_views_strided(full_group, max_views_per_sample, 42, sample_idx)
                else:
                    group = sample_views_temporally(full_group, max_views_per_sample, 42, sample_idx)

                image_paths = group["image_path"].tolist()

                samples.append(SampleInfo(
                    dataset=dataset_name,
                    scene_id=scene_id,
                    sample_idx=sample_idx,
                    num_samples=num_samples,
                    total_views=total_views,
                    image_paths=image_paths,
                    cache_dir=scene_cache_dir,
                    use_strided=use_strided,
                    hard_ratio=hard_ratio if sampling_strategy == 'mixed' else None
                ))

    return samples, skipped


# =============================================================================
# Input Validation
# =============================================================================

def validate_inputs(samples: List[SampleInfo], check_all: bool = False) -> Dict[str, Any]:
    """Validate that input images exist and are readable."""
    stats = {
        'total_samples': len(samples),
        'total_images': sum(len(s.image_paths) for s in samples),
        'missing_images': [],
        'unreadable_images': []
    }

    # Check subset or all
    check_samples = samples if check_all else samples[:min(10, len(samples))]

    for sample in check_samples:
        for img_path in sample.image_paths:
            path = Path(img_path)
            if not path.exists():
                stats['missing_images'].append(str(img_path))
            else:
                try:
                    with Image.open(path) as img:
                        img.verify()
                except Exception:
                    stats['unreadable_images'].append(str(img_path))

    stats['valid'] = len(stats['missing_images']) == 0 and len(stats['unreadable_images']) == 0
    return stats


# =============================================================================
# Memory Estimation
# =============================================================================

def estimate_memory_usage(num_views: int = 20, height: int = 224, width: int = 518) -> Dict[str, float]:
    """Estimate GPU memory usage for inference."""
    # Model size (~959M params * 2 bytes for bf16)
    model_gb = 959e6 * 2 / 1e9

    # Input images: [1, N, 3, H, W] * 4 bytes (float32)
    input_gb = 1 * num_views * 3 * height * width * 4 / 1e9

    # Output tensors: points, local_points, conf, poses
    # Roughly 4x input size for intermediate activations
    activations_gb = input_gb * 4

    return {
        'model_gb': model_gb,
        'input_gb': input_gb,
        'activations_gb': activations_gb,
        'total_estimated_gb': model_gb + input_gb + activations_gb,
        'recommended_gpu_gb': (model_gb + input_gb + activations_gb) * 1.5  # 50% headroom
    }


# =============================================================================
# PLY Export
# =============================================================================

def save_ply(xyz: np.ndarray, rgb: np.ndarray, path: Path):
    """Save colored point cloud to PLY file."""
    if not HAS_OPEN3D:
        return
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(path), pc, compressed=True)


# =============================================================================
# Processing Functions
# =============================================================================

def process_sample(
    sample: SampleInfo,
    imgs: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    target_height: int,
    target_width: int,
    export_ply: bool
) -> Dict[str, Any]:
    """Process a single sample through the model and save results."""

    # Create cache directory
    sample.cache_dir.mkdir(parents=True, exist_ok=True)

    # Move to device
    imgs = imgs.to(device)

    # Run inference
    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype, enabled=(device.type == 'cuda')):
            res = model(imgs[None])  # Add batch dimension
    inference_time = time.perf_counter() - t0

    # Extract outputs
    xyz_local = res['local_points'][0].cpu()
    conf_raw = res['conf'][0].cpu()
    camera_poses = res['camera_poses'][0].cpu()
    xyz_global = res['points'][0].cpu()

    # Apply sigmoid to confidence
    conf = torch.sigmoid(conf_raw)

    N, H_out, W_out, _ = xyz_local.shape

    # Convert to numpy
    xyz_local_np = xyz_local.numpy()
    conf_np = conf.numpy()
    camera_poses_np = camera_poses.numpy()
    xyz_global_np = xyz_global.numpy()

    # Resize if needed
    if (H_out, W_out) != (target_height, target_width):
        xyz_local_t = torch.from_numpy(xyz_local_np).permute(0, 3, 1, 2)
        xyz_local_t = F.interpolate(xyz_local_t, size=(target_height, target_width),
                                     mode='bilinear', align_corners=False)
        xyz_local_np = xyz_local_t.permute(0, 2, 3, 1).numpy()

        conf_t = torch.from_numpy(conf_np).permute(0, 3, 1, 2)
        conf_t = F.interpolate(conf_t, size=(target_height, target_width),
                               mode='bilinear', align_corners=False)
        conf_np = conf_t.permute(0, 2, 3, 1).numpy()

        if export_ply:
            xyz_global_t = torch.from_numpy(xyz_global_np).permute(0, 3, 1, 2)
            xyz_global_t = F.interpolate(xyz_global_t, size=(target_height, target_width),
                                         mode='bilinear', align_corners=False)
            xyz_global_np = xyz_global_t.permute(0, 2, 3, 1).numpy()

    # Create validity mask
    conf_threshold = 0.1
    depth = xyz_local_np[..., 2]
    depth_t = torch.from_numpy(depth)
    non_edge = ~depth_edge(depth_t, rtol=0.03)
    conf_mask = conf_np[..., 0] > conf_threshold
    valid_mask = conf_mask & non_edge.numpy()

    # Prepare arrays
    xyz_local_list = []
    conf_list = []
    mask_rle_list = []

    for i in range(N):
        xyz_i = xyz_local_np[i].copy()
        conf_i = conf_np[i].copy()
        mask_i = valid_mask[i]

        invalid_mask = ~mask_i
        xyz_i[invalid_mask] = np.nan
        conf_i[invalid_mask] = 0

        mask_rle = encode_rle(mask_i)

        xyz_local_list.append(xyz_i.astype(np.float16))
        conf_list.append(conf_i.astype(np.float16))
        mask_rle_list.append(mask_rle)

    # Stack arrays
    xyz_local_stacked = np.stack(xyz_local_list, axis=0)
    conf_stacked = np.stack(conf_list, axis=0)
    camera_poses_stacked = camera_poses_np.astype(np.float32)
    masks_array = np.array(mask_rle_list, dtype=object)

    # Save consolidated file (EXACT SAME FORMAT as export_pi3.py)
    np.savez_compressed(
        sample.cache_dir / "consolidated.npz",
        xyz_local=xyz_local_stacked,
        conf=conf_stacked,
        camera_poses=camera_poses_stacked,
        masks=masks_array,
        num_views=N,
        inference_time=np.float32(inference_time),
        teacher_model='pi3'
    )

    # Save sampled views JSON
    sampled_view_info = {
        'dataset': sample.dataset,
        'scene_id': sample.scene_id,
        'sample_id': sample.sample_idx,
        'total_samples': sample.num_samples,
        'original_count': sample.total_views,
        'sampled_count': len(sample.image_paths),
        'sampled_image_paths': sample.image_paths,
        'teacher_model': 'pi3',
        'sampling_strategy': 'strided' if sample.use_strided else 'temporal',
        'hard_ratio': sample.hard_ratio
    }

    with open(sample.cache_dir / "sampled_views.json", 'w') as f:
        json.dump(sampled_view_info, f, indent=2)

    # Export PLY if requested
    if export_ply and HAS_OPEN3D:
        xyz_list, rgb_list = [], []

        for i, img_path in enumerate(sample.image_paths):
            try:
                orig = Image.open(img_path).convert("RGB")
                xyz = xyz_global_np[i]
                H_viz, W_viz = xyz.shape[:2]
                rgb = np.array(orig.resize((W_viz, H_viz), Image.LANCZOS))

                mask = valid_mask[i]
                if mask.any():
                    xyz_list.append(xyz[mask])
                    rgb_list.append(rgb[mask])
            except Exception:
                continue

        if xyz_list:
            xyz_all = np.concatenate(xyz_list, 0)
            rgb_all = np.concatenate(rgb_list, 0)
            valid_pts = np.isfinite(xyz_all).all(axis=1)
            if valid_pts.any():
                save_ply(xyz_all[valid_pts], rgb_all[valid_pts],
                        sample.cache_dir / "fused_pi3.ply")

        np.save(sample.cache_dir / "poses_c2w.npy", camera_poses_stacked)

    return {
        'inference_time': inference_time,
        'num_views': N
    }


# =============================================================================
# GPU Worker Function
# =============================================================================

def gpu_worker(
    gpu_id: int,
    samples: List[SampleInfo],
    args,
    result_queue: mp.Queue
):
    """Worker function for a single GPU."""

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # Determine dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Load model
    if args.checkpoint is not None:
        model = Pi3().to(device).eval()
        if args.checkpoint.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.checkpoint)
        else:
            weight = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    # Create dataset and dataloader
    dataset = SampleDataset(samples, target_size=(args.target_height, args.target_width))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_samples,
        pin_memory=True
    )

    # Process samples
    total_images = 0
    total_time = 0
    processed = 0
    failed = 0

    start_time = time.perf_counter()

    pbar = tqdm(loader, desc=f"GPU {gpu_id}", position=gpu_id)
    for batch in pbar:
        if not batch['load_success']:
            failed += 1
            continue

        sample = batch['sample']
        imgs = batch['imgs']

        try:
            result = process_sample(
                sample, imgs, model, device, dtype,
                args.target_height, args.target_width,
                args.export_ply
            )

            total_images += result['num_views']
            total_time += result['inference_time']
            processed += 1

            # Update progress bar
            elapsed = time.perf_counter() - start_time
            img_per_sec = total_images / elapsed if elapsed > 0 else 0
            remaining = len(samples) - processed - failed
            eta_sec = remaining * (elapsed / max(processed, 1))
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

            pbar.set_postfix({
                'img/s': f'{img_per_sec:.1f}',
                'ETA': eta_str
            })

        except Exception as e:
            print(f"GPU {gpu_id}: Error processing {sample.scene_id}: {e}")
            failed += 1

        # Clear cache periodically
        if processed % 50 == 0:
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - start_time

    result_queue.put({
        'gpu_id': gpu_id,
        'processed': processed,
        'failed': failed,
        'total_images': total_images,
        'total_time': elapsed,
        'img_per_sec': total_images / elapsed if elapsed > 0 else 0
    })


# =============================================================================
# Main Function
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='OTT3R Pseudo-label Generator - Fast parallel cache generation from π³',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single GPU with 4 data loading workers
    python ott3r_generator.py --num-workers 4

    # Two GPUs with 4 workers each
    python ott3r_generator.py --num-gpus 2 --num-workers 4

    # Estimate time without running
    python ott3r_generator.py --dry-run

    # Generate with PLY visualizations
    python ott3r_generator.py --export-ply
        """
    )

    # Core arguments
    parser.add_argument('--manifest', type=str, default='processed_data/images/manifest.csv',
                        help='Path to manifest CSV')
    parser.add_argument('--cache-dir', type=str, default='caches/teacher_cache',
                        help='Output cache directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to π³ checkpoint (default: HuggingFace)')

    # Parallelization
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers per GPU (default: 4)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs to use (default: 1)')

    # Sampling
    parser.add_argument('--sampling-strategy', type=str, default='mixed',
                        choices=['temporal', 'strided', 'mixed'],
                        help='View sampling strategy (default: mixed)')
    parser.add_argument('--hard-ratio', type=float, default=0.5,
                        help='Ratio of strided samples in mixed mode (default: 0.5)')
    parser.add_argument('--max-samples-per-scene', type=int, default=10000,
                        help='Maximum samples per scene')

    # Modes
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing cache')
    parser.add_argument('--dry-run', action='store_true',
                        help='Estimate time without processing')
    parser.add_argument('--export-ply', action='store_true',
                        help='Export PLY point cloud visualizations')
    parser.add_argument('--validate', action='store_true',
                        help='Validate all input images before processing')

    # Filtering
    parser.add_argument('--filter-dataset', type=str, default=None,
                        help='Filter to single dataset')

    args = parser.parse_args()

    # Fixed parameters
    args.max_views_per_sample = 20
    args.target_height = 224
    args.target_width = 518

    # Paths
    manifest_path = Path(args.manifest)
    if args.filter_dataset and args.cache_dir == 'caches/teacher_cache':
        base_cache_dir = Path(f'caches/teacher_cache_{args.filter_dataset}')
    else:
        base_cache_dir = Path(args.cache_dir)

    # Validate manifest exists
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        return

    # Load manifest
    df = pd.read_csv(manifest_path)
    if args.filter_dataset:
        df = df[df['dataset'] == args.filter_dataset]
        if len(df) == 0:
            print(f"Error: No data found for dataset '{args.filter_dataset}'")
            return

    print("=" * 80)
    print("OTT3R Pseudo-label Generator")
    print("=" * 80)
    print(f"Manifest: {manifest_path} ({len(df):,} images)")
    print(f"Cache directory: {base_cache_dir}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Workers per GPU: {args.num_workers}")
    print(f"Sampling: {args.sampling_strategy}" +
          (f" (hard ratio: {args.hard_ratio:.0%})" if args.sampling_strategy == 'mixed' else ""))
    print(f"Resume: {args.resume}")
    print(f"Export PLY: {args.export_ply}")
    print()

    # Enumerate samples
    print("Enumerating samples...")
    base_cache_dir.mkdir(parents=True, exist_ok=True)

    samples, skipped = enumerate_all_samples(
        df, base_cache_dir, args.sampling_strategy, args.hard_ratio,
        args.max_samples_per_scene, args.max_views_per_sample, args.resume
    )

    total_images = sum(len(s.image_paths) for s in samples)

    print(f"Samples to process: {len(samples):,}")
    print(f"Samples skipped (existing): {skipped:,}")
    print(f"Total images: {total_images:,}")
    print()

    if len(samples) == 0:
        print("Nothing to process. Exiting.")
        return

    # Validate inputs
    if args.validate:
        print("Validating inputs...")
        stats = validate_inputs(samples, check_all=True)
        if not stats['valid']:
            print(f"WARNING: Found {len(stats['missing_images'])} missing images")
            print(f"WARNING: Found {len(stats['unreadable_images'])} unreadable images")
            if stats['missing_images'][:5]:
                print("  Examples:", stats['missing_images'][:5])
        else:
            print("All inputs valid.")
        print()

    # Memory estimation
    mem = estimate_memory_usage(args.max_views_per_sample, args.target_height, args.target_width)
    print(f"Estimated GPU memory: {mem['total_estimated_gb']:.1f} GB")
    print(f"Recommended GPU: {mem['recommended_gpu_gb']:.1f} GB+")
    print()

    # Dry run
    if args.dry_run:
        # Estimate based on ~17 img/sec baseline
        baseline_img_per_sec = 17
        speedup = min(args.num_gpus, 4)  # Assume near-linear up to 4 GPUs
        effective_img_per_sec = baseline_img_per_sec * speedup

        estimated_sec = total_images / effective_img_per_sec
        estimated_time = datetime.timedelta(seconds=int(estimated_sec))

        print("=" * 80)
        print("DRY RUN - Time Estimation")
        print("=" * 80)
        print(f"Total images: {total_images:,}")
        print(f"Estimated throughput: {effective_img_per_sec:.1f} img/sec")
        print(f"  (baseline: {baseline_img_per_sec} img/sec × {speedup} GPUs)")
        print(f"Estimated time: {estimated_time}")
        print()
        print("Run without --dry-run to process.")
        return

    # Process
    print("=" * 80)
    print("Starting Processing")
    print("=" * 80)

    start_time = time.perf_counter()

    if args.num_gpus == 1:
        # Single GPU - run directly
        result_queue = mp.Queue()
        gpu_worker(0, samples, args, result_queue)
        results = [result_queue.get()]
    else:
        # Multi-GPU - use multiprocessing
        mp.set_start_method('spawn', force=True)

        # Shard samples across GPUs
        sharded_samples = [[] for _ in range(args.num_gpus)]
        for i, sample in enumerate(samples):
            sharded_samples[i % args.num_gpus].append(sample)

        print(f"Sharding {len(samples)} samples across {args.num_gpus} GPUs:")
        for i, shard in enumerate(sharded_samples):
            print(f"  GPU {i}: {len(shard)} samples")
        print()

        # Launch workers
        result_queue = mp.Queue()
        processes = []

        for gpu_id in range(args.num_gpus):
            p = mp.Process(
                target=gpu_worker,
                args=(gpu_id, sharded_samples[gpu_id], args, result_queue)
            )
            p.start()
            processes.append(p)

        # Wait for completion
        for p in processes:
            p.join()

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

    total_time = time.perf_counter() - start_time

    # Aggregate results
    total_processed = sum(r['processed'] for r in results)
    total_failed = sum(r['failed'] for r in results)
    total_images_done = sum(r['total_images'] for r in results)
    overall_img_per_sec = total_images_done / total_time if total_time > 0 else 0

    # Print summary
    print()
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Samples processed: {total_processed:,}")
    print(f"Samples failed: {total_failed:,}")
    print(f"Images processed: {total_images_done:,}")
    print(f"Total time: {datetime.timedelta(seconds=int(total_time))}")
    print(f"Throughput: {overall_img_per_sec:.1f} img/sec")
    print(f"Cache saved to: {base_cache_dir}")
    print()

    # Per-GPU breakdown
    if args.num_gpus > 1:
        print("Per-GPU breakdown:")
        for r in sorted(results, key=lambda x: x['gpu_id']):
            print(f"  GPU {r['gpu_id']}: {r['processed']} samples, "
                  f"{r['total_images']} images, {r['img_per_sec']:.1f} img/sec")
        print()

    # Save summary JSON
    summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'manifest': str(manifest_path),
        'cache_dir': str(base_cache_dir),
        'num_gpus': args.num_gpus,
        'num_workers': args.num_workers,
        'sampling_strategy': args.sampling_strategy,
        'hard_ratio': args.hard_ratio,
        'samples_processed': total_processed,
        'samples_failed': total_failed,
        'samples_skipped': skipped,
        'images_processed': total_images_done,
        'total_time_sec': total_time,
        'throughput_img_per_sec': overall_img_per_sec,
        'per_gpu_results': results
    }

    summary_path = base_cache_dir / "generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to: {summary_path}")

    print()
    print("Cache contents per sample:")
    print(f"  - xyz_local:     [N, {args.target_height}, {args.target_width}, 3] float16")
    print(f"  - conf:          [N, {args.target_height}, {args.target_width}, 1] float16")
    print(f"  - camera_poses:  [N, 4, 4] float32")
    print(f"  - masks:         [N] RLE-encoded validity masks")


if __name__ == "__main__":
    main()
