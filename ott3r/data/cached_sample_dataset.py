"""
Cached Sample Dataset for OTT3R Training

Loads teacher cache for knowledge distillation.

Cache format:
- xyz_local: [N, H, W, 3] float16
- conf: [N, H, W, 1] float16
- camera_poses: [N, 4, 4] float32
- masks: RLE encoded validity masks
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class CachedSampleDataset(Dataset):
    """Dataset that loads images and metadata from teacher cache directory."""

    def __init__(
        self,
        teacher_cache_dir: str,
        image_size: int = 512,
        num_views: int = 20,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
    ):
        self.teacher_cache_dir = Path(teacher_cache_dir)
        self.image_size = image_size
        self.num_views = num_views
        self.transform = transform
        self.max_samples = max_samples

        # Collect all samples from cache directories
        self.samples = self._collect_samples()

        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]
            print(f"Limited to {len(self.samples)} samples (max_samples={self.max_samples})")
        else:
            print(f"Found {len(self.samples)} cached samples in {teacher_cache_dir}")

    def _collect_samples(self) -> List[Dict]:
        """Collect all sample information from sampled_views.json files."""
        samples = []

        if not self.teacher_cache_dir.exists():
            raise ValueError(f"Teacher cache directory not found: {self.teacher_cache_dir}")

        # Traverse dataset directories
        for dataset_dir in self.teacher_cache_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            # Traverse scene/sample directories
            for scene_dir in dataset_dir.iterdir():
                if not scene_dir.is_dir():
                    continue

                sampled_views_file = scene_dir / "sampled_views.json"
                if not sampled_views_file.exists():
                    continue

                # Check for consolidated cache file
                consolidated_file = scene_dir / "consolidated.npz"
                if not consolidated_file.exists():
                    continue

                try:
                    with open(sampled_views_file, 'r') as f:
                        sample_info = json.load(f)

                    # Get image paths
                    image_paths = sample_info.get(
                        'final_image_paths',
                        sample_info.get('sampled_image_paths', [])
                    )

                    if len(image_paths) < 2:  # Need at least 2 views
                        continue

                    sample_data = {
                        'dataset': sample_info['dataset'],
                        'scene_id': sample_info['scene_id'],
                        'sample_id': sample_info.get('sample_id', 0),
                        'image_paths': image_paths,
                        'cache_dir': scene_dir,
                        'total_views': len(image_paths),
                        'teacher_model': sample_info.get('teacher_model', 'pi3')
                    }

                    samples.append(sample_data)

                except Exception as e:
                    print(f"Warning: Failed to load sample info from {sampled_views_file}: {e}")
                    continue

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Get the image paths for this sample
        all_image_paths = sample['image_paths']
        selected_paths = all_image_paths[:self.num_views]

        # Load images
        images = []
        for path_str in selected_paths:
            path = Path(path_str)
            if not path.is_absolute():
                path = Path.cwd() / path

            try:
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Create dummy image
                img = Image.new('RGB', (self.image_size, self.image_size), color='black')
                images.append(img)

        return {
            'images': images,
            'paths': selected_paths,
            'dataset': sample['dataset'],
            'scene_id': sample['scene_id'],
            'sample_id': sample['sample_id'],
            'cache_dir': sample['cache_dir'],
            'num_views': len(images),
        }


def load_teacher_cache_for_sample(cache_dir: Path, num_views: int) -> Dict:
    """Load teacher cache from consolidated.npz format."""
    consolidated_path = cache_dir / "consolidated.npz"
    if not consolidated_path.exists():
        raise FileNotFoundError(f"Consolidated cache not found: {consolidated_path}")

    try:
        cache_data = np.load(consolidated_path, allow_pickle=True)

        # Load π³ cache format
        xyz_local = cache_data['xyz_local']  # [N, H, W, 3] float16
        conf = cache_data['conf']  # [N, H, W, 1] float16
        camera_poses = cache_data['camera_poses']  # [N, 4, 4] float32
        masks_rle = cache_data.get('masks', None)  # [N] object array of RLE

        # Convert to tensors
        teacher_data = {
            'local_points': torch.from_numpy(xyz_local.astype(np.float32)),
            'conf': torch.from_numpy(conf.astype(np.float32)),
            'camera_poses': torch.from_numpy(camera_poses.astype(np.float32)),
        }

        # Decode masks if present
        if masks_rle is not None:
            from ott3r.teacher.rle_helpers import decode_rle
            H, W = xyz_local.shape[1:3]
            masks = []
            for i in range(len(masks_rle)):
                mask_rle = list(masks_rle[i])
                mask = decode_rle(mask_rle, (H, W))
                masks.append(torch.from_numpy(mask).bool())
            teacher_data['masks'] = torch.stack(masks, dim=0)
        else:
            # Create default mask from valid xyz coordinates
            valid = torch.isfinite(teacher_data['local_points']).all(dim=-1)
            teacher_data['masks'] = valid

        cache_data.close()
        return teacher_data

    except Exception as e:
        raise RuntimeError(f"Failed to load π³ cache from {consolidated_path}: {e}")


def collate_cached_samples(batch: List[Dict]) -> Dict:
    """Collate function that resizes images and loads teacher supervision."""
    # Target resolution for π³ (patch_size=14)
    target_h, target_w = 224, 518

    all_images = []
    all_paths = []
    all_datasets = []
    all_scene_ids = []
    all_sample_ids = []
    all_teacher_data = []
    all_masks = []

    max_views = 20  # π³ uses 20 views

    for sample in batch:
        images = sample['images']

        if len(images) != max_views:
            print(f"Warning: Expected {max_views} views, got {len(images)}")

        # Convert PIL images to tensors and resize
        image_tensors = []
        for img in images[:max_views]:
            if isinstance(img, Image.Image):
                img_tensor = transforms.ToTensor()(img)  # [C, H, W]
            elif isinstance(img, torch.Tensor):
                img_tensor = img.squeeze(0) if img.dim() == 4 else img
            else:
                raise ValueError(f"Unknown image type: {type(img)}")

            # Ensure landscape mode
            if img_tensor.shape[-1] < img_tensor.shape[-2]:
                img_tensor = img_tensor.transpose(-1, -2)

            # Resize to target dimensions
            if img_tensor.shape[-2:] != (target_h, target_w):
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            image_tensors.append(img_tensor)

        # Pad if needed (should not happen if cache is always 20 views)
        while len(image_tensors) < max_views:
            image_tensors.append(image_tensors[-1].clone())

        # Load teacher cache data
        teacher_data = load_teacher_cache_for_sample(sample['cache_dir'], max_views)

        # randomize view order to prevent index-based pose shortcut
        perm = torch.randperm(max_views)

        # permute images
        image_tensors = [image_tensors[i] for i in perm.tolist()]

        # permute teacher supervision
        teacher_data['local_points'] = teacher_data['local_points'][perm]
        teacher_data['conf'] = teacher_data['conf'][perm]
        teacher_data['camera_poses'] = teacher_data['camera_poses'][perm]
        teacher_data['masks'] = teacher_data['masks'][perm]

        # keep paths consistent with permuted views (debug/metadata)
        permuted_paths = [sample['paths'][i] for i in perm.tolist()]

        all_images.append(torch.stack(image_tensors))
        all_paths.append(permuted_paths)
        all_datasets.append(sample['dataset'])
        all_scene_ids.append(sample['scene_id'])
        all_sample_ids.append(sample['sample_id'])
        all_teacher_data.append(teacher_data)
        all_masks.append(teacher_data['masks'])

    # Stack teacher data across batch
    batched_teacher = {
        'local_points': torch.stack([td['local_points'] for td in all_teacher_data]),
        'conf': torch.stack([td['conf'] for td in all_teacher_data]),
        'camera_poses': torch.stack([td['camera_poses'] for td in all_teacher_data]),
    }

    return {
        'images': torch.stack(all_images),  # [B, N, C, H, W]
        'paths': all_paths,
        'datasets': all_datasets,
        'scene_ids': all_scene_ids,
        'sample_ids': all_sample_ids,
        'teacher_data': batched_teacher,
        'masks': torch.stack(all_masks),  # [B, N, H, W]
        'true_shape': (target_h, target_w),
    }


def create_cached_samples_collate_fn():
    """Factory function to create collate function."""
    return collate_cached_samples
