#!/usr/bin/env python3
"""
OTT3R Inference Script

Generates colored 3D point clouds from multi-view images.

Usage:
    python utils/inference.py images/ --checkpoint checkpoints/ott3r/last.ckpt
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ott3r.student.model import OTT3R
from ott3r.student.distillation_module import OTT3RLitModule

# Add pi3 to path for geometry utils
pi3_path = str(project_root / "external" / "pi3")
sys.path.insert(0, pi3_path)
from pi3.utils.geometry import se3_inverse, homogenize_points


def run_timed(func, *args, **kwargs):
    """Run function with timing."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - t0


def save_ply(xyz: np.ndarray, rgb: np.ndarray, path: Path):
    """Save point cloud to PLY file."""
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz)
        pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(path), pc, compressed=True)
    except ImportError:
        # Fallback: save as simple PLY without open3d
        print("open3d not available, saving simple PLY format")
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(xyz)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(len(xyz)):
                f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                       f"{int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}\n")


def load_student_model(checkpoint_path: str, device: torch.device):
    """Load trained OTT3R model from checkpoint."""
    print(f"Loading student model from {checkpoint_path}")

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        ckpt_files = list(ckpt_path.glob("*.ckpt"))
        if not ckpt_files:
            raise ValueError(f"No .ckpt files found in {checkpoint_path}")
        checkpoint_path = str(ckpt_files[0])
        print(f"Using checkpoint: {checkpoint_path}")

    lit_module = OTT3RLitModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False
    )
    lit_module = lit_module.to(device)
    lit_module.eval()

    # Extract the student model
    student_model = lit_module.student
    student_model = student_model.to(device)
    student_model.eval()

    print(f"Student model loaded successfully")
    print(f"Total parameters: {sum(p.numel() for p in student_model.parameters()):,}")

    return student_model


def load_images_from_directory(image_dir: Path, num_views: int = None, target_size: tuple = (224, 518)):
    """
    Load images from directory and preprocess for inference.

    Args:
        image_dir: Directory containing images
        num_views: Maximum number of views to load (None = all)
        target_size: (H, W) target size for model input

    Returns:
        images_tensor: [1, N, 3, H, W] batch of normalized images
        original_images: List of PIL images for coloring
        image_paths: List of image paths
    """
    import torchvision.transforms.functional as TF

    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(image_dir / ext)))
        image_paths.extend(glob.glob(str(image_dir / ext.upper())))

    image_paths = sorted(image_paths)

    if not image_paths:
        raise ValueError(f"No images found in directory: {image_dir}")

    # Limit number of views if specified
    if num_views is not None and num_views < len(image_paths):
        image_paths = image_paths[:num_views]

    print(f"Found {len(image_paths)} images in {image_dir}")
    for i, path in enumerate(image_paths[:5]):
        print(f"  {i+1}: {Path(path).name}")
    if len(image_paths) > 5:
        print(f"  ... and {len(image_paths) - 5} more")

    # Load and preprocess images
    H, W = target_size
    images = []
    original_images = []

    for path in image_paths:
        # Load original image
        img = Image.open(path).convert('RGB')
        original_images.append(img)

        # Resize and convert to tensor
        img_resized = img.resize((W, H), Image.LANCZOS)
        img_tensor = TF.to_tensor(img_resized)  # [3, H, W], range [0, 1]
        images.append(img_tensor)

    # Stack into batch [N, 3, H, W]
    images_tensor = torch.stack(images, dim=0)

    # Add batch dimension [1, N, 3, H, W]
    images_tensor = images_tensor.unsqueeze(0)

    return images_tensor, original_images, image_paths


def gauge_fix_to_frame0(output: dict) -> dict:
    """
    Transform output to frame 0's coordinate system.

    Args:
        output: Model output dict with 'camera_poses' [B, N, 4, 4] and 'points' [B, N, H, W, 3]

    Returns:
        Modified output with gauge-fixed poses and points
    """
    camera_poses = output['camera_poses']  # [B, N, 4, 4]
    points = output['points']  # [B, N, H, W, 3]

    # Get world-to-camera transform of frame 0
    # This will transform everything so frame 0 becomes identity
    w2c_0 = se3_inverse(camera_poses[:, 0:1])  # [B, 1, 4, 4]

    # Transform all poses: T_i_new = T_0^{-1} @ T_i
    # After this, T_0 will be identity
    camera_poses_fixed = torch.matmul(w2c_0, camera_poses)  # [B, N, 4, 4]

    # Transform global points to frame 0's coordinate system
    # points_new = T_0^{-1} @ points
    points_fixed = torch.einsum(
        'bij, bnhwj -> bnhwi',
        w2c_0.squeeze(1),  # [B, 4, 4]
        homogenize_points(points)
    )[..., :3]

    # Return modified output
    return {
        'local_points': output['local_points'],  # unchanged (already in camera frame)
        'conf': output['conf'],  # unchanged
        'camera_poses': camera_poses_fixed,
        'points': points_fixed,
    }


def run_inference(student_model, images: torch.Tensor, device: torch.device, gauge_fix: bool = True):
    """
    Run inference using OTT3R model.

    Args:
        student_model: Trained OTT3R model
        images: [B, N, 3, H, W] input images (already normalized will be done in model)
        device: torch device
        gauge_fix: If True, transform output to frame 0's coordinate system

    Returns:
        dict with 'local_points', 'conf', 'camera_poses', 'points'
    """
    images = images.to(device)

    with torch.no_grad():
        output = student_model(images)

        # Apply gauge-fixing to anchor to frame 0
        if gauge_fix:
            output = gauge_fix_to_frame0(output)

    return output


def extract_point_cloud(output: dict, original_images: list, conf_threshold: float = 0.5):
    """
    Extract colored point cloud from model output.

    Args:
        output: Model output with 'points' [B, N, H, W, 3] and 'conf' [B, N, H, W, 1]
        original_images: List of PIL images for coloring
        conf_threshold: Confidence threshold (0-1), applied to sigmoid(conf)

    Returns:
        points: [M, 3] numpy array of 3D points
        colors: [M, 3] numpy array of RGB colors
    """
    # Get global points and confidence
    points_tensor = output['points']  # [B, N, H, W, 3]
    conf_tensor = output['conf']  # [B, N, H, W, 1] - raw logits

    # Remove batch dimension
    points_tensor = points_tensor[0]  # [N, H, W, 3]
    conf_tensor = conf_tensor[0]  # [N, H, W, 1]

    N, H, W, _ = points_tensor.shape

    all_points = []
    all_colors = []

    print(f"\nExtracting points from {N} views:")

    for view_idx in range(N):
        xyz = points_tensor[view_idx].cpu().numpy()  # [H, W, 3]
        conf_logits = conf_tensor[view_idx, :, :, 0].cpu().numpy()  # [H, W]

        # Convert logits to probabilities via sigmoid
        conf = 1.0 / (1.0 + np.exp(-conf_logits))  # sigmoid

        # Create validity mask
        finite_mask = np.isfinite(xyz).all(axis=-1)
        conf_mask = conf > conf_threshold
        valid_mask = finite_mask & conf_mask

        valid_count = valid_mask.sum()
        total_count = valid_mask.size

        print(f"  View {view_idx}: {valid_count:,}/{total_count:,} valid points "
              f"(logits: {conf_logits.min():.3f}-{conf_logits.max():.3f}, "
              f"prob: {conf.min():.3f}-{conf.max():.3f})")

        if valid_count == 0:
            continue

        # Extract valid points
        valid_xyz = xyz[valid_mask]

        # Get colors from original image
        if view_idx < len(original_images):
            orig_img = original_images[view_idx].resize((W, H), Image.LANCZOS)
            rgb_array = np.asarray(orig_img)
            valid_colors = rgb_array[valid_mask]
        else:
            # Fallback: use view-based coloring
            view_colors = np.array([
                [255, 0, 0], [0, 255, 0], [0, 0, 255],
                [255, 255, 0], [255, 0, 255], [0, 255, 255],
                [128, 0, 0], [0, 128, 0], [0, 0, 128],
            ])
            color = view_colors[view_idx % len(view_colors)]
            valid_colors = np.tile(color, (len(valid_xyz), 1))

        all_points.append(valid_xyz)
        all_colors.append(valid_colors)

    if all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        return points, colors
    else:
        return None, None


def main():
    parser = argparse.ArgumentParser(description="OTT3R Inference")
    parser.add_argument("image_dir", type=str,
                        help="Directory containing images to process")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to student model checkpoint (.ckpt)")
    parser.add_argument("--output-dir", default="inference_results",
                        help="Output directory for point clouds (default: inference_results)")
    parser.add_argument("--num-views", type=int, default=None,
                        help="Maximum number of views to process (default: all)")
    parser.add_argument("--conf-threshold", type=float, default=0.1,
                        help="Confidence threshold 0-1 (default: 0.1)")
    parser.add_argument("--resolution", type=str, default="224x518",
                        help="Input resolution HxW (default: 224x518). Examples: 448x1036, 336x777")
    parser.add_argument("--device", default="cuda",
                        help="Device to use: cuda or cpu (default: cuda)")
    parser.add_argument("--no-gauge-fix", action="store_true",
                        help="Disable gauge-fixing to frame 0 (default: enabled)")

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    image_dir = Path(args.image_dir)

    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")

    # Parse resolution
    try:
        res_parts = args.resolution.lower().split('x')
        target_size = (int(res_parts[0]), int(res_parts[1]))
    except (ValueError, IndexError):
        raise ValueError(f"Invalid resolution format: {args.resolution}. Use HxW (e.g., 224x518)")

    print("=" * 60)
    print("OTT3R INFERENCE")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Image directory: {image_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Resolution: {target_size[0]}x{target_size[1]}")
    print(f"Confidence threshold: {args.conf_threshold}")
    if args.num_views:
        print(f"Max views: {args.num_views}")
    print("=" * 60)

    # Load student model
    student_model, load_time = run_timed(load_student_model, args.checkpoint, device)
    print(f"Model load time: {load_time:.2f}s")

    # Load images
    images, original_images, image_paths = load_images_from_directory(
        image_dir, num_views=args.num_views, target_size=target_size
    )
    print(f"\nProcessing {images.shape[1]} images")
    print(f"Input shape: {images.shape}")

    # Run inference
    gauge_fix = not args.no_gauge_fix
    print(f"\nRunning inference...")
    print(f"  Gauge-fix to frame 0: {gauge_fix}")
    output, inference_time = run_timed(run_inference, student_model, images, device, gauge_fix)
    print(f"Inference time: {inference_time:.2f}s")
    print(f"  Per-view: {inference_time / images.shape[1] * 1000:.1f}ms")

    # Print output shapes
    print(f"\nOutput shapes:")
    for key, val in output.items():
        if val is not None:
            print(f"  {key}: {val.shape}")

    # Show camera pose info (verify gauge-fix worked)
    if gauge_fix:
        poses = output['camera_poses'][0]  # [N, 4, 4]
        print(f"\nCamera poses (gauge-fixed to frame 0):")
        print(f"  Frame 0 pose (should be identity):")
        print(f"    R diagonal: {poses[0, 0, 0]:.4f}, {poses[0, 1, 1]:.4f}, {poses[0, 2, 2]:.4f}")
        print(f"    translation: {poses[0, :3, 3].cpu().numpy()}")
        if poses.shape[0] > 1:
            print(f"  Frame 1 pose:")
            print(f"    R diagonal: {poses[1, 0, 0]:.4f}, {poses[1, 1, 1]:.4f}, {poses[1, 2, 2]:.4f}")
            print(f"    translation: {poses[1, :3, 3].cpu().numpy()}")

    # Extract point cloud
    print(f"\nExtracting point cloud (conf_threshold={args.conf_threshold})...")
    points, colors = extract_point_cloud(output, original_images, args.conf_threshold)

    if points is not None:
        print(f"\nTotal points extracted: {len(points):,}")

        # Save point cloud
        scene_name = image_dir.name
        output_path = output_dir / f"{scene_name}_pointcloud.ply"
        save_ply(points, colors, output_path)
        print(f"Saved point cloud: {output_path}")

        # Save metadata
        info_path = output_dir / f"{scene_name}_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"OTT3R Inference\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"Image directory: {image_dir}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Device: {device}\n\n")
            f.write(f"Number of images: {len(image_paths)}\n")
            f.write(f"Input size: {target_size[0]}x{target_size[1]}\n")
            f.write(f"Confidence threshold: {args.conf_threshold}\n\n")
            f.write(f"Model load time: {load_time:.2f}s\n")
            f.write(f"Inference time: {inference_time:.2f}s\n")
            f.write(f"Per-view time: {inference_time / len(image_paths) * 1000:.1f}ms\n\n")
            f.write(f"Total points: {len(points):,}\n\n")
            f.write(f"Image files processed:\n")
            for i, path in enumerate(image_paths):
                f.write(f"  {i+1}: {Path(path).name}\n")

        print(f"Saved info: {info_path}")
    else:
        print("ERROR: No valid points extracted")
        print("Try lowering --conf-threshold")

    print(f"\n{'=' * 60}")
    print("Inference complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
