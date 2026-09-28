"""
Manifest generation and image preprocessing for OTT3R training.

Reads dataset configs, resizes images to 960px max edge, and creates a
manifest CSV listing all images with their scene IDs and paths.

Usage:
    from ott3r.data.datasets import get_manifest
    df = get_manifest("configs/data_paths.yaml")
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


# Restrict imports to avoid circular dependencies
__all__ = [
    "DatasetSpec",
    "PrepCfg",
    "prepare_split",
    "get_manifest",
]

# -----------------------------------------------------------------------------#
# Configuration dataclasses
# -----------------------------------------------------------------------------#


@dataclass
class DatasetSpec:
    """Specification for *one* subset (e.g. CO3D train)."""

    name: str  # CO3D, ARK, …
    root: Path  # directory containing raw files
    pattern: str  # glob underneath root, e.g. "**/*.jpg"
    sample_frac: float  # 0.03 → 3 %
    split: str  # train / val / test
    split_file: str = None  # e.g., "TrainSplit.txt"


@dataclass
class PrepCfg:
    """Generic preprocessing settings."""

    resize_max: int = 960  # longest edge after scaling
    out_dir: Path = Path("data/images")  # where to write PNGs


# -----------------------------------------------------------------------------#
# Scene ID extraction functions
# -----------------------------------------------------------------------------#


def _extract_scene_id(file_path: Path, dataset_name: str) -> str:
    """Extract scene ID from file path based on dataset structure."""
    parts = file_path.parts
    
    if dataset_name == "CO3D" or dataset_name == "CO3D_single":
        # Structure: dataset/co3d/category/sequence/images/frame_*.jpg
        # OR: dataset/co3d_single_sequence/category/sequence/images/frame_*.jpg
        # Scene ID: CO3D_category_sequence or CO3D_single_category_sequence
        for i, part in enumerate(parts):
            if (part == "co3d" or part == "co3d_single_sequence") and i + 2 < len(parts):
                category = parts[i + 1]
                sequence = parts[i + 2]
                prefix = "CO3D_single" if part == "co3d_single_sequence" else "CO3D"
                return f"{prefix}_{category}_{sequence}"
        
    elif dataset_name == "ARKitScenes":
        # Structure: raw_data/arkitscenes/3dod/Training/scene_id/scene_id_frames/lowres_wide/*.png
        # Scene ID: ARKitScenes_scene_id
        for i, part in enumerate(parts):
            if part in ["Training", "Validation"] and i + 1 < len(parts):
                scene_id = parts[i + 1]
                return f"ARKitScenes_{scene_id}"
                
    elif dataset_name == "MegaDepth":
        # Structure: dataset/megadepth/scene_id/dense*/imgs/*.jpg
        # Scene ID: MegaDepth_scene_id_dense_dir
        for i, part in enumerate(parts):
            if part == "megadepth" and i + 2 < len(parts):
                scene_id = parts[i + 1]
                dense_dir = parts[i + 2]  # e.g., dense0, dense1
                return f"MegaDepth_{scene_id}_{dense_dir}"
                
    elif dataset_name == "BlendedMVS":
        # Structure: raw_data/blendedmvs/scene_id/blended_images/*.jpg
        # Scene ID: BlendedMVS_scene_id
        for i, part in enumerate(parts):
            if part == "blendedmvs" and i + 1 < len(parts):
                scene_id = parts[i + 1]
                return f"BlendedMVS_{scene_id}"

    elif dataset_name == "Habitat":
        # Structure: dataset/habitat/scene_id/matterport_color_images/*.jpg
        # Scene ID: Habitat_scene_id
        for i, part in enumerate(parts):
            if part == "habitat" and i + 1 < len(parts):
                scene_id = parts[i + 1]
                return f"Habitat_{scene_id}"

    elif dataset_name == "ScanNetPP":
        # Structure: dataset/scannetpp/scene_id/dslr/resized_images/*.JPG
        # Scene ID: ScanNetPP_scene_id
        for i, part in enumerate(parts):
            if part == "scannetpp" and i + 1 < len(parts):
                scene_id = parts[i + 1]
                return f"ScanNetPP_{scene_id}"

    elif dataset_name == "7Scenes":
        # Structure: 7scenes/scene/seq-XX/frame-NNNNNN.color.png
        # Scene ID: 7Scenes_scene_seq-XX
        for i, part in enumerate(parts):
            if part == "7scenes" and i + 2 < len(parts):
                scene = parts[i + 1]  # chess, fire, heads, etc.
                seq = parts[i + 2]    # seq-01, seq-02, etc.
                return f"7Scenes_{scene}_{seq}"

    # Fallback: use parent directory name
    return f"{dataset_name}_{file_path.parent.name}"


def _group_files_by_scene(files: List[Path], dataset_name: str) -> Dict[str, List[Path]]:
    """Group files by scene ID."""
    scene_groups = {}
    for file_path in files:
        scene_id = _extract_scene_id(file_path, dataset_name)
        if scene_id not in scene_groups:
            scene_groups[scene_id] = []
        scene_groups[scene_id].append(file_path)
    return scene_groups


def _sample_scenes(scene_groups: Dict[str, List[Path]], sample_frac: float, split: str, val_scene_split: bool = False) -> List[Path]:
    """Sample a fraction of scenes and return all images from selected scenes."""
    scene_ids = list(scene_groups.keys())
    scene_ids.sort()  # Ensure reproducible ordering
    
    if val_scene_split and len(scene_ids) > 1:
        # Scene-level splitting: deterministically assign scenes to train/val
        random.seed(42)  # Fixed seed for reproducible splits
        random.shuffle(scene_ids)
        
        if split == "val":
            # Take the last 15% of scenes for validation
            val_count = max(1, int(len(scene_ids) * 0.15))
            selected_scene_ids = scene_ids[-val_count:]
            # Apply sample_frac to validation scenes
            keep_count = max(1, int(len(selected_scene_ids) * sample_frac + 0.5))
            selected_scene_ids = selected_scene_ids[:keep_count]
        else:  # train
            # Take the first 85% of scenes for training
            train_count = len(scene_ids) - max(1, int(len(scene_ids) * 0.15))
            train_scene_ids = scene_ids[:train_count]
            # Apply sample_frac to training scenes
            keep_count = max(1, int(len(train_scene_ids) * sample_frac + 0.5))
            selected_scene_ids = train_scene_ids[:keep_count]
    else:
        # Regular sampling: just take sample_frac of all scenes
        random.seed(0)  # For backwards compatibility
        random.shuffle(scene_ids)
        keep_count = max(1, int(len(scene_ids) * sample_frac + 0.5))
        selected_scene_ids = scene_ids[:keep_count]
    
    # Collect all files from selected scenes
    selected_files = []
    for scene_id in selected_scene_ids:
        selected_files.extend(scene_groups[scene_id])
    
    return selected_files


def _filter_by_split_file(files: List[Path], root: Path, split_file: str) -> List[Path]:
    """Filter files to sequences listed in per-scene split files (e.g., TrainSplit.txt)."""
    # Build set of valid (scene, sequence) pairs from split files
    valid_seqs = set()

    for scene_dir in root.iterdir():
        if not scene_dir.is_dir():
            continue

        split_path = scene_dir / split_file
        if not split_path.exists():
            continue

        for line in split_path.read_text().strip().split('\n'):
            if not line.strip():
                continue
            # Parse "sequence1" -> "seq-01" (7Scenes format)
            num = ''.join(filter(str.isdigit, line))
            if num:
                seq = f"seq-{num.zfill(2)}"
                valid_seqs.add((scene_dir.name, seq))

    if not valid_seqs:
        print(f"  Warning: No valid sequences found in {split_file} files", file=sys.stderr)
        return files

    # Filter files to only those matching valid sequences
    # Expected path structure: .../scene/seq-XX/frame.png
    filtered = []
    for f in files:
        parts = f.parts
        if len(parts) >= 3:
            scene = parts[-3]  # e.g., "chess"
            seq = parts[-2]    # e.g., "seq-01"
            if (scene, seq) in valid_seqs:
                filtered.append(f)

    print(f"  Split filter: {len(files)} -> {len(filtered)} files ({len(valid_seqs)} valid sequences)")
    return filtered


# -----------------------------------------------------------------------------#
# Helper functions
# -----------------------------------------------------------------------------#


def _resize_and_save(src: Path, dst: Path, max_edge: int) -> tuple[int, int, float]:
    """Resize image so max edge = max_edge, save as PNG. Returns (H, W, scale)."""


    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(src)
    h0, w0 = img.shape[:2]

    
    scale = min(1.0, max_edge / max(h0, w0))

    
     
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(round(w0 * scale)), int(round(h0 * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )

    
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    h, w = img.shape[:2]
    return h, w, scale

 
def _default_intrinsics(h: int, w: int) -> np.ndarray:
    """Fallback pinhole K—principal point at centre, f ≈ max(H,W)."""
    f = max(h, w)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)


def _save_intrinsics(json_path: Path, K: np.ndarray) -> None:
    """Save camera intrinsics to a JSON file."""
    json_path.write_text(json.dumps({"K": K.tolist()}, indent=2))


# -----------------------------------------------------------------------------#
# Split preparation
# -----------------------------------------------------------------------------#


def prepare_split(spec: DatasetSpec, prep: PrepCfg, val_scene_split: bool = False) -> List[Dict]:
    """Resize images for a dataset split and return manifest rows."""

    
    files = sorted(spec.root.glob(spec.pattern))
    if not files:
        print(f"  No files matched for {spec.name} ({spec.root})", file=sys.stderr)
        return []

    
    if spec.split_file:
        files = _filter_by_split_file(files, spec.root, spec.split_file)
        if not files:
            print(f"  No files after split filtering for {spec.name}", file=sys.stderr)
            return []

    
    scene_groups = _group_files_by_scene(files, spec.name)
    keep = _sample_scenes(scene_groups, spec.sample_frac, spec.split, val_scene_split)

    
    
    rows: List[Dict] = []
    for fp in tqdm(keep, desc=f"{spec.name}-{spec.split}", unit="img"):
        
        scene_id = _extract_scene_id(fp, spec.name)
        
        
        
        unique_sample_id = f"{scene_id}_{fp.stem}"
        
        
        out_png = (
            prep.out_dir / spec.name / spec.split / f"{unique_sample_id.lower()}.png"
        )
        out_json = out_png.with_suffix(".json")

        
        if out_png.exists():
            h, w = cv2.imread(str(out_png), cv2.IMREAD_COLOR).shape[:2]
            
            meta = json.loads(out_json.read_text())
            h, w = meta["H"], meta["W"]
            K = np.array(meta["K"])
        else:
            
            h, w, scale = _resize_and_save(fp, out_png, prep.resize_max)
            K = _default_intrinsics(int(h / scale), int(w / scale)) * scale

            
            meta = {"K": K.tolist(), "H": h, "W": w}
            out_json.write_text(json.dumps(meta, indent=2))


        rows.append(
            {
                "dataset": spec.name,
                "split": spec.split,
                "sample_id": unique_sample_id,  
                "scene_id": scene_id,
                "image_path": str(out_png),
                "K": K,
                "H": h,
                "W": w,
            }
        )

    return rows


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#


def get_manifest(cfg_path: str | Path = "configs/data_paths.yaml") -> pd.DataFrame:
    """Process all datasets in config and return manifest DataFrame. Incremental - reuses existing files."""

    
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    
    
    prep_config = cfg.get("prep", {})
    if "out_dir" in prep_config and isinstance(prep_config["out_dir"], str):
        prep_config["out_dir"] = Path(prep_config["out_dir"])
    
    prep = PrepCfg(**prep_config)

    all_rows: List[Dict] = []

    
    for ds in cfg["datasets"]:
        spec = DatasetSpec(
            name=ds["name"],
            root=Path(ds["root"]).expanduser(),
            pattern=ds["pattern"],
            sample_frac=float(ds["sample_frac"]),
            split=ds["split"],
            split_file=ds.get("split_file"),  
        )
        val_scene_split = ds.get("val_scene_split", False)
        all_rows.extend(prepare_split(spec, prep, val_scene_split)) # make manifest

    df = pd.DataFrame(all_rows)
    out_csv = prep.out_dir / "manifest.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Manifest saved to: {out_csv}")
    return df