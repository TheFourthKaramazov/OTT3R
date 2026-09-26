# OTT3R: Multi-View 3D Reconstruction and Fast Dataset Generation at 1% Compute

**Brandon Leblanc, Charalambos Poullis**

Concordia University, ICT Lab

**Paper** (ACCV 2026 — to appear) · [**arXiv**](https://arxiv.org/abs/XXXX.XXXXX) · [**Project Page**](https://thefourthkaramazov.github.io/OTT3R/)

---

## Abstract

Feed-forward 3D reconstruction models have achieved impressive performance by scaling model and dataset size, but their cost excludes most research groups and precludes edge deployment. Additionally, generating 3D supervision without sensors still relies on slow, unreliable Structure-from-Motion, as the community lacks a COLMAP-like system for neural 3D pseudo-label generation. We present OTT3R (RGB-Only Tiny Transformer for 3D Reconstruction), a knowledge distillation framework that addresses both problems on a single workstation equipped with 2 GPUs. Distilling $\pi^3$ (959M parameters) into a 102M-parameter student yields 9.4$\times$ compression and up to 7$\times$ faster inference, trained at 1.6\% of VGGT's training compute. An integrated pseudo-label pipeline offers a reliable, high-throughput alternative to COLMAP, generating dense per-pixel point maps and SE(3) camera poses for a 667K-image corpus in 3.5 hours on two commodity GPUs and succeeding on every sequence we tested, including those where COLMAP fails. The general student tracks the teacher on in-distribution monocular depth and, zero-shot, outperforms COLMAP on 7-Scenes and on DTU completion, but it does not replace the teacher on out-of-distribution multi-view geometry. The deployable artifact is the domain-specialized student: after specialization at 0.2\% compute, it is 4$\times$ more accurate than COLMAP on 7-Scenes at 980$\times$ throughput, with near-teacher completion.

---

## Installation

### 1. Clone with submodules
```bash
git clone --recurse-submodules git@github.com:TheFourthKaramazov/OTT3R.git
cd OTT3R
```

### 2. Create the conda environment
```bash
conda env create -f environment.yml
conda activate ott3r
```

---

## Domain-Specific Training (7-Scenes)

This example trains a domain-specialized model on 7-Scenes.

### Step 1: Download 7-Scenes

Download from the [official source](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and extract to `dataset/7scenes/`.

### Step 2: Configure data paths

Update the `root` path in `configs/data_paths_7scenes.yaml` to your 7-Scenes location.

### Step 3: Generate manifest

```bash
python utils/generate_manifest.py --config configs/data_paths_7scenes.yaml
```

### Step 4: Generate teacher cache

```bash
python ott3r/teacher/ott3r_generator.py \
    --manifest processed_data/images_7scenes/manifest.csv \
    --cache-dir caches/teacher_cache_7scenes \
    --num-gpus 2 \
    --num-workers 4
```

### Step 5: Train

```bash
python ott3r/train.py --config configs/ott3r_7scenes.yaml
```

---

## General Training (Multi-Dataset)

For training on mixed datasets, you'll need to download the datasets yourself and configure the paths.

### Datasets used in the paper

| Dataset | Source |
|---------|--------|
| CO3D | [GitHub](https://github.com/facebookresearch/co3d) |
| MegaDepth | [Project Page](https://www.cs.cornell.edu/projects/megadepth/) |
| BlendedMVS | [GitHub](https://github.com/YoYo000/BlendedMVS) |
| ARKitScenes | [GitHub](https://github.com/apple/ARKitScenes) |
| ScanNet++ | [Project Page](https://kaldir.vc.in.tum.de/scannetpp/) |
| Habitat | [GitHub](https://github.com/facebookresearch/habitat-sim) |

> **Note:** These datasets total many terabytes. We subsampled to ~400GB by increasing the sampling stride based on scene size (scenes with 1000 images get a larger stride than scenes with 100), maintaining scene diversity while reducing redundancy and increasing variance between consecutive frames in order to effictevly learn pose estimation. Obviously, this only works for datasets that are sequential.

### Step 1: Configure data paths

Edit `configs/data_paths.yaml` to match your dataset locations and structure. Each dataset entry requires:
- `name`: Dataset identifier
- `root`: Path to dataset root
- `pattern`: Glob pattern for images
- `sample_frac`: Fraction of scenes to use
- `split`: train/val/test

You may need to modify `ott3r/data/datasets.py` to handle your dataset's directory structure (see `_extract_scene_id` function).

### Step 2: Generate manifest

```bash
python utils/generate_manifest.py --config configs/data_paths.yaml
```

### Step 3: Generate teacher cache

```bash
python ott3r/teacher/ott3r_generator.py \
    --manifest processed_data/images/manifest.csv \
    --cache-dir caches/teacher_cache \
    --num-gpus 2 \
    --num-workers 4
```

### Step 4: Train

```bash
python ott3r/train.py --config configs/ott3r_base.yaml
```

---

## Inference

```bash
python utils/inference.py path/to/images/ --checkpoint checkpoints/ott3r/last.ckpt
```

Output: `{scene}_pointcloud.ply` in `inference_results/`.

---

## Project Structure

```
OTT3R/
├── configs/
│   ├── ott3r_base.yaml        # General training
│   ├── ott3r_7scenes.yaml     # Domain-specific (paper results)
│   ├── data_paths.yaml
│   └── data_paths_7scenes.yaml
├── ott3r/
│   ├── student/
│   │   ├── model.py           # OTT3R architecture
│   │   ├── loss.py            # Distillation losses
│   │   └── distillation_module.py
│   ├── teacher/
│   │   ├── ott3r_generator.py # Cache generation
│   │   └── rle_helpers.py
│   ├── data/
│   │   ├── datasets.py        # Manifest generation
│   │   └── cached_sample_dataset.py
│   └── train.py
├── utils/
│   ├── generate_manifest.py
│   └── inference.py
├── external/
│   └── pi3/                   # Pi3 teacher (submodule)
├── caches/
├── checkpoints/
├── processed_data/
└── logs/
```

---

## Citation

```bibtex
@inproceedings{leblanc2026ott3r,
  title={OTT3R: Multi-View 3D Reconstruction and Fast Dataset Generation at 1\% Compute},
  author={Leblanc, Brandon and Poullis, Charalambos},
  booktitle={Proceedings of the 18th Asian Conference on Computer Vision (ACCV)},
  year={2026},
  note={To appear}
}
```

If you use the Pi3 teacher model:

```bibtex
@inproceedings{fan2025pi3,
  title={Pi3: Permutation-Invariant Point-wise Learning for 3D Reconstruction},
  author={Fan, Yiming and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

---

## Acknowledgments

Developed at the Immersive and Creative Technologies Lab (ICT Lab), Concordia University.

We thank the authors of [Pi3](https://github.com/yyfz/Pi3), [DUNE](https://github.com/naver/dune), and [DINOv2](https://github.com/facebookresearch/dinov2) for making their work publicly available.

---

## License

Code is released under the MIT License. Model weights distilled from Pi3 are CC BY-NC 4.0 (non-commercial) due to Pi3's license. Weights distilled from a different teacher may have more permissive licensing. See [LICENSE](LICENSE).
