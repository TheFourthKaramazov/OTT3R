#!/usr/bin/env python3
"""
Generate image manifest for OTT3R training.

Usage:
    python utils/generate_manifest.py --config configs/data_paths.yaml
"""

import argparse
import sys
import shutil
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import yaml
from ott3r.data.datasets import get_manifest


def main():
    parser = argparse.ArgumentParser(description="Generate dataset manifest for OTT3R")
    parser.add_argument("--config", type=str, default="configs/data_paths.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Generating manifest from {args.config}")

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    df = get_manifest(config_path)

    cfg = yaml.safe_load(config_path.read_text())
    out_dir = Path(cfg.get("prep", {}).get("out_dir", "processed_data/images"))
    manifest_path = out_dir / "manifest.csv"

    if args.output and Path(args.output) != manifest_path:
        shutil.copy2(manifest_path, args.output)
        print(f"Copied to: {args.output}")

    print(f"\nTotal: {len(df)} images")
    print("\nBreakdown:")
    for dataset, group in df.groupby('dataset'):
        scenes = group['scene_id'].nunique()
        print(f"  {dataset}: {scenes} scenes, {len(group)} images")


if __name__ == "__main__":
    main()
