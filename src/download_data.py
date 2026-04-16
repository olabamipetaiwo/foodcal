"""
download_data.py

Downloads Food-101 from Hugging Face and organises it into:
    data/food101/<class_name>/<index>.jpg

Up to MAX_PER_CLASS images are saved per class (default 100, giving ~10,100 total).
Skips images that are already on disk so the script is safe to re-run.

Usage:
    python src/download_data.py                        # 100 images per class
    python src/download_data.py --max_per_class 50     # smaller subset
    python src/download_data.py --max_per_class 0      # all images (~750 per class)
"""

import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm
from typing import Dict


def parse_args():
    p = argparse.ArgumentParser(description="Download and organise Food-101")
    p.add_argument(
        "--out_dir", default="data/food101",
        help="Root output directory (default: data/food101)",
    )
    p.add_argument(
        "--max_per_class", type=int, default=100,
        help="Max images per class. 0 = all (default: 100)",
    )
    p.add_argument(
        "--split", default="train",
        choices=["train", "validation", "all"],
        help="Dataset split to download (default: train)",
    )
    return p.parse_args()


def download_food101(out_dir: str, max_per_class: int = 100, split: str = "train"):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run:  pip install datasets")
        sys.exit(1)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validation"] if split == "all" else [split]
    total_saved = 0

    for sp in splits:
        print(f"\nLoading Food-101 split='{sp}' from Hugging Face Hub ...")
        ds = load_dataset("food101", split=sp, trust_remote_code=True)
        label_names = ds.features["label"].names
        print(f"  {len(ds):,} images | {len(label_names)} classes")

        # Count already-saved images per class
        counts: Dict[str, int] = {}
        for cls in label_names:
            cls_dir = out_root / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            counts[cls] = len(list(cls_dir.glob("*.jpg")))

        skipped = 0
        for item in tqdm(ds, desc=f"Saving {sp}", unit="img"):
            cls = label_names[item["label"]]
            if max_per_class and counts[cls] >= max_per_class:
                skipped += 1
                continue
            out_path = out_root / cls / f"{counts[cls]:05d}.jpg"
            if not out_path.exists():
                try:
                    item["image"].convert("RGB").save(out_path, format="JPEG", quality=90)
                except Exception as e:
                    print(f"  Warning: could not save {out_path}: {e}")
                    continue
            counts[cls] += 1
            total_saved += 1

        print(f"  Saved {total_saved:,} images | Skipped {skipped:,} (over cap)")

    # Summary
    classes_found = [d.name for d in out_root.iterdir() if d.is_dir()]
    total_on_disk = sum(len(list((out_root / c).glob("*.jpg"))) for c in classes_found)
    print(f"\nDone. {len(classes_found)} classes | {total_on_disk:,} images on disk → {out_root}")
    return total_on_disk


if __name__ == "__main__":
    args = parse_args()
    download_food101(
        out_dir=args.out_dir,
        max_per_class=args.max_per_class,
        split=args.split,
    )
