#!/usr/bin/env bash
# 02_download.sh — Download Food-101 dataset
# Run:   bash scripts/02_download.sh
# Usage: bash scripts/02_download.sh [max_per_class]
#   max_per_class: images per class (default 100, use 0 for all ~750)
set -e
cd "$(dirname "$0")/.."
MAX_PER_CLASS=${1:-100}

echo "==> Downloading Food-101 (max_per_class=$MAX_PER_CLASS)..."
python src/download_data.py --max_per_class "$MAX_PER_CLASS"

echo "Done. Next: run scripts/03_caption.sh"
