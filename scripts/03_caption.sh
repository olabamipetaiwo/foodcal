#!/usr/bin/env bash
# 03_caption.sh — Generate BLIP-2 and LLaVA captions for training images
# Run:   bash scripts/03_caption.sh
# Usage: bash scripts/03_caption.sh [model] [image_dir]
#   model:     blip2 | llava | both (default: both)
#   image_dir: path to images (default: data/food101)
set -e
cd "$(dirname "$0")/.."
MODEL=${1:-both}
IMAGE_DIR=${2:-data/food101}

echo "==> Generating captions (model=$MODEL, dir=$IMAGE_DIR)..."
python src/caption.py --model "$MODEL" --image_dir "$IMAGE_DIR"

echo "Done. Next: run scripts/04_embed.sh"
