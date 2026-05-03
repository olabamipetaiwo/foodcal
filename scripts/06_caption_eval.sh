#!/usr/bin/env bash
# 06_caption_eval.sh — Generate captions for your real-world eval photos
# Place your 30-50 dining photos under data/eval/ before running this.
# Run:   bash scripts/06_caption_eval.sh
# Usage: bash scripts/06_caption_eval.sh [model]
#   model: blip2 | llava | both (default: both)
set -e
cd "$(dirname "$0")/.."
MODEL=${1:-both}

if [ ! -d "data/eval" ] || [ -z "$(ls -A data/eval 2>/dev/null)" ]; then
    echo "ERROR: data/eval/ is empty. Add your food photos first."
    exit 1
fi

echo "==> Generating eval captions (model=$MODEL, force regenerate)..."
python src/caption.py --model "$MODEL" --image_dir data/eval --force

echo "Done. Next: run scripts/07_ablation.sh"
