#!/usr/bin/env bash
# 05_train.sh — Train all 5 model variants
# Run:   bash scripts/05_train.sh
# Usage: bash scripts/05_train.sh [variant] [epochs]
#   variant: image_only | text_blip2 | text_llava | multimodal_blip2 | multimodal_llava | all
#   epochs:  number of training epochs (default: 30)
set -e
cd "$(dirname "$0")/.."
VARIANT=${1:-all}
EPOCHS=${2:-30}

echo "==> Training variant=$VARIANT for $EPOCHS epochs..."
python src/train.py --variant "$VARIANT" --epochs "$EPOCHS"

echo "Done. Next: collect eval photos → data/eval/, then run scripts/06_caption_eval.sh"
