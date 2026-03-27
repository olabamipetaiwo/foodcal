#!/usr/bin/env bash
# 04_embed.sh — Precompute CLIP and Sentence-BERT embeddings
# Run: bash scripts/04_embed.sh
set -e
cd "$(dirname "$0")/.."

echo "==> Precomputing embeddings..."
python src/embed.py \
    --image_dir data/food101 \
    --caption_dir captions \
    --label_file data/labels.json \
    --out_dir embeddings

echo "Done. Next: run scripts/05_train.sh"
