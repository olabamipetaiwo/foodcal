#!/usr/bin/env bash
# 07_ablation.sh — Run full ablation study across all trained variants
# Run: bash scripts/07_ablation.sh
# Outputs metrics, bar charts, and confusion matrices to results/
set -e
cd "$(dirname "$0")/.."

echo "==> Running ablation study..."
python src/ablation.py \
    --eval_dir data/eval \
    --label_file data/labels.json

echo "Done. Check results/metrics.json and results/figures/"
echo "Next: run scripts/08_app.sh to launch the Gradio app"
