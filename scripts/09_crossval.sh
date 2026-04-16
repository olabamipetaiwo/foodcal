#!/usr/bin/env bash
# 09_crossval.sh — 5-fold cross-validation on real-world eval images
# Trains and evaluates entirely within the Nigerian food domain.
# Run: bash scripts/09_crossval.sh
set -e
cd "$(dirname "$0")/.."

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "==> Running 5-fold cross-validation on eval images..."
python src/cross_val.py

echo "Done. Results saved to results/crossval_metrics.json"
