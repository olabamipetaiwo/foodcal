#!/usr/bin/env bash
# 01_setup.sh — Install dependencies and generate calorie labels
# Run: bash scripts/01_setup.sh
set -e
cd "$(dirname "$0")/.."

echo "==> Installing dependencies..."
pip install -r requirements.txt

echo "==> Generating calorie label map..."
python src/label_mapping.py

echo "Done. Next: run scripts/02_download.sh"
