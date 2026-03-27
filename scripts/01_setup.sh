#!/usr/bin/env bash
# 01_setup.sh — Create virtual environment, install dependencies, generate calorie labels
# Run: bash scripts/01_setup.sh
set -e
cd "$(dirname "$0")/.."

# Create virtual environment if it doesn't already exist
if [ ! -d "foodcal_env" ]; then
    echo "==> Creating virtual environment (foodcal_env)..."
    python3 -m venv foodcal_env
else
    echo "==> Virtual environment already exists, skipping creation."
fi

# Activate
echo "==> Activating foodcal_env..."
source foodcal_env/bin/activate

echo "==> Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt

echo "==> Generating calorie label map..."
python src/label_mapping.py

echo ""
echo "Done. To activate the environment in your terminal run:"
echo "  source foodcal/bin/activate"
echo ""
echo "Next: run scripts/02_download.sh"
