#!/usr/bin/env bash
# 08_app.sh — Launch the Gradio web app locally
# Run: bash scripts/08_app.sh
# Automatically loads the best-performing variant from results/metrics.json
set -e
cd "$(dirname "$0")/.."
echo "==> Starting Gradio app..."
echo "    Open http://localhost:7860 in your browser"
python app.py
