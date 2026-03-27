# FoodCal — Running Guide

Step-by-step instructions to go from raw data to trained models to Gradio deployment.

---

## Prerequisites

Python 3.10+ recommended. `01_setup.sh` handles everything below automatically.

```bash
# Create and activate the virtual environment
python3 -m venv foodcal_env
source foodcal_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Or just run:

```bash
bash scripts/01_setup.sh
```

---

## Step 1 — Generate Calorie Labels

Assigns Low / Medium / High labels to each Food-101 category using USDA references.
Writes `data/labels.json`.

```bash
python src/label_mapping.py
```

---

## Step 2 — Download Food-101

Download the Food-101 dataset and place it under `data/food101/` so the layout is:

```
data/food101/
    apple_pie/
        1234.jpg
        5678.jpg
        ...
    baby_back_ribs/
        ...
```

You can download it via the Hugging Face `datasets` library:

```python
from datasets import load_dataset
import os, shutil
from pathlib import Path

ds = load_dataset("food101", split="train")
out_root = Path("data/food101")

for item in ds:
    label = ds.features["label"].int2str(item["label"])
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    # count existing to name file
    idx = len(list(out_dir.glob("*.jpg")))
    item["image"].save(out_dir / f"{idx:05d}.jpg")
```

Or use torchvision:

```python
import torchvision
torchvision.datasets.Food101(root="data", download=True)
# then reorganise into data/food101/<class>/<image>.jpg
```

> Tip: The full dataset is ~5 GB. If you want a faster run, pass `--limit 2000`
> to caption.py and embed.py to use a smaller subset.

---

## Step 3 — Generate Captions

Runs BLIP-2 and LLaVA over all training images and writes:
- `captions/blip2_captions.json`
- `captions/llava_captions.json`

Captioning is incremental — if interrupted, it resumes where it left off.

```bash
# Both captioners (recommended — takes several hours on CPU/MPS)
python src/caption.py --model both --image_dir data/food101

# Or separately
python src/caption.py --model blip2 --image_dir data/food101
python src/caption.py --model llava  --image_dir data/food101

# Quick test with 200 images
python src/caption.py --model both --image_dir data/food101 --limit 200
```

> LLaVA (7B) requires ~14 GB RAM. On machines with less memory, run BLIP-2 only
> and skip the LLaVA variants during training.

---

## Step 4 — Precompute Embeddings

Encodes all images with CLIP and all captions with Sentence-BERT. Writes:
- `embeddings/clip_embeddings.pt`
- `embeddings/sbert_blip2.pt`
- `embeddings/sbert_llava.pt`

```bash
python src/embed.py \
    --image_dir data/food101 \
    --caption_dir captions \
    --label_file data/labels.json \
    --out_dir embeddings
```

To skip one modality:

```bash
python src/embed.py --skip_clip       # Sentence-BERT only
python src/embed.py --skip_sbert      # CLIP only
```

---

## Step 5 — Train All 5 Variants

Trains each variant with a stratified 85/15 train/val split and saves the best
checkpoint (by val accuracy) per variant to `results/<variant>/best_model.pt`.

```bash
# Train all 5 variants (default 30 epochs)
python src/train.py

# Train a single variant
python src/train.py --variant multimodal_llava

# Tune hyperparameters
python src/train.py --epochs 50 --lr 5e-4 --hidden_layers 1

# Quick smoke-test
python src/train.py --epochs 5 --variant image_only
```

Available variants:
- `image_only`
- `text_blip2`
- `text_llava`
- `multimodal_blip2`
- `multimodal_llava`

Training writes:
- `results/<variant>/best_model.pt`
- `results/<variant>/history.json`
- `results/best_variant.json` (updated after each run)

---

## Step 6 — Collect Real-World Eval Photos

Take 30–50 food photos at a dining hall or restaurant with your phone.

Organise them in either of two ways:

**Option A — by class (mirrors Food-101 layout):**
```
data/eval/
    pizza/
        img1.jpg
    salad/
        img2.jpg
```
Labels are inferred from the folder name using `data/labels.json`.

**Option B — flat folder with a label file:**
```
data/eval/
    img1.jpg
    img2.jpg
    labels_eval.json     ← {"data/eval/img1.jpg": "High", ...}
```

---

## Step 7 — Generate Eval Captions

Same as Step 3 but pointed at the eval directory:

```bash
python src/caption.py --model both --image_dir data/eval
```

Captions are appended to the same `captions/blip2_captions.json` and
`captions/llava_captions.json` files.

---

## Step 8 — Run Ablation Study

Evaluates all trained variants on the eval set, runs McNemar's pairwise tests,
and produces figures.

```bash
python src/ablation.py \
    --eval_dir data/eval \
    --label_file data/labels.json
```

Outputs:
- `results/metrics.json` — accuracy, macro-F1, McNemar results, best variant
- `results/figures/ablation_bar_chart.png` — grouped bar chart
- `results/figures/cm_<variant>.png` — confusion matrix per variant
- `results/<variant>/eval_results.json` — per-variant detailed results

---

## Step 9 — Launch the Gradio App

Loads the best-performing variant automatically from `results/metrics.json`.

```bash
python app.py
```

Open `http://localhost:7860` in your browser. Upload a food photo and get
an instant calorie range prediction with per-class confidence scores.

---

## Step 10 — Deploy to Hugging Face Spaces

Once you have confirmed the best variant locally:

1. Create a new Space at huggingface.co/spaces (SDK: Gradio)
2. Upload model checkpoints to HF Hub:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="results/<best_variant>",
    repo_id="your-username/foodcal",
    repo_type="model",
)
```

3. Push the code (including `app.py`, `src/`, `requirements.txt`) to the Space repo.
4. The Space will install dependencies and run `app.py` automatically.

---

## Quick Reference — Full Run

Use the shell scripts in `scripts/` — no need to type Python commands manually.

```bash
bash scripts/01_setup.sh                  # install deps + generate labels
bash scripts/02_download.sh               # download Food-101 (100 imgs/class)
bash scripts/03_caption.sh                # generate BLIP-2 + LLaVA captions
bash scripts/04_embed.sh                  # precompute embeddings
bash scripts/05_train.sh                  # train all 5 variants
# — collect 30-50 eval photos → data/eval/ —
bash scripts/06_caption_eval.sh           # caption your eval photos
bash scripts/07_ablation.sh              # run ablation + generate figures
bash scripts/08_app.sh                    # launch Gradio app
```

### Script options

| Script | Options |
|---|---|
| `02_download.sh` | `[max_per_class]` — e.g. `bash scripts/02_download.sh 50` |
| `03_caption.sh` | `[model] [image_dir]` — e.g. `bash scripts/03_caption.sh blip2` |
| `05_train.sh` | `[variant] [epochs]` — e.g. `bash scripts/05_train.sh multimodal_llava 50` |
| `06_caption_eval.sh` | `[model]` — e.g. `bash scripts/06_caption_eval.sh blip2` |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| LLaVA OOM error | Run with BLIP-2 only: `--model blip2`. Skip `text_llava` and `multimodal_llava` variants. |
| `clip_embeddings.pt` missing | Re-run `embed.py` without `--skip_clip` |
| Variant skipped during training | Check that the required `.pt` files exist in `embeddings/` |
| Gradio app says "Model not loaded" | Run `train.py` first to generate checkpoints in `results/` |
| McNemar skipped for a pair | Both variants must have been evaluated on the exact same eval set |
