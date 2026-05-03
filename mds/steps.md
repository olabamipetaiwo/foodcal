# Project Steps

## Step 1 — Add eval photos
- [x] Collect 30–50 real-world food photos
- [x] Place all photos flat in `data/eval/` (no subfolders)
- [x] Create `data/eval/labels_eval.json` mapping each image path to `"Low"`, `"Medium"`, or `"High"`

## Step 2 — Caption eval photos
- [x] Run `bash scripts/06_caption_eval.sh`
- [x] Outputs: captions for eval photos (BLIP-2 + LLaVA)

## Step 3 — Ablation + evaluation
- [x] Run `bash scripts/07_ablation.sh`
- [x] Outputs: accuracy/F1 for all 5 variants on real-world eval set, McNemar's test results, figures
- [x] Run `bash scripts/09_crossval.sh` — 5-fold in-domain CV on 86 Nigerian food photos

## Step 4 — Launch + deploy demo
- [x] Run `bash scripts/08_app.sh`
- [x] Deployed to HF Spaces: https://huggingface.co/spaces/teeola4/foodcalcounter

---

## Step 5 — Performance improvements

### 5a — Data augmentation (quick win)
- [x] Add Gaussian noise to embedding features during training in `src/cross_val.py` (`augment_features()`, cosine LR decay)
- [ ] Rerun `bash scripts/09_crossval.sh` and compare accuracy vs baseline (68%)

### 5b — Better caption prompts (quick win)
- [x] Updated LLaVA prompt in `src/caption.py` and `app.py` to focus on calorie-relevant factors (frying, sauces, portions, fatty meats)
- [x] Regenerate eval captions: `CUDA_VISIBLE_DEVICES=1 bash scripts/06_caption_eval.sh llava`
- [ ] Rerun CV: `bash scripts/09_crossval.sh`

### 5c — Ensemble top variants (quick win)
- [x] Added ensemble variant to `src/cross_val.py`: averages softmax probs from `text_llava` + `multimodal_llava` + `image_only` per fold
- [ ] Rerun CV (ensemble runs automatically with `--variant all`): `bash scripts/09_crossval.sh`

### 5d — Collect more Nigerian food images (moderate effort)
- [ ] Collect 200–300 additional labeled Nigerian food photos
- [ ] Add to `data/eval/` and update `labels_eval.json`
- [ ] Rerun CV: `bash scripts/09_crossval.sh`

### 5e — Direct classification prompt (moderate effort)
- [ ] Update LLaVA prompt to ask directly:
  *"Is this meal Low (<300 kcal), Medium (300–500 kcal), or High (>500 kcal) calorie? Explain why."*
- [ ] Use response as feature instead of generic description
- [ ] Regenerate captions, rerun CV

### 5f — Hyperparameter tuning (moderate effort)
- [ ] Increase CV epochs from 60 → 150 with cosine LR decay
- [ ] Try heavier dropout (0.4–0.5) for small dataset
- [ ] Compare CV results

### 5g — Fine-tune CLIP on Nigerian food (higher effort)
- [ ] Unfreeze CLIP visual encoder and fine-tune on in-domain images
- [ ] Requires careful regularisation to avoid overfitting on 86 images

---

## Label guide (revised thresholds)
- `Low` — under 300 kcal: salads, fruits, pap, cornflakes
- `Medium` — 300–500 kcal: plain rice, simple pasta, beans & bread
- `High` — over 500 kcal: fried foods, heavy meat dishes, loaded plates


<!-- URL - https://huggingface.co/spaces/teeola4/foodcalcounter -->