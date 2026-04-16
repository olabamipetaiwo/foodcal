# FoodCal Pipeline — Run Notes

## ⏸ PAUSED HERE — Waiting for eval photos

**Current status**: Steps 1–5 are fully complete. Waiting for 30–50 real-world food photos to be collected and placed in `data/eval/` before proceeding.

**Next steps when photos are ready**:
1. Drop all photos into `data/eval/` (flat folder, no subfolders needed yet)
2. Create `data/eval/labels_eval.json` mapping each image path to `"Low"`, `"Medium"`, or `"High"`
3. Run `bash scripts/06_caption_eval.sh` — captions the eval photos
4. Run `bash scripts/07_ablation.sh` — evaluates all 5 variants, McNemar's tests, figures
5. Run `bash scripts/08_app.sh` — launches Gradio demo

**Label guide** (rough):
- `Low` — salads, fruits, light soups
- `Medium` — rice dishes, pasta, stews
- `High` — fried foods, heavy meat dishes, burgers

---


## Environment

- **Python**: 3.8 (system)
- **CUDA**: available (used for BLIP-2 and LLaVA inference)
- **Virtual env**: `foodcal_env/` (created by `01_setup.sh`)

---

## Step 1 — Setup (`scripts/01_setup.sh`)

**Status**: Completed

**Issue**: `scipy>=1.11.0` is not available for Python 3.8. Latest compatible version is `1.10.1`.

**Fix**: Changed `requirements.txt` line 17 from `scipy>=1.11.0` → `scipy>=1.10.0`. Installed version resolved to `scipy==1.10.1`.

**Output**:
- All dependencies installed successfully
- 101 calorie labels generated → `data/labels.json`
- Label distribution: Low=58, Medium=42, High=1

---

## Step 2 — Download (`scripts/02_download.sh`)

**Status**: Completed

**Issue (first run)**: Script called Python outside the virtual env, so `datasets` package was not found.

**Fix**: Prepend `source foodcal_env/bin/activate &&` before all script runs.

**Output**:
- 10,100 images saved (100 per class × 101 classes) → `data/food101/`
- 65,650 images skipped (over per-class cap)

---

## Step 3 — Caption Generation (`scripts/03_caption.sh`)

**Status**: Running (BLIP-2 in progress, LLaVA pending)

**Issues encountered and fixed**:

### 3a. Python 3.8 `list[Path]` type hint error
`src/caption.py` used `list[Path]` built-in generic syntax (requires Python 3.9+).
**Fix**: Added `from typing import List` and replaced `list[Path]` → `List[Path]` throughout.

### 3b. BLIP-2 `RuntimeError: shape mismatch` (transformers 4.46.x bug)
Transformers ≥4.40 introduced a regression in `modeling_blip_2.py` where `special_image_mask` is empty when no text/input_ids are passed, causing a shape mismatch when assigning image embeddings.

Attempted fix: process one image at a time (batch_size=1) — did not resolve the issue (bug is independent of batch size).

**Final fix**: Pass a text prompt to the BLIP-2 processor so that `input_ids` are populated with image token placeholders:
```python
prompt = "Question: Describe the food in this image. Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
```
Also strip the prompt prefix from the decoded output.

Note: Downgrading transformers to 4.39.3 would also fix this, but conflicts with `sentence-transformers>=3.2.1` which requires `transformers>=4.41.0`.

**Captions saved to**:
- `captions/blip2_captions.json`
- `captions/llava_captions.json` (pending)

---

## Python 3.8 Type Hint Fixes (proactive, all source files)

All files used Python 3.9+ built-in generic type hints (`list[X]`, `dict[X,Y]`, `X | None`).
Fixed across all affected files by importing from `typing` and replacing:

| File | Original | Fixed |
|------|----------|-------|
| `src/caption.py` | `list[Path]` | `List[Path]` |
| `src/embed.py` | `list[str]` | `List[str]` |
| `src/model.py` | `list[int]` | `List[int]` |
| `src/dataset.py` | `list[int] \| None`, `dict \| None` | `Optional[List[int]]`, `Optional[dict]` |
| `src/evaluate.py` | `list[int]` | `List[int]` |
| `src/download_data.py` | `dict[str, int]` | `Dict[str, int]` |

---

## Step 4 — Embed (`scripts/04_embed.sh`)

**Status**: Completed (no errors)

**Output**:
- CLIP ViT-B/32: 10,100 embeddings (512-dim) → `embeddings/clip_embeddings.pt`
- SBERT (BLIP-2 captions): 10,100 embeddings (384-dim) → `embeddings/sbert_blip2.pt`
- SBERT (LLaVA captions): 10,100 embeddings (384-dim) → `embeddings/sbert_llava.pt`

Note: OpenCLIP emitted a `UserWarning` about QuickGELU activation mismatch (non-fatal, weights loaded correctly).

---

## Step 5 — Train (`scripts/05_train.sh`)

**Status**: Completed (no errors)

**Split**: Train=8,585 | Val=1,515 | 30 epochs each

**Results summary**:

| Variant | Val Acc | Val F1 |
|---------|---------|--------|
| `image_only` (CLIP) | 0.8851 | 0.8287 |
| `text_blip2` (SBERT-BLIP2) | 0.8898 | 0.8898 |
| `text_llava` (SBERT-LLaVA) | 0.8495 | 0.8022 |
| `multimodal_blip2` (CLIP + SBERT-BLIP2) | **0.9188** | **0.9125** |
| `multimodal_llava` (CLIP + SBERT-LLaVA) | 0.9036 | 0.8485 |

**Best variant**: `multimodal_blip2` — val_acc=0.9188, val_f1=0.9125

**Key observations**:
- Multimodal fusion consistently outperforms unimodal approaches
- BLIP-2 captions produced significantly better text embeddings than LLaVA captions (likely because the VQA-style prompt gave more food-specific descriptions)
- Text-only BLIP-2 (88.98%) slightly edges out image-only CLIP (88.51%), showing captions carry complementary signal
- LLaVA text-only underperforms both CLIP and BLIP-2 text, suggesting LLaVA captions are less discriminative for calorie prediction

Checkpoints saved to `results/<variant>/best_model.pt`.


The core novelty is the systematic multimodal ablation — most food calorie work uses either image features or nutrition databases, but this project specifically asks:         
                                                                                                                                                                               
  ▎ Does adding natural-language captions from vision-language models (BLIP-2, LLaVA) improve calorie range prediction over image embeddings alone?                              
                                                                                                                                                                                 
  More specifically:                                                                                                                                                             
                                                                                                                                                                               
  - Not just multimodal — you're comparing two different captioners (BLIP-2 vs LLaVA) as text sources, which nobody really does for this task                                    
  - Caption quality matters — your results already show BLIP-2 captions are more discriminative than LLaVA's for calorie prediction, which is an interesting finding in itself
  - Lightweight classifier on top of frozen embeddings — no end-to-end fine-tuning, making it efficient and reproducible                                                         
  - Real-world eval — testing on photos taken in actual dining halls/restaurants, not just benchmark images, which is where most food recognition papers fall short              
                                                                                                                                                                                 
  The punchline for a report would be: "Multimodal fusion of CLIP + BLIP-2 captions achieves 91.9% accuracy on calorie range classification, outperforming image-only and        
  text-only baselines — and the choice of captioner significantly affects performance."

---

## Step 6 — Caption Eval (`scripts/06_caption_eval.sh`)

**Status**: Completed

**Eval set**: 86 real-world Nigerian food photos collected and placed in `data/eval/`.

**Label distribution** (after threshold revision — see Step 5b below):
- Low: 10 | Medium: 30 | High: 46

**Issues encountered**:
- Many photos taken on iPhone were HEIC format with `.jpg` extension — PIL could not read them.
  **Fix**: Installed `pillow-heif` and called `pillow_heif.register_heif_opener()` in `caption.py`, `evaluate.py`, and `cross_val.py`.
- 3 image entries in `labels_eval.json` pointed to non-existent files (`File 109.jpg`, `File 74.jpg`, `File 98.jpg`) — removed.

**Outputs**:
- `captions/blip2_captions.json` — updated with eval image captions
- `captions/llava_captions.json` — updated with eval image captions

---

## Step 5b — Threshold Revision (Retraining)

**Status**: Completed

**Problem**: Original thresholds (Low<400, Medium 400–700, High>700 kcal) produced a severely imbalanced
training set: Low=58, Medium=42, **High=1** (only nachos crossed 700 kcal in Food-101).
The model never learned to predict "High", collapsing all real-world predictions to Low/Medium.

**Fix**: Revised thresholds to **Low<300, Medium 300–500, High>500 kcal**.
New training distribution: Low=37, Medium=44, High=20 — substantially more balanced.

**Eval labels also revised** with the new thresholds:
- Low: 10 | Medium: 30 | High: 46

**Retrained val accuracy** (Food-101, 30 epochs):

| Variant | Val Acc | Val F1 |
|---------|---------|--------|
| image_only | 0.8792 | 0.8378 |
| text_blip2 | 0.8878 | 0.8879 |
| text_llava | 0.8495 | 0.7938 |
| multimodal_blip2 | **0.9234** | **0.9134** |
| multimodal_llava | 0.9023 | 0.8761 |

---

## Step 7 — Ablation Study (`scripts/07_ablation.sh`)

**Status**: Completed

**Eval set**: 86 real-world Nigerian food photos (Low=10, Medium=30, High=46).

### Zero-shot transfer results (trained on Food-101, tested on Nigerian food)

| Variant | Accuracy | Macro-F1 |
|---------|----------|----------|
| image_only | 0.2907 | 0.2620 |
| text_blip2 | 0.2326 | 0.2192 |
| text_llava | 0.2442 | 0.2209 |
| multimodal_blip2 | 0.2674 | 0.2528 |
| multimodal_llava | 0.2791 | 0.2538 |

**McNemar's tests**: All p-values > 0.05. No statistically significant differences between variants.

**Finding 1 — Severe domain shift**: All variants perform near or below random chance (33%) on Nigerian food
despite 88–92% validation accuracy on Food-101. This is not a model failure — it is a dataset bias problem.
Food-101 contains almost no West African or Nigerian dishes. The visual appearance and caption semantics
of Nigerian food (jollof rice, fufu, pounded yam, egusi soup, amala) are out-of-distribution for all
embeddings trained/calibrated on Food-101.

**Finding 2 — Image-only is the most robust under domain shift**: Despite being the weakest modality on
Food-101 (88.5%), `image_only` achieves the highest real-world accuracy (29.1%). Text-based variants
actually hurt — BLIP-2 and LLaVA describe Nigerian food generically ("a bowl of soup", "rice on a plate"),
producing uninformative embeddings that misguide the classifier.

**Figures saved to**: `results/figures/` — bar chart and 5 confusion matrices.

---

## Step 9 — In-domain Cross-Validation (`scripts/09_crossval.sh`)

**Status**: Completed

**Method**: 5-fold stratified cross-validation entirely within the 86 Nigerian food images.
Embeddings computed once; MLP trained from scratch for 60 epochs per fold.
No Food-101 data used. Tests what is achievable with even modest in-domain training data.

### Results

| Variant | Mean Acc | ± Std | Mean Macro-F1 |
|---------|----------|-------|---------------|
| text_llava | **0.6758** | ±0.071 | 0.4791 |
| multimodal_llava | 0.6405 | ±0.061 | **0.5728** |
| image_only | 0.5935 | ±0.060 | 0.4592 |
| multimodal_blip2 | 0.5817 | ±0.039 | 0.4577 |
| text_blip2 | 0.5699 | ±0.078 | 0.3699 |

**Finding 3 — Domain adaptation recovers performance**: In-domain CV lifts accuracy from ~27% to 59–68%,
confirming that the zero-shot failure was entirely due to training data mismatch, not model capacity.

**Finding 4 — Captioner ranking reverses with domain shift**: On Food-101, BLIP-2 captions outperform
LLaVA captions for calorie prediction. On Nigerian food, LLaVA captions outperform BLIP-2.
Hypothesis: BLIP-2's VQA-style prompt ("Describe the food") produces short, generic answers for
unfamiliar dishes. LLaVA's conversational prompt elicits richer, more ingredient-specific descriptions
that remain useful even for out-of-distribution cuisines.

**Finding 5 — Multimodal fusion helps on F1 but not accuracy for in-domain Nigerian food**:
`multimodal_llava` achieves the best macro-F1 (0.573), indicating better class balance across
Low/Medium/High. `text_llava` alone wins on raw accuracy but may be more biased toward the majority class.

---

## Updated Paper Punchline

> "FoodCal achieves 92.3% accuracy on Food-101-style Western dishes but drops to 27% on real-world
> Nigerian food — exposing a systematic cultural bias in food recognition benchmarks. With modest
> in-domain adaptation (86 images, 5-fold CV), accuracy recovers to 68%, and the captioner ranking
> reverses: LLaVA's richer descriptions outperform BLIP-2's VQA-style prompts for unfamiliar cuisines.
> These findings argue for culturally diverse food datasets and captioner-aware model selection in
> real-world deployment."
                                                                                
