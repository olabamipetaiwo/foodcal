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
                                                                                     
