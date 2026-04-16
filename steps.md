# Remaining Steps

## Step 1 — Add eval photos
- [ ] Collect 30–50 real-world food photos
- [ ] Place all photos flat in `data/eval/` (no subfolders)
- [ ] Create `data/eval/labels_eval.json` mapping each image path to `"Low"`, `"Medium"`, or `"High"`

## Step 2 — Caption eval photos
- [ ] Run `bash scripts/06_caption_eval.sh`
- [ ] Outputs: captions for eval photos (BLIP-2 + LLaVA)

## Step 3 — Ablation + evaluation
- [ ] Run `bash scripts/07_ablation.sh`
- [ ] Outputs: accuracy/F1 for all 5 variants on real-world eval set, McNemar's test results, figures

## Step 4 — Launch demo
- [ ] Run `bash scripts/08_app.sh`
- [ ] Outputs: Gradio demo — upload a food photo, get predicted calorie range

---

## Label guide
- `Low` — salads, fruits, light soups
- `Medium` — rice dishes, pasta, stews
- `High` — fried foods, heavy meat dishes, burgers
