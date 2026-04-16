---
title: FoodCal — Calorie Range Estimator
emoji: 🍽️
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: "4.20.0"
app_file: app.py
pinned: false
license: mit
---

# FoodCal — Automated Calorie Range Estimator

Upload a food photo and get an instant calorie range estimate powered by multimodal AI.

## How it works

1. You upload a food photo
2. BLIP-2 generates a natural-language caption describing the dish
3. CLIP extracts visual features from the image
4. A trained MLP classifies the combined features into a calorie range
5. A weighted estimate and optional USDA FoodData Central lookup provide a kcal figure

## Calorie classes

| Class | Range | Examples |
|-------|-------|---------|
| Low | < 300 kcal | Salads, fruits, light soups |
| Medium | 300–500 kcal | Rice dishes, pasta, stews |
| High | > 500 kcal | Fried foods, heavy meat dishes |

## Model

Best variant: `multimodal_blip2` — CLIP ViT-B/32 image embeddings + SBERT embeddings of BLIP-2 captions, classified by a 2-layer MLP.

Val accuracy on Food-101: **92.3%**

## Research findings

- Models trained on Food-101 (Western dishes) drop to ~27% accuracy on Nigerian food — exposing cultural bias in food AI benchmarks
- With in-domain adaptation (5-fold CV on 86 Nigerian photos), accuracy recovers to **68%**
- LLaVA captions outperform BLIP-2 for unfamiliar cuisines; BLIP-2 outperforms LLaVA on Western food

*This is a proof-of-concept research tool. Not intended for clinical or dietary use.*
