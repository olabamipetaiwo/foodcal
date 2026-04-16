"""
embed.py

Precomputes and caches embeddings for all food images and captions:
  - CLIP (ViT-B/32)       → embeddings/clip_embeddings.pt
  - Sentence-BERT (BLIP-2 captions) → embeddings/sbert_blip2.pt
  - Sentence-BERT (LLaVA captions)  → embeddings/sbert_llava.pt

Each .pt file is a dict:
    {
        "keys":       List[str],   # image paths (same order for CLIP & SBERT)
        "embeddings": Tensor,      # shape (N, D)
        "labels":     List[str],   # Low / Medium / High
    }

Usage:
    python src/embed.py --image_dir data/food101 \
                        --caption_dir captions \
                        --label_file data/labels.json \
                        --out_dir embeddings
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from typing import List


LABEL2IDX = {"Low": 0, "Medium": 1, "High": 2}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_labels(label_file: str) -> dict:
    with open(label_file) as f:
        return json.load(f)  # {food_class: "Low"/"Medium"/"High"}


def food_class_from_path(path: str) -> str:
    """Extract Food-101 class name from image path."""
    # Expected layout: .../food101/<class_name>/<image>.jpg
    return Path(path).parent.name


# ---------------------------------------------------------------------------
# CLIP embeddings
# ---------------------------------------------------------------------------

def embed_clip(image_paths: List[str], label_map: dict, out_path: str, batch_size: int = 32):
    import open_clip
    from PIL import Image

    device = get_device()
    print(f"Loading CLIP ViT-B/32 on {device} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()

    keys, embeddings, labels = [], [], []
    todo = image_paths

    for i in tqdm(range(0, len(todo), batch_size), desc="CLIP"):
        batch_paths = todo[i: i + batch_size]
        imgs = []
        valid_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
                valid_paths.append(p)
            except Exception as e:
                print(f"  Skipping {p}: {e}")

        if not imgs:
            continue

        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalise

        for path, feat in zip(valid_paths, feats):
            cls = food_class_from_path(path)
            if cls not in label_map:
                continue
            keys.append(str(path))
            embeddings.append(feat.cpu())
            labels.append(label_map[cls])

    out = {
        "keys": keys,
        "embeddings": torch.stack(embeddings),
        "labels": labels,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out, out_path)
    print(f"CLIP: saved {len(keys)} embeddings ({out['embeddings'].shape}) → {out_path}")
    return out


# ---------------------------------------------------------------------------
# Sentence-BERT embeddings
# ---------------------------------------------------------------------------

def embed_sbert(caption_file: str, label_map: dict, out_path: str, batch_size: int = 64):
    from sentence_transformers import SentenceTransformer

    with open(caption_file) as f:
        captions: dict = json.load(f)

    print(f"Loading all-MiniLM-L6-v2 ...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    keys, texts, labels = [], [], []
    for path, caption in captions.items():
        cls = food_class_from_path(path)
        if cls not in label_map:
            continue
        keys.append(path)
        texts.append(caption)
        labels.append(label_map[cls])

    print(f"Encoding {len(texts)} captions ...")
    embs = sbert.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    out = {
        "keys": keys,
        "embeddings": embs.cpu(),
        "labels": labels,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out, out_path)
    print(f"SBERT: saved {len(keys)} embeddings ({out['embeddings'].shape}) → {out_path}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Precompute CLIP and Sentence-BERT embeddings")
    p.add_argument("--image_dir", default="data/food101")
    p.add_argument("--caption_dir", default="captions")
    p.add_argument("--label_file", default="data/labels.json")
    p.add_argument("--out_dir", default="embeddings")
    p.add_argument("--clip_batch", type=int, default=32)
    p.add_argument("--sbert_batch", type=int, default=64)
    p.add_argument("--skip_clip", action="store_true")
    p.add_argument("--skip_sbert", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    label_map = load_labels(args.label_file)

    if not args.skip_clip:
        from pathlib import Path as _Path
        image_paths = sorted(
            str(p) for p in _Path(args.image_dir).rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        print(f"Found {len(image_paths)} images for CLIP")
        embed_clip(
            image_paths, label_map,
            out_path=os.path.join(args.out_dir, "clip_embeddings.pt"),
            batch_size=args.clip_batch,
        )

    if not args.skip_sbert:
        for name, fname in [("blip2", "blip2_captions.json"), ("llava", "llava_captions.json")]:
            cap_path = os.path.join(args.caption_dir, fname)
            if not os.path.exists(cap_path):
                print(f"Caption file not found: {cap_path} — skipping {name}")
                continue
            embed_sbert(
                cap_path, label_map,
                out_path=os.path.join(args.out_dir, f"sbert_{name}.pt"),
                batch_size=args.sbert_batch,
            )
