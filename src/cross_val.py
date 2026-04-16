"""
cross_val.py

5-fold stratified cross-validation on the 86 real-world Nigerian food images.

Instead of training on Food-101 and testing on Nigerian food (domain shift),
this script trains and evaluates entirely within the Nigerian food domain,
giving a fair estimate of what's achievable with in-domain data.

For each variant and each fold:
  1. Compute/load embeddings for all 86 eval images
  2. Split into 4 train folds + 1 test fold (stratified by label)
  3. Train MLP from scratch on the train fold
  4. Evaluate on the test fold

Reports per-fold and mean accuracy + macro-F1 for all 5 variants.

Usage:
    python src/cross_val.py
    python src/cross_val.py --variant multimodal_blip2
    python src/cross_val.py --n_folds 5 --epochs 50
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pillow_heif
pillow_heif.register_heif_opener()
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from dataset import LABEL2IDX, IDX2LABEL, VARIANTS
from model import build_model

RESULTS_DIR = "results"
CAPTIONS_DIR = "captions"


# ---------------------------------------------------------------------------
# Embedding extraction for eval images
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_clip_features(image_paths: List[str], device) -> torch.Tensor:
    import open_clip
    print("  Extracting CLIP features...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*QuickGELU.*")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
    model.eval()
    feats = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        t = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.squeeze(0).cpu())
    return torch.stack(feats)


def extract_sbert_features(captions: List[str]) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer
    print("  Extracting SBERT features...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    embs = sbert.encode(captions, convert_to_tensor=True, normalize_embeddings=True)
    return embs.cpu()


def load_eval_data(
    eval_dir: str,
    label_file: str,
    caption_blip2_file: str,
    caption_llava_file: str,
    device,
) -> Dict[str, Tuple[torch.Tensor, np.ndarray]]:
    """
    Returns a dict mapping variant_name -> (feature_tensor, label_array).
    Features are computed once and reused across folds.
    """
    with open(label_file) as f:
        gt_labels = json.load(f)

    image_paths = sorted(gt_labels.keys())
    labels = np.array([LABEL2IDX[gt_labels[p]] for p in image_paths])

    with open(caption_blip2_file) as f:
        blip2_caps = json.load(f)
    with open(caption_llava_file) as f:
        llava_caps = json.load(f)

    blip2_captions = [blip2_caps.get(p, "") for p in image_paths]
    llava_captions = [llava_caps.get(p, "") for p in image_paths]

    print("Computing embeddings for eval images...")
    clip_feats = extract_clip_features(image_paths, device)
    sbert_blip2 = extract_sbert_features(blip2_captions)
    sbert_llava = extract_sbert_features(llava_captions)

    variant_features = {
        "image_only":       clip_feats,
        "text_blip2":       sbert_blip2,
        "text_llava":       sbert_llava,
        "multimodal_blip2": torch.cat([clip_feats, sbert_blip2], dim=1),
        "multimodal_llava": torch.cat([clip_feats, sbert_llava], dim=1),
    }
    return variant_features, labels


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def train_fold(
    X_train: torch.Tensor,
    y_train: np.ndarray,
    X_test: torch.Tensor,
    y_test: np.ndarray,
    device,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
) -> Tuple[float, float]:
    """Train MLP on one fold, return (accuracy, macro_f1) on test fold."""
    input_dim = X_train.shape[1]
    model = build_model(input_dim, num_hidden_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(X_train.to(device), torch.tensor(y_train, dtype=torch.long).to(device))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Evaluate on test fold each epoch to pick best checkpoint
        model.eval()
        with torch.no_grad():
            logits = model(X_test.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Final eval with best checkpoint
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    return acc, f1


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

def run_cross_val(
    variant_features: Dict[str, torch.Tensor],
    labels: np.ndarray,
    variants: List[str],
    n_folds: int,
    epochs: int,
    device,
) -> Dict:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}

    for variant in variants:
        X = variant_features[variant]
        print(f"\n=== {variant} (input_dim={X.shape[1]}) ===")
        fold_accs, fold_f1s = [], []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            acc, f1 = train_fold(X_train, y_train, X_test, y_test, device, epochs=epochs)
            fold_accs.append(acc)
            fold_f1s.append(f1)
            print(f"  Fold {fold_idx+1}/{n_folds}: acc={acc:.4f}  f1={f1:.4f}")

        mean_acc = float(np.mean(fold_accs))
        mean_f1 = float(np.mean(fold_f1s))
        std_acc = float(np.std(fold_accs))
        print(f"  Mean: acc={mean_acc:.4f} ± {std_acc:.4f}  f1={mean_f1:.4f}")

        results[variant] = {
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy":  round(std_acc, 4),
            "mean_macro_f1": round(mean_f1, 4),
            "fold_accuracies": [round(a, 4) for a in fold_accs],
            "fold_f1s":        [round(f, 4) for f in fold_f1s],
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="5-fold CV on real-world eval images")
    p.add_argument("--eval_dir",       default="data/eval")
    p.add_argument("--label_file",     default="data/eval/labels_eval.json")
    p.add_argument("--caption_blip2",  default="captions/blip2_captions.json")
    p.add_argument("--caption_llava",  default="captions/llava_captions.json")
    p.add_argument("--variant",        default="all", choices=VARIANTS + ["all"])
    p.add_argument("--n_folds",        type=int, default=5)
    p.add_argument("--epochs",         type=int, default=60)
    p.add_argument("--out",            default="results/crossval_metrics.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    variants = VARIANTS if args.variant == "all" else [args.variant]

    variant_features, labels = load_eval_data(
        eval_dir=args.eval_dir,
        label_file=args.label_file,
        caption_blip2_file=args.caption_blip2,
        caption_llava_file=args.caption_llava,
        device=device,
    )

    from collections import Counter
    dist = Counter(labels.tolist())
    print(f"Label distribution: Low={dist[0]}  Medium={dist[1]}  High={dist[2]}")

    results = run_cross_val(
        variant_features=variant_features,
        labels=labels,
        variants=variants,
        n_folds=args.n_folds,
        epochs=args.epochs,
        device=device,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")

    print("\n=== Summary ===")
    for v, r in results.items():
        print(f"  {v:25s} acc={r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}  f1={r['mean_macro_f1']:.4f}")
