"""
train.py

Trains all 5 model variants (or a specified subset) and saves model checkpoints.

Checkpoints are saved to:
    results/<variant_name>/best_model.pt

Each checkpoint contains:
    {
        "model_state_dict": ...,
        "val_accuracy":     float,
        "val_f1":           float,
        "input_dim":        int,
        "num_hidden_layers": int,
        "variant":          str,
        "label2idx":        dict,
    }

Usage:
    # Train all 5 variants (default)
    python src/train.py

    # Train a specific variant
    python src/train.py --variant multimodal_llava

    # Quick smoke-test with fewer epochs
    python src/train.py --epochs 5 --variant image_only
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# Make src/ importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dataset import FoodCalDataset, VARIANTS, LABEL2IDX, load_embedding_store
from model import build_model


# 
# Defaults
# 
EMBED_DIR = "embeddings"
RESULTS_DIR = "results"
DEFAULT_EPOCHS = 30
DEFAULT_BATCH = 256
DEFAULT_LR = 1e-3
DEFAULT_HIDDEN = 2   # number of hidden layers
VAL_SPLIT = 0.15
SEED = 42


# 
# Helpers
# 

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_stores(embed_dir: str) -> dict:
    stores = {}
    clip_path = os.path.join(embed_dir, "clip_embeddings.pt")
    blip2_path = os.path.join(embed_dir, "sbert_blip2.pt")
    llava_path = os.path.join(embed_dir, "sbert_llava.pt")

    if os.path.exists(clip_path):
        print("Loading CLIP embeddings ...")
        stores["clip"] = load_embedding_store(clip_path)
    if os.path.exists(blip2_path):
        print("Loading SBERT-BLIP2 embeddings ...")
        stores["sbert_blip2"] = load_embedding_store(blip2_path)
    if os.path.exists(llava_path):
        print("Loading SBERT-LLaVA embeddings ...")
        stores["sbert_llava"] = load_embedding_store(llava_path)
    return stores


def make_dataset(variant: str, stores: dict) -> FoodCalDataset:
    return FoodCalDataset(
        variant=variant,
        clip_store=stores.get("clip"),
        sbert_blip2_store=stores.get("sbert_blip2"),
        sbert_llava_store=stores.get("sbert_llava"),
    )


def stratified_split(dataset: FoodCalDataset, val_split: float, seed: int):
    labels = dataset.labels.numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(sss.split(labels, labels))
    return train_idx.tolist(), val_idx.tolist()


def accuracy(preds, targets):
    return (preds == targets).float().mean().item()


# 
# Training loop
# 

def train_variant(
    variant: str,
    stores: dict,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    num_hidden_layers: int = DEFAULT_HIDDEN,
    results_dir: str = RESULTS_DIR,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Training variant: {variant}")
    print(f"{'='*60}")

    device = get_device()
    print(f"Device: {device}")

    # Build full dataset, then stratified split
    full_ds = make_dataset(variant, stores)
    train_idx, val_idx = stratified_split(full_ds, val_split, seed)

    # Rebuild with split indices
    train_ds = FoodCalDataset(
        variant=variant,
        clip_store=stores.get("clip"),
        sbert_blip2_store=stores.get("sbert_blip2"),
        sbert_llava_store=stores.get("sbert_llava"),
        indices=train_idx,
    )
    val_ds = FoodCalDataset(
        variant=variant,
        clip_store=stores.get("clip"),
        sbert_blip2_store=stores.get("sbert_blip2"),
        sbert_llava_store=stores.get("sbert_llava"),
        indices=val_idx,
    )

    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Input dim: {train_ds.input_dim}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(train_ds.input_dim, num_hidden_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_ckpt = None
    history = []

    out_dir = os.path.join(results_dir, variant)
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)
        scheduler.step()

        # --- Validate ---
        model.eval()
        all_preds, all_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(y)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(y.cpu().tolist())

        val_loss /= len(val_ds)
        val_acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
        })

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = {
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "input_dim": train_ds.input_dim,
                "num_hidden_layers": num_hidden_layers,
                "variant": variant,
                "label2idx": LABEL2IDX,
                "epoch": epoch,
            }

    # Save best checkpoint
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    torch.save(best_ckpt, ckpt_path)
    print(f"Best val_acc={best_val_acc:.4f} → saved to {ckpt_path}")

    # Save training history
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return best_ckpt


# 
# CLI
# 

def parse_args():
    p = argparse.ArgumentParser(description="Train FoodCal model variants")
    p.add_argument("--variant", choices=VARIANTS + ["all"], default="all",
                   help="Which variant to train (default: all)")
    p.add_argument("--embed_dir", default=EMBED_DIR)
    p.add_argument("--results_dir", default=RESULTS_DIR)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--hidden_layers", type=int, default=DEFAULT_HIDDEN, choices=[1, 2])
    p.add_argument("--val_split", type=float, default=VAL_SPLIT)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stores = load_stores(args.embed_dir)

    variants_to_run = VARIANTS if args.variant == "all" else [args.variant]
    # Only run variants whose required stores are present
    runnable = []
    for v in variants_to_run:
        needs_clip = v in ("image_only", "multimodal_blip2", "multimodal_llava")
        needs_blip2 = v in ("text_blip2", "multimodal_blip2")
        needs_llava = v in ("text_llava", "multimodal_llava")
        missing = []
        if needs_clip and "clip" not in stores:
            missing.append("clip_embeddings.pt")
        if needs_blip2 and "sbert_blip2" not in stores:
            missing.append("sbert_blip2.pt")
        if needs_llava and "sbert_llava" not in stores:
            missing.append("sbert_llava.pt")
        if missing:
            print(f"Skipping {v}: missing embeddings {missing}")
        else:
            runnable.append(v)

    summary = {}
    for v in runnable:
        ckpt = train_variant(
            variant=v,
            stores=stores,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_hidden_layers=args.hidden_layers,
            results_dir=args.results_dir,
            val_split=args.val_split,
            seed=args.seed,
        )
        summary[v] = {"val_accuracy": ckpt["val_accuracy"], "val_f1": ckpt["val_f1"]}

    print("\n" + "="*60)
    print("Training summary:")
    for v, m in summary.items():
        print(f"  {v:<25} acc={m['val_accuracy']:.4f}  f1={m['val_f1']:.4f}")

    # Identify best variant by val accuracy
    if summary:
        best_v = max(summary, key=lambda v: summary[v]["val_accuracy"])
        print(f"\nBest variant: {best_v} (val_acc={summary[best_v]['val_accuracy']:.4f})")
        with open(os.path.join(args.results_dir, "best_variant.json"), "w") as f:
            json.dump({"best_variant": best_v, "metrics": summary}, f, indent=2)
