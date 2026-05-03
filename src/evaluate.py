"""
evaluate.py

Evaluates a trained model checkpoint on a held-out evaluation set and returns:
  - Classification accuracy
  - Macro-averaged F1-score
  - Confusion matrix
  - Per-class predictions (for McNemar's test between variant pairs)

Usage:
    python src/evaluate.py \
        --variant multimodal_llava \
        --ckpt_dir results/multimodal_llava \
        --eval_dir data/eval \
        --caption_dir captions \
        --label_file data/labels.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import pillow_heif
pillow_heif.register_heif_opener()
from typing import List, Optional
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, os.path.dirname(__file__))
from dataset import LABEL2IDX, IDX2LABEL
from model import build_model


# 
# Load checkpoint
# 

def load_checkpoint(ckpt_dir: str) -> dict:
    path = os.path.join(ckpt_dir, "best_model.pt")
    assert os.path.exists(path), f"Checkpoint not found: {path}"
    return torch.load(path, map_location="cpu", weights_only=False)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# 
# Feature extraction at inference time for eval set
# 

def extract_clip_feature(image: Image.Image, device) -> torch.Tensor:
    import open_clip
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*QuickGELU.*")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
    model.eval()
    with torch.no_grad():
        t = preprocess(image).unsqueeze(0).to(device)
        feat = model.encode_image(t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu()


def extract_sbert_feature(caption: str) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    emb = sbert.encode(caption, convert_to_tensor=True, normalize_embeddings=True)
    return emb.cpu()


def caption_image_blip2(image: Image.Image, device) -> str:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if str(device) != "cpu" else torch.float32,
    ).to(device)
    model.eval()
    prompt = "Question: Describe the food in this image. Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=80)
    text = processor.decode(ids[0], skip_special_tokens=True)
    if prompt in text:
        text = text[text.index(prompt) + len(prompt):].strip()
    return text.strip()


def caption_image_llava(image: Image.Image, device) -> str:
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if str(device) != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    prompt = (
        "USER: <image>\nDescribe this food dish in one sentence, focusing on the "
        "ingredients, cooking method, and approximate portion size.\nASSISTANT:"
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100)
    text = processor.decode(out[0], skip_special_tokens=True)
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    return text


def build_feature_vector(
    image: Image.Image,
    variant: str,
    device,
    blip2_caption: Optional[str] = None,
    llava_caption: Optional[str] = None,
) -> torch.Tensor:
    """Build the input feature for a single image given a variant."""
    if variant == "image_only":
        return extract_clip_feature(image, device)
    elif variant == "text_blip2":
        cap = blip2_caption or caption_image_blip2(image, device)
        return extract_sbert_feature(cap)
    elif variant == "text_llava":
        cap = llava_caption or caption_image_llava(image, device)
        return extract_sbert_feature(cap)
    elif variant == "multimodal_blip2":
        clip_feat = extract_clip_feature(image, device)
        cap = blip2_caption or caption_image_blip2(image, device)
        sbert_feat = extract_sbert_feature(cap)
        return torch.cat([clip_feat, sbert_feat])
    elif variant == "multimodal_llava":
        clip_feat = extract_clip_feature(image, device)
        cap = llava_caption or caption_image_llava(image, device)
        sbert_feat = extract_sbert_feature(cap)
        return torch.cat([clip_feat, sbert_feat])
    else:
        raise ValueError(f"Unknown variant: {variant}")


# 
# Evaluate on the real-world eval set
# 

def evaluate_on_eval_set(
    variant: str,
    ckpt_dir: str,
    eval_dir: str,
    caption_blip2_file: Optional[str] = None,
    caption_llava_file: Optional[str] = None,
    label_file: Optional[str] = None,
) -> dict:
    device = get_device()
    ckpt = load_checkpoint(ckpt_dir)
    model = build_model(ckpt["input_dim"], ckpt["num_hidden_layers"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # Load pre-generated eval captions if available
    blip2_caps = {}
    llava_caps = {}
    if caption_blip2_file and os.path.exists(caption_blip2_file):
        with open(caption_blip2_file) as f:
            blip2_caps = json.load(f)
    if caption_llava_file and os.path.exists(caption_llava_file):
        with open(caption_llava_file) as f:
            llava_caps = json.load(f)

    # Load ground-truth labels for eval images
    # Expected: data/eval/<class_name>/<image>.jpg
    # OR a labels_eval.json file: {path: label}
    labels_eval_path = os.path.join(eval_dir, "labels_eval.json")
    if os.path.exists(labels_eval_path):
        with open(labels_eval_path) as f:
            gt_labels = json.load(f)
    elif label_file:
        with open(label_file) as f:
            cat_labels = json.load(f)
        # Derive from directory structure
        from pathlib import Path
        gt_labels = {}
        for p in Path(eval_dir).rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                cls = p.parent.name
                if cls in cat_labels:
                    gt_labels[str(p)] = cat_labels[cls]
    else:
        raise FileNotFoundError(
            f"No label file found. Provide --label_file or put labels_eval.json in {eval_dir}"
        )

    image_paths = list(gt_labels.keys())
    print(f"Evaluating {len(image_paths)} images for variant={variant}")

    all_preds, all_targets = [], []
    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            feat = build_feature_vector(
                image, variant, device,
                blip2_caption=blip2_caps.get(path),
                llava_caption=llava_caps.get(path),
            )
            with torch.no_grad():
                logit = model(feat.unsqueeze(0).to(device))
                pred = logit.argmax(dim=1).item()
            all_preds.append(pred)
            all_targets.append(LABEL2IDX[gt_labels[path]])
        except Exception as e:
            print(f"  Error on {path}: {e}")

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2]).tolist()
    report = classification_report(
        all_targets, all_preds,
        target_names=["Low", "Medium", "High"],
        zero_division=0,
    )

    result = {
        "variant": variant,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
        "predictions": all_preds,
        "targets": all_targets,
    }

    print(f"\n{variant}")
    print(f"  Accuracy: {acc:.4f}  |  Macro-F1: {macro_f1:.4f}")
    print(report)
    print(f"  Confusion matrix:\n  {np.array(cm)}")

    return result


# 
# McNemar's test
# 

def mcnemar_test(preds_a: List[int], preds_b: List[int], targets: List[int]) -> dict:
    """
    Compute McNemar's test between two classifiers on the same eval set.
    Returns p-value and test statistic.
    """
    from scipy.stats import chi2

    assert len(preds_a) == len(preds_b) == len(targets)
    # b: A correct, B wrong   c: A wrong, B correct
    b = sum(1 for pa, pb, t in zip(preds_a, preds_b, targets) if pa == t and pb != t)
    c = sum(1 for pa, pb, t in zip(preds_a, preds_b, targets) if pa != t and pb == t)

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}

    # Continuity-corrected McNemar statistic
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(statistic, df=1)
    return {"statistic": round(statistic, 4), "p_value": round(p_value, 4), "b": b, "c": c}


# 
# CLI
# 

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a FoodCal model variant on the eval set")
    p.add_argument("--variant", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--eval_dir", default="data/eval")
    p.add_argument("--caption_blip2", default="captions/blip2_captions.json")
    p.add_argument("--caption_llava", default="captions/llava_captions.json")
    p.add_argument("--label_file", default="data/labels.json")
    p.add_argument("--out_dir", default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate_on_eval_set(
        variant=args.variant,
        ckpt_dir=args.ckpt_dir,
        eval_dir=args.eval_dir,
        caption_blip2_file=args.caption_blip2,
        caption_llava_file=args.caption_llava,
        label_file=args.label_file,
    )
    out_path = os.path.join(args.out_dir, args.variant, "eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")
