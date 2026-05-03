"""
caption.py

Generates natural-language captions for all food images using:
  - BLIP-2  (Salesforce/blip2-opt-2.7b)
  - LLaVA   (llava-hf/llava-1.5-7b-hf)

Captions are saved to:
  captions/blip2_captions.json
  captions/llava_captions.json

Each JSON maps  image_path (relative to data/food101/)  →  caption string.

Usage:
    python src/caption.py --model blip2 --image_dir data/food101 --out_dir captions
    python src/caption.py --model llava  --image_dir data/food101 --out_dir captions
    python src/caption.py --model both   --image_dir data/food101 --out_dir captions
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
import pillow_heif
pillow_heif.register_heif_opener()
from PIL import Image
from tqdm import tqdm


# 
# Helpers
# 

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_images(image_dir: str) -> List[Path]:
    """Return sorted list of all .jpg/.jpeg/.png paths under image_dir."""
    root = Path(image_dir)
    images = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return images


def load_existing(out_path: str) -> dict:
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return {}


def save_captions(captions: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(captions, f, indent=2)


# 
# BLIP-2
# 

def load_blip2():
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    device = get_device()
    print(f"Loading BLIP-2 on {device} ...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return processor, model, device


def caption_blip2(image_paths: List[Path], out_path: str, batch_size: int = 4):
    processor, model, device = load_blip2()
    captions = load_existing(out_path)
    todo = [p for p in image_paths if str(p) not in captions]
    print(f"BLIP-2: {len(todo)} images to caption ({len(captions)} already done)")

    # Process one image at a time; pass a text prompt so input_ids/image mask are
    # populated correctly in transformers 4.40+ (avoids special_image_mask shape bug)
    prompt = "Question: Describe the food in this image. Answer:"
    for i, path in enumerate(tqdm(todo, desc="BLIP-2")):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=80)
        text = processor.decode(generated_ids[0], skip_special_tokens=True)
        # Strip the prompt from the output
        if prompt in text:
            text = text[text.index(prompt) + len(prompt):].strip()
        captions[str(path)] = text.strip()
        # Save incrementally every 100 images
        if (i + 1) % 100 == 0:
            save_captions(captions, out_path)
    save_captions(captions, out_path)

    print(f"BLIP-2 captions saved to {out_path}")
    return captions


# 
# LLaVA
# 

def load_llava():
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    device = get_device()
    print(f"Loading LLaVA on {device} ...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return processor, model, device


LLAVA_PROMPT = (
    "USER: <image>\n"
    "Describe this food dish in one sentence, focusing on the ingredients, "
    "cooking method, and approximate portion size.\nASSISTANT:"
)


def caption_llava(image_paths: List[Path], out_path: str, batch_size: int = 1):
    """LLaVA inference (batch_size=1 recommended for 7B on MPS/CPU)."""
    processor, model, device = load_llava()
    captions = load_existing(out_path)
    todo = [p for p in image_paths if str(p) not in captions]
    print(f"LLaVA: {len(todo)} images to caption ({len(captions)} already done)")

    for path in tqdm(todo, desc="LLaVA"):
        image = Image.open(path).convert("RGB")
        inputs = processor(
            text=LLAVA_PROMPT, images=image, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100)
        text = processor.decode(output[0], skip_special_tokens=True)
        # Strip the prompt prefix from the output
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()
        captions[str(path)] = text
        save_captions(captions, out_path)

    print(f"LLaVA captions saved to {out_path}")
    return captions


# 
# CLI
# 

def parse_args():
    p = argparse.ArgumentParser(description="Generate food captions via BLIP-2 or LLaVA")
    p.add_argument("--model", choices=["blip2", "llava", "both"], default="both")
    p.add_argument("--image_dir", default="data/food101",
                   help="Root directory of food images")
    p.add_argument("--out_dir", default="captions",
                   help="Directory to write JSON caption files")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size for BLIP-2 (LLaVA always uses 1)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of images (for quick testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    images = collect_images(args.image_dir)
    if args.limit:
        images = images[: args.limit]
    print(f"Found {len(images)} images in {args.image_dir}")

    blip2_out = os.path.join(args.out_dir, "blip2_captions.json")
    llava_out = os.path.join(args.out_dir, "llava_captions.json")

    if args.model in ("blip2", "both"):
        caption_blip2(images, blip2_out, batch_size=args.batch_size)

    if args.model in ("llava", "both"):
        caption_llava(images, llava_out)
