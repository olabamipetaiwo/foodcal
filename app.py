"""
app.py

Gradio web application for FoodCal — Automated Calorie Range Estimation.

Loads the best-performing model variant (determined by ablation study) and
provides a zero-installation browser interface:
  - User uploads a food photo
  - System captions it (BLIP-2 or LLaVA depending on best variant)
  - Embeds image and/or caption
  - MLP predicts calorie range: Low / Medium / High
  - Returns prediction + per-class confidence scores

Run locally:
    python app.py

Deploy to Hugging Face Spaces:
    Push this repo to HF Spaces with a Space SDK set to "gradio".
"""

import json
import os
import sys

import gradio as gr
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dataset import LABEL2IDX, IDX2LABEL
from model import build_model

# ---------------------------------------------------------------------------
# Load best variant config
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"
CAPTIONS_DIR = "captions"

LABEL_COLORS = {
    "Low": "#27ae60",
    "Medium": "#f39c12",
    "High": "#e74c3c",
}

LABEL_DESCRIPTIONS = {
    "Low":    "Under 400 kcal — lighter meal",
    "Medium": "400–700 kcal — moderate meal",
    "High":   "Over 700 kcal — high-calorie meal",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_best_variant() -> tuple[str, dict]:
    """Load the best variant name and its checkpoint."""
    best_path = os.path.join(RESULTS_DIR, "best_variant.json")
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
        variant = data["best_variant"]
    elif os.path.exists(best_path):
        with open(best_path) as f:
            data = json.load(f)
        variant = data["best_variant"]
    else:
        # Fallback: scan for any available checkpoint
        for v in ["multimodal_llava", "multimodal_blip2", "image_only",
                  "text_llava", "text_blip2"]:
            ckpt = os.path.join(RESULTS_DIR, v, "best_model.pt")
            if os.path.exists(ckpt):
                variant = v
                break
        else:
            raise FileNotFoundError(
                "No trained checkpoints found. Run train.py first."
            )

    ckpt_path = os.path.join(RESULTS_DIR, variant, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Loaded variant: {variant}  (val_acc={ckpt.get('val_accuracy', '?'):.4f})")
    return variant, ckpt


# ---------------------------------------------------------------------------
# Per-variant inference helpers (lazy-loaded on first call)
# ---------------------------------------------------------------------------

_clip_model = None
_clip_preprocess = None
_sbert_model = None
_blip2_processor = None
_blip2_model = None
_llava_processor = None
_llava_model = None


def get_clip(device):
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import open_clip
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        _clip_model.eval()
    return _clip_model, _clip_preprocess


def get_sbert():
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert_model


def get_blip2(device):
    global _blip2_processor, _blip2_model
    if _blip2_model is None:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        _blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        _blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if str(device) != "cpu" else torch.float32,
        ).to(device)
        _blip2_model.eval()
    return _blip2_processor, _blip2_model


def get_llava(device):
    global _llava_processor, _llava_model
    if _llava_model is None:
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        model_id = "llava-hf/llava-1.5-7b-hf"
        _llava_processor = AutoProcessor.from_pretrained(model_id)
        _llava_model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if str(device) != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        _llava_model.eval()
    return _llava_processor, _llava_model


def encode_clip(image: Image.Image, device) -> torch.Tensor:
    model, preprocess = get_clip(device)
    t = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu()


def encode_sbert(caption: str) -> torch.Tensor:
    sbert = get_sbert()
    return sbert.encode(caption, convert_to_tensor=True, normalize_embeddings=True).cpu()


def generate_blip2_caption(image: Image.Image, device) -> str:
    proc, model = get_blip2(device)
    inputs = proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=80)
    return proc.decode(ids[0], skip_special_tokens=True).strip()


def generate_llava_caption(image: Image.Image, device) -> str:
    proc, model = get_llava(device)
    prompt = (
        "USER: <image>\nDescribe this food dish in one sentence, focusing on the "
        "ingredients, cooking method, and approximate portion size.\nASSISTANT:"
    )
    inputs = proc(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100)
    text = proc.decode(out[0], skip_special_tokens=True)
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    return text


def build_input_feature(image: Image.Image, variant: str, device) -> tuple[torch.Tensor, str]:
    """Returns (feature_tensor, caption_text)."""
    caption = ""
    if variant == "image_only":
        feat = encode_clip(image, device)
    elif variant == "text_blip2":
        caption = generate_blip2_caption(image, device)
        feat = encode_sbert(caption)
    elif variant == "text_llava":
        caption = generate_llava_caption(image, device)
        feat = encode_sbert(caption)
    elif variant == "multimodal_blip2":
        caption = generate_blip2_caption(image, device)
        clip_feat = encode_clip(image, device)
        sbert_feat = encode_sbert(caption)
        feat = torch.cat([clip_feat, sbert_feat])
    elif variant == "multimodal_llava":
        caption = generate_llava_caption(image, device)
        clip_feat = encode_clip(image, device)
        sbert_feat = encode_sbert(caption)
        feat = torch.cat([clip_feat, sbert_feat])
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return feat, caption


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

device = get_device()
print(f"Using device: {device}")

try:
    VARIANT, CKPT = load_best_variant()
    classifier = build_model(CKPT["input_dim"], CKPT["num_hidden_layers"])
    classifier.load_state_dict(CKPT["model_state_dict"])
    classifier.to(device).eval()
    MODEL_LOADED = True
    print(f"Classifier ready. Input dim: {CKPT['input_dim']}")
except Exception as e:
    MODEL_LOADED = False
    VARIANT = "N/A"
    print(f"WARNING: Could not load model — {e}")


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

def predict(image: Image.Image) -> tuple[str, dict, str]:
    """
    Returns:
        label:       "Low" / "Medium" / "High"
        confidences: {"Low": float, "Medium": float, "High": float}
        caption:     generated caption (empty for image-only variant)
    """
    if not MODEL_LOADED:
        return "Model not loaded", {}, "Please run train.py first."

    feat, caption = build_input_feature(image, VARIANT, device)
    with torch.no_grad():
        logits = classifier(feat.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    label = IDX2LABEL[probs.argmax().item()]
    confidences = {IDX2LABEL[i]: round(probs[i].item(), 4) for i in range(3)}
    return label, confidences, caption


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def gradio_predict(image):
    if image is None:
        return "No image provided", {}, ""

    pil_image = Image.fromarray(image).convert("RGB")
    label, confidences, caption = predict(pil_image)
    description = LABEL_DESCRIPTIONS[label]

    # Format confidence as labelled dict for Gradio Label component
    conf_display = {f"{k} ({LABEL_DESCRIPTIONS[k]})": v for k, v in confidences.items()}

    caption_out = caption if caption else "(image-only variant — no caption generated)"
    result_text = f"**{label}** — {description}"
    return result_text, conf_display, caption_out


with gr.Blocks(title="FoodCal — Calorie Range Estimator") as demo:
    gr.Markdown(
        """
        # FoodCal — Automated Calorie Range Estimator
        Upload a food photo and get an instant calorie range estimate.
        No nutritional knowledge required.

        **Classes:** Low (< 400 kcal) | Medium (400–700 kcal) | High (> 700 kcal)
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload a food photo", type="numpy")
            submit_btn = gr.Button("Estimate Calories", variant="primary")

        with gr.Column():
            prediction_out = gr.Markdown(label="Prediction")
            confidence_out = gr.Label(label="Confidence Scores", num_top_classes=3)
            caption_out = gr.Textbox(label="Generated Caption", interactive=False)

    gr.Markdown(
        f"""
        ---
        **Active model variant:** `{VARIANT}`
        Best variant selected from local ablation study.
        *This is a proof-of-concept tool. Not intended for clinical use.*
        """
    )

    submit_btn.click(
        fn=gradio_predict,
        inputs=[image_input],
        outputs=[prediction_out, confidence_out, caption_out],
    )

    gr.Examples(
        examples=[],
        inputs=image_input,
    )


if __name__ == "__main__":
    demo.launch(share=False)
