"""
dataset.py

PyTorch Dataset that loads precomputed embeddings for any of the 5 model variants.

Variant names (used as keys throughout the codebase):
    "image_only"        — CLIP embedding only
    "text_blip2"        — Sentence-BERT (BLIP-2 captions) only
    "text_llava"        — Sentence-BERT (LLaVA captions) only
    "multimodal_blip2"  — CLIP + Sentence-BERT (BLIP-2) concatenated
    "multimodal_llava"  — CLIP + Sentence-BERT (LLaVA) concatenated
"""

import torch
from torch.utils.data import Dataset

LABEL2IDX = {"Low": 0, "Medium": 1, "High": 2}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

VARIANTS = [
    "image_only",
    "text_blip2",
    "text_llava",
    "multimodal_blip2",
    "multimodal_llava",
]


def load_embedding_store(path: str) -> dict:
    """Load a .pt embedding store produced by embed.py."""
    store = torch.load(path, map_location="cpu", weights_only=False)
    return store  # keys: "keys", "embeddings", "labels"


class FoodCalDataset(Dataset):
    """
    Builds a dataset for a given variant from precomputed embedding stores.

    Args:
        variant:      one of VARIANTS
        clip_store:   dict from load_embedding_store (clip_embeddings.pt)
        sbert_blip2:  dict from load_embedding_store (sbert_blip2.pt)   [optional]
        sbert_llava:  dict from load_embedding_store (sbert_llava.pt)   [optional]
        indices:      optional list of int to select a subset (for train/val split)
    """

    def __init__(
        self,
        variant: str,
        clip_store: dict | None = None,
        sbert_blip2_store: dict | None = None,
        sbert_llava_store: dict | None = None,
        indices: list[int] | None = None,
    ):
        assert variant in VARIANTS, f"Unknown variant: {variant}"
        self.variant = variant

        # Build key → embedding index maps
        clip_key2idx = (
            {k: i for i, k in enumerate(clip_store["keys"])} if clip_store else {}
        )
        blip2_key2idx = (
            {k: i for i, k in enumerate(sbert_blip2_store["keys"])}
            if sbert_blip2_store
            else {}
        )
        llava_key2idx = (
            {k: i for i, k in enumerate(sbert_llava_store["keys"])}
            if sbert_llava_store
            else {}
        )

        # Determine the common keys for this variant
        if variant == "image_only":
            common_keys = list(clip_store["keys"])
        elif variant == "text_blip2":
            common_keys = list(sbert_blip2_store["keys"])
        elif variant == "text_llava":
            common_keys = list(sbert_llava_store["keys"])
        elif variant == "multimodal_blip2":
            clip_set = set(clip_store["keys"])
            common_keys = [k for k in sbert_blip2_store["keys"] if k in clip_set]
        elif variant == "multimodal_llava":
            clip_set = set(clip_store["keys"])
            common_keys = [k for k in sbert_llava_store["keys"] if k in clip_set]

        # Build tensors
        feats, labels = [], []
        for key in common_keys:
            if variant == "image_only":
                feat = clip_store["embeddings"][clip_key2idx[key]]
                label = clip_store["labels"][clip_key2idx[key]]
            elif variant == "text_blip2":
                feat = sbert_blip2_store["embeddings"][blip2_key2idx[key]]
                label = sbert_blip2_store["labels"][blip2_key2idx[key]]
            elif variant == "text_llava":
                feat = sbert_llava_store["embeddings"][llava_key2idx[key]]
                label = sbert_llava_store["labels"][llava_key2idx[key]]
            elif variant == "multimodal_blip2":
                clip_feat = clip_store["embeddings"][clip_key2idx[key]]
                sbert_feat = sbert_blip2_store["embeddings"][blip2_key2idx[key]]
                feat = torch.cat([clip_feat, sbert_feat], dim=-1)
                label = clip_store["labels"][clip_key2idx[key]]
            elif variant == "multimodal_llava":
                clip_feat = clip_store["embeddings"][clip_key2idx[key]]
                sbert_feat = sbert_llava_store["embeddings"][llava_key2idx[key]]
                feat = torch.cat([clip_feat, sbert_feat], dim=-1)
                label = clip_store["labels"][clip_key2idx[key]]

            feats.append(feat)
            labels.append(LABEL2IDX[label])

        self.features = torch.stack(feats).float()
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.keys = common_keys

        if indices is not None:
            self.features = self.features[indices]
            self.labels = self.labels[indices]
            self.keys = [self.keys[i] for i in indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    @property
    def input_dim(self) -> int:
        return self.features.shape[1]
