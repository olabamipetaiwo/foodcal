"""
ablation.py

Runs the full ablation study:
  1. Evaluates all trained variants on the real-world eval set
  2. Runs McNemar's test for every pair of variants
  3. Produces grouped bar charts (accuracy + macro-F1) and confusion matrices
  4. Saves results/metrics.json and identifies the best variant

Usage:
    python src/ablation.py --eval_dir data/eval --label_file data/labels.json
"""

import argparse
import itertools
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from dataset import VARIANTS, IDX2LABEL
from evaluate import evaluate_on_eval_set, mcnemar_test


RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_grouped_bars(metrics: dict, out_dir: str):
    """Grouped bar chart: accuracy and macro-F1 for all variants."""
    os.makedirs(out_dir, exist_ok=True)
    variant_labels = list(metrics.keys())
    accs = [metrics[v]["accuracy"] for v in variant_labels]
    f1s = [metrics[v]["macro_f1"] for v in variant_labels]

    x = np.arange(len(variant_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, f1s, width, label="Macro-F1", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title("Calorie Range Classification — Ablation Results")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [v.replace("_", "\n") for v in variant_labels], fontsize=9
    )
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "ablation_bar_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Bar chart saved → {path}")


def plot_confusion_matrix(cm: list, variant: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {variant}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"cm_{variant}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {path}")


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    eval_dir: str,
    caption_blip2_file: str,
    caption_llava_file: str,
    label_file: str,
    results_dir: str,
    figures_dir: str,
):
    all_results = {}

    # Evaluate each variant that has a trained checkpoint
    for variant in VARIANTS:
        ckpt_dir = os.path.join(results_dir, variant)
        ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"No checkpoint for {variant} — skipping")
            continue
        result = evaluate_on_eval_set(
            variant=variant,
            ckpt_dir=ckpt_dir,
            eval_dir=eval_dir,
            caption_blip2_file=caption_blip2_file,
            caption_llava_file=caption_llava_file,
            label_file=label_file,
        )
        # Save per-variant results
        out_path = os.path.join(ckpt_dir, "eval_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results[variant] = result

    if not all_results:
        print("No trained variants found. Run train.py first.")
        return

    # McNemar's pairwise tests
    mcnemar_results = {}
    evaluated = list(all_results.keys())
    for va, vb in itertools.combinations(evaluated, 2):
        # Align on common sample indices (same targets)
        targets_a = all_results[va]["targets"]
        targets_b = all_results[vb]["targets"]
        if targets_a != targets_b:
            print(f"  Warning: {va} and {vb} have different eval targets — skipping McNemar")
            continue
        key = f"{va} vs {vb}"
        mcnemar_results[key] = mcnemar_test(
            all_results[va]["predictions"],
            all_results[vb]["predictions"],
            targets_a,
        )

    # Print McNemar table
    print("\n" + "="*60)
    print("McNemar's Test Results (p < 0.05 → statistically significant)")
    print("="*60)
    for pair, res in mcnemar_results.items():
        sig = "**" if res["p_value"] < 0.05 else "  "
        print(f"  {sig} {pair:<45} p={res['p_value']:.4f}  χ²={res['statistic']:.4f}")

    # Identify best variant
    best_variant = max(all_results, key=lambda v: all_results[v]["accuracy"])
    best_acc = all_results[best_variant]["accuracy"]
    best_f1 = all_results[best_variant]["macro_f1"]
    print(f"\nBest variant: {best_variant}  (acc={best_acc:.4f}, macro-F1={best_f1:.4f})")

    # Save consolidated metrics
    metrics_summary = {
        v: {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
        for v, r in all_results.items()
    }
    final = {
        "metrics": metrics_summary,
        "mcnemar": mcnemar_results,
        "best_variant": best_variant,
    }
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nConsolidated metrics saved → {metrics_path}")

    # Plots
    plot_grouped_bars(metrics_summary, figures_dir)
    for variant, result in all_results.items():
        plot_confusion_matrix(result["confusion_matrix"], variant, figures_dir)

    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run full ablation study across all trained variants")
    p.add_argument("--eval_dir", default="data/eval")
    p.add_argument("--caption_blip2", default="captions/blip2_captions.json")
    p.add_argument("--caption_llava", default="captions/llava_captions.json")
    p.add_argument("--label_file", default="data/labels.json")
    p.add_argument("--results_dir", default=RESULTS_DIR)
    p.add_argument("--figures_dir", default=FIGURES_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation(
        eval_dir=args.eval_dir,
        caption_blip2_file=args.caption_blip2,
        caption_llava_file=args.caption_llava,
        label_file=args.label_file,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )
