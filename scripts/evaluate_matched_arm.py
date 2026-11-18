"""Evaluates a trained matched model checkpoint on the official BDD100K held-out test split (19,902 images)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from src.phase23b.bdd_dataset import (
    BDD_CLASSES,
    NUM_BDD_CLASSES,
    load_bdd_split_samples,
)
from src.phase23b.models import build_model
from scripts.run_cmpr_matched_experiments import evaluate_split, compute_sha256

MATCHED_DIR = os.path.join(PROJECT_ROOT, "outputs", "cmpr_matched_training")
CHECKPOINTS_DIR = os.path.join(MATCHED_DIR, "checkpoints")
CM_DIR = os.path.join(MATCHED_DIR, "confusion_matrices")
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "bdd100k", "annotations")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "bdd100k", "images")


def evaluate_checkpoint(run_name: str, arch_config: str, device: str = "cuda") -> None:
    best_ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{run_name}_best.pth")
    if not os.path.isfile(best_ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {best_ckpt_path}")

    print(f"[*] Loading model {arch_config} from {best_ckpt_path}...", flush=True)
    dev = torch.device(device)
    model = build_model(arch_config, num_classes=NUM_BDD_CLASSES, pretrained=False)
    ckpt = torch.load(best_ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)
    model.eval()

    test_samples = load_bdd_split_samples(os.path.join(ANNOTATIONS_DIR, "test_full.json"))
    test_dir = os.path.join(IMAGES_DIR, "100k", "test")

    print(f"[*] Evaluating on {len(test_samples):,} test images...", flush=True)
    t0 = time.time()
    test_res = evaluate_split(model, test_samples, test_dir, dev, batch_size=64)
    elapsed = round(time.time() - t0, 1)

    print(f"[+] Evaluation finished in {elapsed}s!", flush=True)
    print(f"    Top-1 Accuracy: {test_res['top1_accuracy']}%", flush=True)
    print(f"    Macro-F1:       {test_res['macro_f1']}%", flush=True)
    print(f"    Weighted-F1:    {test_res['weighted_f1']}%", flush=True)

    for pc in test_res["per_class"]:
        print(f"    Class {pc['class_id']} ({pc['class_name']:12s}): Recall={pc['recall']:5.2f}%, Precision={pc['precision']:5.2f}%, F1={pc['f1']:5.2f}%", flush=True)

    cm_path = os.path.join(CM_DIR, f"{run_name}_confusion_matrix.json")
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_name": run_name,
            "classes": BDD_CLASSES,
            "confusion_matrix": test_res["confusion_matrix"],
            "per_class": test_res["per_class"],
            "top1_accuracy": test_res["top1_accuracy"],
            "macro_f1": test_res["macro_f1"],
            "weighted_f1": test_res["weighted_f1"],
            "test_predictions": test_res["predictions"],
            "test_targets": test_res["targets"],
            "test_logits": test_res["logits"],
        }, f, indent=2)
    print(f"[+] Saved confusion matrix and predictions to: {cm_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True, help="e.g. M1_seed42")
    parser.add_argument("--arch", type=str, required=True, help="e.g. evit_small_k05_fuse")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluate_checkpoint(args.run_name, args.arch, args.device)
