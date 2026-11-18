"""Evaluate historical matched DeiT, K05-CE, and K05-CMPR for unified test set records."""

import json
import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_cross_keep_rate_experiments import (
    ANNOTATIONS_DIR,
    CM_DIR,
    IMAGES_DIR,
    evaluate_split,
)
from src.phase23b.bdd_dataset import BDD_CLASSES, load_bdd_split_samples
from src.phase23b.models import build_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_samples = load_bdd_split_samples(os.path.join(ANNOTATIONS_DIR, "test_full.json"))
    test_dir = os.path.join(IMAGES_DIR, "100k", "test")

    models_to_eval = [
        ("DeiT", "deit_small_baseline", "outputs/cmpr_matched_training/checkpoints/M0_seed42_best.pth"),
        ("K05-CE", "evit_small_k05_fuse", "outputs/cmpr_matched_training/checkpoints/M1_seed42_best.pth"),
        ("K05-CMPR", "evit_small_k05_fuse", "outputs/cmpr_matched_training/checkpoints/M6_seed42_best.pth"),
    ]

    for model_id, arch, ckpt_rel in models_to_eval:
        run_name = f"{model_id}_seed42"
        out_json = os.path.join(CM_DIR, f"{run_name}_confusion_matrix.json")
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_rel)
        if os.path.isfile(out_json):
            print(f"Skipping {run_name}, already exists at {out_json}")
            continue
        print(f"Evaluating {run_name} from {ckpt_path}...")
        m = build_model(arch, num_classes=6, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(device)
        m.eval()
        res = evaluate_split(m, test_samples, test_dir, device, batch_size=64)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "model_id": model_id,
                "run_name": run_name,
                "classes": BDD_CLASSES,
                "confusion_matrix": res["confusion_matrix"],
                "per_class": res["per_class"],
                "top1_accuracy": res["top1_accuracy"],
                "macro_f1": res["macro_f1"],
                "weighted_f1": res["weighted_f1"],
                "test_predictions": res["predictions"],
                "test_targets": res["targets"],
                "test_logits": res["logits"],
            }, f, indent=2)
        print(f"[DONE] {run_name}: Top-1={res['top1_accuracy']}%, Macro-F1={res['macro_f1']}%")


if __name__ == "__main__":
    main()
