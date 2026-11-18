"""Train and evaluate the 4 DeiT/EViT model configurations on BDD100K driving scenes."""

from __future__ import annotations

import os
import sys
import json
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.phase23b.models import MODEL_CONFIGS
from src.phase23b.bdd_finetuning import finetune_bdd_model

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("================================================================================")
    print(" BDD100K Fine-Tuning: 4 DeiT / EViT Configurations (6 Driving Scene Classes)")
    print(f" Device: {device}")
    print("================================================================================")

    results = {}
    total_t0 = time.time()

    for cfg_name in MODEL_CONFIGS.keys():
        print(f"\n>>> Fine-tuning {cfg_name} on BDD100K (5 epochs, batch size 16)...")
        res = finetune_bdd_model(
            config_name=cfg_name,
            dataset_root=os.path.join(PROJECT_ROOT, "data", "bdd100k"),
            device=device,
            epochs=5,
            batch_size=16,
            output_dir=CKPT_DIR,
            seed=42,
            force_retrain=True,
        )
        results[cfg_name] = res

    total_time = round(time.time() - total_t0, 2)
    print(f"\nAll 4 models successfully fine-tuned on BDD100K in {total_time}s!")

    summary_path = os.path.join(OUTPUT_DIR, "bdd_finetuning_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved fine-tuning results to {summary_path}")


if __name__ == "__main__":
    main()
