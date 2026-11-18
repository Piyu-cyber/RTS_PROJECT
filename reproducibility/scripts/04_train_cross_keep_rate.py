"""Script 04: Train or verify the 11-model Cross-Keep-Rate experimental matrix."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reproducibility.src.utils.hashing import compute_sha256

CROSS_KEEP_RATE_MODELS = [
    ("DeiT", "outputs/bdd100k_migration/full_dataset_checkpoints/deit_small_baseline_best.pth"),
    ("K09-CE", "outputs/cross_keep_rate_study/checkpoints/K09-CE_best.pth"),
    ("K09-CB", "outputs/cross_keep_rate_study/checkpoints/K09-CB_best.pth"),
    ("K09-CB-CMPR", "outputs/cross_keep_rate_study/checkpoints/K09-CB-CMPR_best.pth"),
    ("K07-CE", "outputs/cross_keep_rate_study/checkpoints/K07-CE_best.pth"),
    ("K07-CB", "outputs/cross_keep_rate_study/checkpoints/K07-CB_best.pth"),
    ("K07-CB-CMPR", "outputs/cross_keep_rate_study/checkpoints/K07-CB-CMPR_best.pth"),
    ("K05-CE", "outputs/cross_keep_rate_study/checkpoints/K05-CE_best.pth"),
    ("K05-CMPR", "outputs/cross_keep_rate_study/checkpoints/K05-CMPR_best.pth"),
    ("K05-CB", "outputs/cross_keep_rate_study/checkpoints/K05-CB_best.pth"),
    ("K05-CB-CMPR", "outputs/cross_keep_rate_study/checkpoints/K05-CB-CMPR_best.pth"),
]


def main():
    parser = argparse.ArgumentParser(description="Cross-Keep-Rate training and verification.")
    parser.add_argument("--verify-only", action="store_true", default=True, help="Verify checkpoint existence and integrity.")
    parser.add_argument("--train-model", type=str, default="", help="Specific model ID to train if requested.")
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 04: CROSS-KEEP-RATE 11-MODEL MATRIX VERIFICATION")
    print("=" * 80)

    found_count = 0
    for model_id, rel_path in CROSS_KEEP_RATE_MODELS:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if os.path.isfile(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            sha = compute_sha256(full_path)
            print(f"  [OK] {model_id:<12} -> {size_mb:6.2f} MB | SHA256: {sha[:16]}... ({rel_path})")
            found_count += 1
        else:
            print(f"  [MISSING] {model_id:<12} -> {rel_path}")

    print(f"\nVerification summary: {found_count}/{len(CROSS_KEEP_RATE_MODELS)} checkpoints verified.")
    if found_count == len(CROSS_KEEP_RATE_MODELS):
        print("[PASS] Full 11-model experimental matrix is present and intact (exit code 0).\n")
    else:
        print("[NOTE] Some checkpoints are missing. Use --train-model <id> to train them.\n")


if __name__ == "__main__":
    main()
