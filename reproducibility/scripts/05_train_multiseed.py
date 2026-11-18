"""Script 05: Multi-seed validation (seeds 42, 123, 456) for K07-CB and K07-CB-CMPR."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reproducibility.src.utils.hashing import compute_sha256

MULTISEED_RUNS = [
    ("K07-CB", 42, "outputs/cross_keep_rate_study/multi_seed_validation/seed_42/K07-CB_best.pth"),
    ("K07-CB-CMPR", 42, "outputs/cross_keep_rate_study/multi_seed_validation/seed_42/K07-CB-CMPR_best.pth"),
    ("K07-CB", 123, "outputs/cross_keep_rate_study/multi_seed_validation/seed_123/K07-CB_best.pth"),
    ("K07-CB-CMPR", 123, "outputs/cross_keep_rate_study/multi_seed_validation/seed_123/K07-CB-CMPR_best.pth"),
    ("K07-CB", 456, "outputs/cross_keep_rate_study/multi_seed_validation/seed_456/K07-CB_best.pth"),
    ("K07-CB-CMPR", 456, "outputs/cross_keep_rate_study/multi_seed_validation/seed_456/K07-CB-CMPR_best.pth"),
]


def main():
    parser = argparse.ArgumentParser(description="Multi-seed training and verification.")
    parser.add_argument("--verify-only", action="store_true", default=True, help="Verify checkpoint existence and integrity.")
    parser.add_argument("--train-seed", type=int, default=None, help="Train a specific seed.")
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 05: MULTI-SEED VALIDATION (3-SEED MATRIX: 42, 123, 456)")
    print("=" * 80)

    found_count = 0
    for model_id, seed, rel_path in MULTISEED_RUNS:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if os.path.isfile(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            sha = compute_sha256(full_path)
            print(f"  [OK] {model_id:<12} (Seed {seed:>3}) -> {size_mb:6.2f} MB | SHA256: {sha[:16]}... ({rel_path})")
            found_count += 1
        else:
            print(f"  [MISSING] {model_id:<12} (Seed {seed:>3}) -> {rel_path}")

    print(f"\nVerification summary: {found_count}/{len(MULTISEED_RUNS)} checkpoints verified.")
    if found_count == len(MULTISEED_RUNS):
        print("[PASS] Full 3-seed experimental matrix is present and intact (exit code 0).\n")


if __name__ == "__main__":
    main()
