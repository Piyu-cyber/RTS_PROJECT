"""Script 01: BDD100K dataset preparation and manifest generation."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reproducibility.src.data.prepare_bdd100k import prepare_and_verify_manifests
from reproducibility.src.utils.logging import get_logger

logger = get_logger("01_prepare_dataset")


def main():
    parser = argparse.ArgumentParser(description="Prepare BDD100K split manifests.")
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "bdd100k", "annotations"),
        help="Directory containing BDD100K annotation files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "reproducibility", "data", "manifests"),
        help="Output directory for verified manifests.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 01: PREPARING BDD100K DATASET MANIFESTS")
    print(f"Annotations source: {args.annotations_dir}")
    print(f"Manifests target:   {args.output_dir}")
    print("=" * 80)

    summary = prepare_and_verify_manifests(
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
    )

    print("\nDataset preparation summary:")
    for split, count in summary.items():
        print(f"  - {split}: {count:,} samples")
    print("\nDataset preparation complete (exit code 0).\n")


if __name__ == "__main__":
    main()
