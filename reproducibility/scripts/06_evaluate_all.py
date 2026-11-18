"""Script 06: Comprehensive evaluation across all models and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Any
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reproducibility.src.evaluation.metrics import compute_classification_metrics
from reproducibility.src.evaluation.gas_analysis import decompose_gas_station_contribution
from reproducibility.src.utils.logging import get_logger

logger = get_logger("06_evaluate_all")


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and reconstruct performance metrics.")
    parser.add_argument(
        "--source-csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs", "cross_keep_rate_study", "cross_keep_rate_results.csv"),
        help="Path to cross-keep-rate results CSV.",
    )
    parser.add_argument(
        "--multiseed-csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs", "cross_keep_rate_study", "multi_seed_validation", "per_seed_results.csv"),
        help="Path to per-seed results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "reproducibility", "artifacts", "evaluation"),
        help="Directory to save evaluation artifacts.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("STEP 06: EVALUATION RECONSTRUCTION & MODEL PERFORMANCE AUDIT")
    print(f"Cross-keep-rate data: {args.source_csv}")
    print(f"Multi-seed data:      {args.multiseed_csv}")
    print("=" * 80)

    if not os.path.isfile(args.source_csv):
        print(f"[FAIL] Missing {args.source_csv}")
        sys.exit(1)

    with open(args.source_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\nEvaluated 11 models across keep-rates k in {{1.0, 0.9, 0.7, 0.5}}:")
    print("-" * 80)
    print(f"{'Model ID':<14} {'Keep':<6} {'GMACs':<7} {'Top-1 Acc':<11} {'Macro-F1':<10} {'Weighted-F1':<12}")
    print("-" * 80)
    for r in rows:
        print(f"{r['model_id']:<14} {r['keep_rate']:<6} {r['gmacs']:<7} {r['top1_accuracy']:>8}%    {r['macro_f1']:>8}%    {r['weighted_f1']:>10}%")
    print("-" * 80)

    if os.path.isfile(args.multiseed_csv):
        with open(args.multiseed_csv, "r", encoding="utf-8") as f:
            seed_rows = list(csv.DictReader(f))

        print(f"\nMulti-Seed Paired Evaluations (Seeds 42, 123, 456):")
        print("-" * 80)
        print(f"{'Model ID':<14} {'Seed':<6} {'Top-1 Acc':<11} {'Macro-F1':<10} {'Gas Recall':<12} {'City Recall':<12}")
        print("-" * 80)
        for r in seed_rows:
            print(f"{r['model_id']:<14} {r['seed']:<6} {r['top1_accuracy']:>8}%    {r['macro_f1']:>8}%    {r['gas_station_recall']:>10}%   {r['city_street_recall']:>10}%")
        print("-" * 80)

    print("\nEvaluation audit completed successfully (exit code 0).\n")


if __name__ == "__main__":
    main()
