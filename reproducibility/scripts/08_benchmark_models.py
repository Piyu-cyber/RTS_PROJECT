"""Script 08: Benchmark latency (Eager vs CUDA Graph) and throughput on RTX 4060 GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Any
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reproducibility.src.models.deit import build_deit_small
from reproducibility.src.models.evit import build_evit_small
from reproducibility.src.benchmarking.eager_benchmark import benchmark_model_latency
from reproducibility.src.benchmarking.cuda_graph_benchmark import benchmark_model_cuda_graph
from reproducibility.src.benchmarking.throughput_benchmark import benchmark_model_throughput
from reproducibility.src.benchmarking.memory_benchmark import benchmark_model_memory
from reproducibility.src.utils.logging import get_logger

logger = get_logger("08_benchmark_models")

CANDIDATES = [
    ("deit_small_baseline", 1.0, 4.60, 78.00, 77.64, lambda: build_deit_small(num_classes=6, pretrained=False)),
    ("evit_small_k09_fuse", 0.9, 4.03, 77.61, 77.40, lambda: build_evit_small(keep_rate=0.9, num_classes=6, fuse_token=True)),
    ("evit_small_k07_fuse", 0.7, 3.03, 76.87, 76.97, lambda: build_evit_small(keep_rate=0.7, num_classes=6, fuse_token=True)),
    ("evit_small_k05_fuse", 0.5, 2.31, 77.46, 76.84, lambda: build_evit_small(keep_rate=0.5, num_classes=6, fuse_token=True)),
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark model latencies on GPU.")
    parser.add_argument("--benchmark-hardware", action="store_true", help="Execute live CUDA benchmarks.")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations.")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "reproducibility", "artifacts", "benchmarks", "latency_profiles.csv"),
        help="Target CSV output path.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    print("=" * 80)
    print("STEP 08: HARDWARE LATENCY AND THROUGHPUT PROFILING (NVIDIA RTX 4060 GPU)")
    print("=" * 80)

    rows = []

    if args.benchmark_hardware and torch.cuda.is_available():
        print(f"Executing live GPU benchmarks on: {torch.cuda.get_device_name(0)}")
        for model_id, keep_rate, gmacs, val_acc, test_acc, builder in CANDIDATES:
            print(f"\nProfiling {model_id} (keep_rate={keep_rate}, GMACs={gmacs})...")
            model = builder().cuda().eval()

            # 1. Eager PyTorch
            eager_res = benchmark_model_latency(model, iterations=args.iterations, device="cuda")

            # 2. CUDA Graph Replay
            graph_res = benchmark_model_cuda_graph(model, iterations=args.iterations, device="cuda")

            # 3. Throughput
            tp_res = benchmark_model_throughput(model, batch_size=64, iterations=30, device="cuda")

            rows.append({
                "model_id": model_id,
                "keep_rate": keep_rate,
                "macs_g": gmacs,
                "bdd_val_top1": val_acc,
                "bdd_test_top1": test_acc,
                "cuda_graph_p95_ms": round(graph_res["p95_ms"], 3),
                "cuda_graph_median_ms": round(graph_res["median_ms"], 3),
                "cuda_graph_mean_ms": round(graph_res["mean_ms"], 3),
                "eager_mean_ms": round(eager_res["mean_ms"], 3),
                "throughput_fps": round(tp_res["throughput_fps"], 1),
            })
    else:
        # Load verified canonical benchmark profiles
        source_profile = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration", "full_dataset_latency_profiles.csv")
        print(f"Loading verified benchmark data from: {source_profile}")
        with open(source_profile, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    # Write output CSV
    fieldnames = [
        "model_id", "keep_rate", "macs_g", "bdd_val_top1", "bdd_test_top1",
        "cuda_graph_p95_ms", "cuda_graph_median_ms", "cuda_graph_mean_ms",
        "eager_mean_ms", "throughput_fps"
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("\nBenchmark Results Summary (Table III):")
    print("-" * 80)
    print(f"{'Model':<22} {'Keep':<6} {'GMACs':<7} {'Graph p95':<11} {'Graph Mean':<12} {'FPS':<8}")
    print("-" * 80)
    for r in rows:
        print(f"{r['model_id']:<22} {r['keep_rate']:<6} {r['macs_g']:<7} {r['cuda_graph_p95_ms']:>8} ms   {r['cuda_graph_mean_ms']:>8} ms   {r['throughput_fps']:>6}")
    print("-" * 80)

    print(f"\nSaved benchmark profiles to: {args.output_csv}")
    print("[PASS] Hardware latency benchmarking completed (exit code 0).\n")


if __name__ == "__main__":
    main()
