"""Phase 10D Part B: Multi-round Latency Stability and Paired Differences Study.

Measures:
  1. Full Attention (Phase 4 Baseline)
  2. Moderate DWConv (Phase 6B Moderate, [2, 5, 8, 11])
  3. Sigma-selected DWConv (Phase 10B Run 3, [1, 2, 3, 4])

Design:
  - 5 independent measurement rounds (sessions) of 30 iterations each (150 total per model).
  - Randomized model execution order within each iteration to eliminate thermal drift confounding.
  - Consistent warmup policy per round (10 warmups).
  - Measures BOTH model-only (GPU host + CUDA event) and end-to-end (prep + transfer + model + post) latencies.
  - Computes round-level and pooled distribution summaries (median, mean, std, p90, p95, p99, max).
  - Computes paired differences (candidate A - candidate B) on matched iterations.
  - Computes bootstrap 95% confidence intervals for median differences.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ["XFORMERS_DISABLED"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from realtime.benchmark_candidates_phase10c import (
    CANDIDATE_DEFS,
    compute_summary_stats,
    load_candidate_models,
)
from realtime.runtime_inference import execute_model_forward, preprocess_image

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase10d_realtime_validation"
PET_DATASET_DIR = PROJECT_ROOT / "datasets" / "oxford-iiit-pet" / "images"


def compute_bootstrap_ci_median_diff(
    diffs: list[float], num_resamples: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(diffs, dtype=np.float64)
    point_est = float(np.median(arr))
    boot_medians = [
        float(np.median(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(num_resamples)
    ]
    ci_lower = float(np.percentile(boot_medians, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    return round(point_est, 4), round(ci_lower, 4), round(ci_upper, 4)


def run_latency_stability_study(
    device: torch.device,
    models: dict[str, tuple[torch.nn.Module, Any]],
    image_paths: list[Path],
    num_rounds: int = 5,
    iters_per_round: int = 30,
    warmup_per_round: int = 10,
    seed: int = 1042,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    candidate_names = list(models.keys())

    round_rows: list[dict[str, Any]] = []
    pooled_by_model: dict[str, dict[str, list[float]]] = {
        c: {"model_host_ms": [], "model_cuda_ms": [], "e2e_ms": []}
        for c in candidate_names
    }
    per_round_summaries: dict[str, dict[str, Any]] = {c: {} for c in candidate_names}

    # Paired matching: record by (round, iter)
    paired_data: list[dict[str, Any]] = []

    for round_idx in range(1, num_rounds + 1):
        # Round warmup
        dummy = torch.randn(1, 3, 224, 224, device=device)
        with torch.inference_mode():
            for name in candidate_names:
                bb, hd = models[name]
                for _ in range(warmup_per_round):
                    feat = bb.forward_features(dummy) if hasattr(bb, "forward_features") else bb(dummy)
                    # Use standard features call
                torch.cuda.synchronize(device)

        round_model_times: dict[str, dict[str, list[float]]] = {
            c: {"model_host_ms": [], "model_cuda_ms": [], "e2e_ms": []}
            for c in candidate_names
        }

        for iter_idx in range(1, iters_per_round + 1):
            shuffled_candidates = list(candidate_names)
            rng.shuffle(shuffled_candidates)
            img_path = image_paths[(iter_idx + round_idx * 17) % len(image_paths)]

            iter_record: dict[str, Any] = {
                "round_id": round_idx,
                "iter_id": iter_idx,
                "image_name": img_path.name,
            }

            for cand_name in shuffled_candidates:
                bb, hd = models[cand_name]

                t0_req = time.perf_counter()
                t0_prep = time.perf_counter()
                img_tensor = preprocess_image(img_path)
                t1_prep = time.perf_counter()
                prep_ms = (t1_prep - t0_prep) * 1000.0

                t0_trans = time.perf_counter()
                img_cuda = img_tensor.to(device, non_blocking=False)
                torch.cuda.synchronize(device)
                t1_trans = time.perf_counter()
                trans_ms = (t1_trans - t0_trans) * 1000.0

                preds_cpu, model_host_ms, model_cuda_ms = execute_model_forward(
                    bb, hd, img_cuda
                )

                t0_post = time.perf_counter()
                _ = preds_cpu.byte().numpy()
                t1_post = time.perf_counter()
                post_ms = (t1_post - t0_post) * 1000.0

                t1_req = time.perf_counter()
                e2e_ms = (t1_req - t0_req) * 1000.0

                # Accumulate
                pooled_by_model[cand_name]["model_host_ms"].append(model_host_ms)
                pooled_by_model[cand_name]["model_cuda_ms"].append(model_cuda_ms)
                pooled_by_model[cand_name]["e2e_ms"].append(e2e_ms)

                round_model_times[cand_name]["model_host_ms"].append(model_host_ms)
                round_model_times[cand_name]["model_cuda_ms"].append(model_cuda_ms)
                round_model_times[cand_name]["e2e_ms"].append(e2e_ms)

                iter_record[f"{cand_name}_model_ms"] = round(model_host_ms, 4)
                iter_record[f"{cand_name}_cuda_ms"] = round(model_cuda_ms, 4)
                iter_record[f"{cand_name}_e2e_ms"] = round(e2e_ms, 4)

                round_rows.append({
                    "round_id": round_idx,
                    "iter_id": iter_idx,
                    "candidate": cand_name,
                    "image_name": img_path.name,
                    "prep_ms": round(prep_ms, 4),
                    "trans_ms": round(trans_ms, 4),
                    "model_host_ms": round(model_host_ms, 4),
                    "model_cuda_ms": round(model_cuda_ms, 4),
                    "post_ms": round(post_ms, 4),
                    "e2e_ms": round(e2e_ms, 4),
                })

            paired_data.append(iter_record)

        # Summarize this round
        for cand_name in candidate_names:
            per_round_summaries[cand_name][f"round_{round_idx}"] = {
                "model_host_ms": compute_summary_stats(
                    round_model_times[cand_name]["model_host_ms"]
                ),
                "model_cuda_ms": compute_summary_stats(
                    round_model_times[cand_name]["model_cuda_ms"]
                ),
                "e2e_ms": compute_summary_stats(
                    round_model_times[cand_name]["e2e_ms"]
                ),
            }

    # Summarize pooled
    pooled_summary: dict[str, Any] = {
        "metadata": {
            "device": torch.cuda.get_device_name(device),
            "num_rounds": num_rounds,
            "iters_per_round": iters_per_round,
            "total_iterations_per_candidate": num_rounds * iters_per_round,
            "warmup_per_round": warmup_per_round,
        },
        "per_round": per_round_summaries,
        "pooled": {
            cand_name: {
                "model_host_ms": compute_summary_stats(
                    pooled_by_model[cand_name]["model_host_ms"]
                ),
                "model_cuda_ms": compute_summary_stats(
                    pooled_by_model[cand_name]["model_cuda_ms"]
                ),
                "e2e_ms": compute_summary_stats(
                    pooled_by_model[cand_name]["e2e_ms"]
                ),
            }
            for cand_name in candidate_names
        },
    }

    # Paired differences analysis
    # Comparisons:
    #   moderate_dwconv vs full_attention (moderate - full)
    #   sigma_b_dwconv vs full_attention (sigma - full)
    #   sigma_b_dwconv vs moderate_dwconv (sigma - moderate)
    pairs = [
        ("moderate_dwconv", "full_attention"),
        ("sigma_b_dwconv", "full_attention"),
        ("sigma_b_dwconv", "moderate_dwconv"),
    ]

    paired_analysis: dict[str, Any] = {}
    for c_a, c_b in pairs:
        pair_key = f"{c_a}_minus_{c_b}"
        model_diffs = [
            r[f"{c_a}_model_ms"] - r[f"{c_b}_model_ms"] for r in paired_data
        ]
        e2e_diffs = [
            r[f"{c_a}_e2e_ms"] - r[f"{c_b}_e2e_ms"] for r in paired_data
        ]

        m_point, m_lo, m_hi = compute_bootstrap_ci_median_diff(model_diffs)
        e_point, e_lo, e_hi = compute_bootstrap_ci_median_diff(e2e_diffs)

        # Proportion where A was faster than B
        faster_m_count = sum(1 for d in model_diffs if d < 0)
        faster_m_pct = round(faster_m_count / len(model_diffs) * 100.0, 2)

        paired_analysis[pair_key] = {
            "candidate_A": c_a,
            "candidate_B": c_b,
            "sample_size": len(model_diffs),
            "model_latency_diff": {
                "median_diff_ms": m_point,
                "bootstrap_95ci_median_diff": [m_lo, m_hi],
                "mean_diff_ms": round(float(np.mean(model_diffs)), 4),
                "std_diff_ms": round(float(np.std(model_diffs)), 4),
                "candidate_A_faster_percentage": faster_m_pct,
            },
            "e2e_latency_diff": {
                "median_diff_ms": e_point,
                "bootstrap_95ci_median_diff": [e_lo, e_hi],
                "mean_diff_ms": round(float(np.mean(e2e_diffs)), 4),
                "std_diff_ms": round(float(np.std(e2e_diffs)), 4),
            },
        }

    return round_rows, pooled_summary, paired_analysis


def main() -> None:
    print("=" * 70)
    print("Phase 10D Part B: Multi-round Latency Stability & Paired Differences")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    models = load_candidate_models(device)
    images = sorted(PET_DATASET_DIR.glob("*.jpg"))[:32]
    print(f"Loaded {len(images)} sample images. Executing 5 rounds x 30 iterations...")

    t0 = time.time()
    round_rows, pooled_summary, paired_analysis = run_latency_stability_study(
        device=device,
        models=models,
        image_paths=images,
        num_rounds=5,
        iters_per_round=30,
        warmup_per_round=10,
        seed=1042,
    )
    t1 = time.time()
    print(f"Measurements finished in {t1 - t0:.2f}s.")

    # Save latency_rounds.csv
    csv_path = OUTPUT_DIR / "latency_rounds.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(round_rows[0].keys()))
        writer.writeheader()
        writer.writerows(round_rows)
    print(f"Saved round-level traces to: {csv_path}")

    # Save latency_summary.json
    sum_json = OUTPUT_DIR / "latency_summary.json"
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(pooled_summary, f, indent=2)
    print(f"Saved pooled and per-round latency summary to: {sum_json}")

    # Save paired_latency_analysis.json
    paired_json = OUTPUT_DIR / "paired_latency_analysis.json"
    with open(paired_json, "w", encoding="utf-8") as f:
        json.dump(paired_analysis, f, indent=2)
    print(f"Saved paired latency analysis to: {paired_json}")

    # Print summary table
    print("\n" + "=" * 70)
    print("POOLED LATENCY SUMMARY (150 measurements per candidate):")
    print(f"{'Candidate':<18} | {'Scope':<10} | {'Median':<8} | {'Mean':<8} | {'p90':<8} | {'p95':<8} | {'p99':<8} | {'Max':<8}")
    print("-" * 70)
    for c_name, c_data in pooled_summary["pooled"].items():
        m_host = c_data["model_host_ms"]
        e2e = c_data["e2e_ms"]
        print(f"{c_name:<18} | {'Model':<10} | {m_host['median']:<8.2f} | {m_host['mean']:<8.2f} | {m_host['p90']:<8.2f} | {m_host['p95']:<8.2f} | {m_host['p99']:<8.2f} | {m_host['max']:<8.2f}")
        print(f"{'':<18} | {'E2E':<10} | {e2e['median']:<8.2f} | {e2e['mean']:<8.2f} | {e2e['p90']:<8.2f} | {e2e['p95']:<8.2f} | {e2e['p99']:<8.2f} | {e2e['max']:<8.2f}")
    print("=" * 70)

    print("\nPAIRED LATENCY DIFFERENCES (Candidate A - Candidate B, 150 matched iterations):")
    for k, v in paired_analysis.items():
        m_diff = v["model_latency_diff"]
        print(f"  {k}:")
        print(f"    Median Diff: {m_diff['median_diff_ms']:+.3f} ms (95% CI: [{m_diff['bootstrap_95ci_median_diff'][0]:+.3f}, {m_diff['bootstrap_95ci_median_diff'][1]:+.3f}])")
        print(f"    Candidate A Faster: {m_diff['candidate_A_faster_percentage']:.1f}% of iterations")


if __name__ == "__main__":
    main()
