"""Simulation study for Phase 10C accuracy-constrained adaptive scheduler.

Evaluates 4 policies on synthetic deadline streams across workloads:
  1. always_full_attention
  2. always_fastest
  3. deadline_only_adaptive (Phase 7 controller logic on 3 candidates)
  4. accuracy_constrained_adaptive (Phase 10C scheduler across multiple tau_miou thresholds)

Computes:
  - Deadline hit rate (%)
  - Quality constraint compliance rate (%)
  - Average selected mIoU
  - Candidate selection distributions
  - Infeasibility and fallback frequencies
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtime.accuracy_scheduler import (
    AccuracyCandidateProfile,
    load_accuracy_candidates,
    select_accuracy_constrained_mode,
)
from realtime.simulate_strategies import WORKLOAD_CONFIGS, generate_deadline_stream

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase10c_adaptive_scheduler"
BENCHMARK_PATH = OUTPUT_DIR / "benchmark_results.json"

QUALITY_FLOORS = [0.45, 0.65, 0.68, 0.70, 0.73]


def run_simulation() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = load_accuracy_candidates(BENCHMARK_PATH)
    candidate_names = [c.name for c in candidates]

    # Generate streams for each workload
    stream_size = 100
    workloads = ["default_mixture", "stress_tight"]

    all_simulation_results: dict[str, Any] = {}
    detailed_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []

    for workload_name in workloads:
        stream = generate_deadline_stream(
            workload_name=workload_name,
            n_requests=stream_size,
            seed=42 if workload_name == "default_mixture" else 1042,
        )

        all_simulation_results[workload_name] = {}

        # 1. Baseline: always_full_attention
        full_cand = next(c for c in candidates if c.name == "full_attention")
        full_p95 = full_cand.latency_for_metric("p95")

        full_hits = sum(1 for req in stream if full_p95 <= req.deadline_ms)
        all_simulation_results[workload_name]["always_full_attention"] = {
            "hit_rate_pct": round(full_hits / stream_size * 100.0, 2),
            "mean_miou": round(full_cand.validation_miou, 4),
            "selection_counts": {"full_attention": stream_size, "moderate_dwconv": 0, "sigma_b_dwconv": 0},
            "rejected_count": 0,
        }

        # 2. Baseline: always_fastest
        # Among candidates, find fastest by p95 latency
        fastest_cand = min(
            candidates,
            key=lambda c: (c.latency_for_metric("p95"), -c.validation_miou, c.name),
        )
        fast_p95 = fastest_cand.latency_for_metric("p95")
        fast_hits = sum(1 for req in stream if fast_p95 <= req.deadline_ms)
        all_simulation_results[workload_name]["always_fastest"] = {
            "hit_rate_pct": round(fast_hits / stream_size * 100.0, 2),
            "mean_miou": round(fastest_cand.validation_miou, 4),
            "selected_mode": fastest_cand.name,
            "selection_counts": {
                name: (stream_size if name == fastest_cand.name else 0) for name in candidate_names
            },
            "rejected_count": 0,
        }

        # 3. Policy: deadline_only_adaptive (Phase 7 controller logic: tau_miou = 0.0)
        dead_hits = 0
        dead_mious = []
        dead_counts = {name: 0 for name in candidate_names}
        for req in stream:
            res = select_accuracy_constrained_mode(
                remaining_deadline_ms=req.deadline_ms,
                candidates=candidates,
                min_miou_threshold=0.0,
                latency_metric="p95",
                fallback_policy="fastest_miss",
            )
            dead_counts[res.selected_candidate] += 1
            dead_mious.append(res.selected_miou)
            if res.selected_latency_bound_ms <= req.deadline_ms:
                dead_hits += 1

            detailed_rows.append({
                "workload": workload_name,
                "request_id": req.request_id,
                "deadline_ms": round(req.deadline_ms, 2),
                "workload_category": req.workload_category,
                "policy": "deadline_only_adaptive",
                "tau_miou": 0.0,
                "selected_candidate": res.selected_candidate,
                "selected_miou": round(res.selected_miou, 4),
                "latency_bound_ms": round(res.selected_latency_bound_ms, 2),
                "deadline_hit": res.selected_latency_bound_ms <= req.deadline_ms,
                "is_fallback": res.is_fallback,
                "is_rejected": res.is_rejected,
            })

        all_simulation_results[workload_name]["deadline_only_adaptive"] = {
            "hit_rate_pct": round(dead_hits / stream_size * 100.0, 2),
            "mean_miou": round(sum(dead_mious) / len(dead_mious), 4),
            "selection_counts": dead_counts,
            "rejected_count": 0,
        }

        # 4. Policy: accuracy_constrained_adaptive across quality floors
        all_simulation_results[workload_name]["accuracy_constrained"] = {}
        for tau in QUALITY_FLOORS:
            for fallback_policy in ["highest_quality_miss", "reject", "fastest_miss"]:
                pol_key = f"tau_{tau:.2f}_{fallback_policy}"
                hits = 0
                quality_compliant = 0
                mious = []
                counts = {name: 0 for name in candidate_names}
                rejected = 0

                for req in stream:
                    res = select_accuracy_constrained_mode(
                        remaining_deadline_ms=req.deadline_ms,
                        candidates=candidates,
                        min_miou_threshold=tau,
                        latency_metric="p95",
                        fallback_policy=fallback_policy,
                    )
                    if res.is_rejected:
                        rejected += 1
                    else:
                        counts[res.selected_candidate] += 1
                        mious.append(res.selected_miou)
                        if res.selected_miou >= tau:
                            quality_compliant += 1
                        if res.selected_latency_bound_ms <= req.deadline_ms:
                            hits += 1

                    detailed_rows.append({
                        "workload": workload_name,
                        "request_id": req.request_id,
                        "deadline_ms": round(req.deadline_ms, 2),
                        "workload_category": req.workload_category,
                        "policy": f"accuracy_constrained_{fallback_policy}",
                        "tau_miou": tau,
                        "selected_candidate": res.selected_candidate if not res.is_rejected else "REJECTED",
                        "selected_miou": round(res.selected_miou, 4) if res.selected_miou is not None else 0.0,
                        "latency_bound_ms": round(res.selected_latency_bound_ms, 2) if res.selected_latency_bound_ms is not None else 0.0,
                        "deadline_hit": (res.selected_latency_bound_ms <= req.deadline_ms) if not res.is_rejected else False,
                        "is_fallback": res.is_fallback,
                        "is_rejected": res.is_rejected,
                    })

                hit_rate = round(hits / stream_size * 100.0, 2)
                qual_rate = round(quality_compliant / stream_size * 100.0, 2)
                avg_miou = round(sum(mious) / len(mious), 4) if mious else 0.0

                result_entry = {
                    "tau_miou": tau,
                    "fallback_policy": fallback_policy,
                    "hit_rate_pct": hit_rate,
                    "quality_compliance_pct": qual_rate,
                    "mean_miou": avg_miou,
                    "selection_counts": counts,
                    "rejected_count": rejected,
                }
                all_simulation_results[workload_name]["accuracy_constrained"][pol_key] = result_entry

                sweep_rows.append({
                    "workload": workload_name,
                    "tau_miou": tau,
                    "fallback_policy": fallback_policy,
                    "hit_rate_pct": hit_rate,
                    "quality_compliance_pct": qual_rate,
                    "mean_miou": avg_miou,
                    "rejected_count": rejected,
                    "full_attention_count": counts.get("full_attention", 0),
                    "moderate_dwconv_count": counts.get("moderate_dwconv", 0),
                    "sigma_b_dwconv_count": counts.get("sigma_b_dwconv", 0),
                })

    return all_simulation_results, detailed_rows, sweep_rows


def main() -> None:
    print("=" * 70)
    print("Phase 10C: Accuracy-Constrained Scheduler Simulation Study")
    print("=" * 70)

    sim_results, detailed_rows, sweep_rows = run_simulation()

    # Save outputs
    json_path = OUTPUT_DIR / "simulation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sim_results, f, indent=2)
    print(f"Saved simulation summary to: {json_path}")

    csv_detailed_path = OUTPUT_DIR / "simulation_detailed_trace.csv"
    with open(csv_detailed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"Saved detailed simulation trace to: {csv_detailed_path}")

    csv_sweep_path = OUTPUT_DIR / "quality_floor_sweep.csv"
    with open(csv_sweep_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"Saved quality floor sweep to: {csv_sweep_path}")

    # Print summary table for default_mixture
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY — default_mixture (100 requests):")
    print(f"{'Policy / Setting':<38} | {'Hit Rate':<10} | {'Qual Comp':<10} | {'Avg mIoU':<8}")
    print("-" * 70)
    dm = sim_results["default_mixture"]
    print(f"{'Always Full Attention':<38} | {dm['always_full_attention']['hit_rate_pct']:<10.1f}% | {'100.0%':<10} | {dm['always_full_attention']['mean_miou']:<8.4f}")
    print(f"{'Always Fastest (' + dm['always_fastest']['selected_mode'] + ')':<38} | {dm['always_fastest']['hit_rate_pct']:<10.1f}% | {'100.0%':<10} | {dm['always_fastest']['mean_miou']:<8.4f}")
    print(f"{'Deadline-Only Adaptive':<38} | {dm['deadline_only_adaptive']['hit_rate_pct']:<10.1f}% | {'--':<10} | {dm['deadline_only_adaptive']['mean_miou']:<8.4f}")
    print("-" * 70)
    print("Accuracy-Constrained (fallback=highest_quality_miss):")
    for tau in QUALITY_FLOORS:
        k = f"tau_{tau:.2f}_highest_quality_miss"
        entry = dm["accuracy_constrained"][k]
        print(f"  tau={tau:.2f} ({entry['fallback_policy']}){'':<14} | {entry['hit_rate_pct']:<10.1f}% | {entry['quality_compliance_pct']:<10.1f}% | {entry['mean_miou']:<8.4f}")
    print("-" * 70)
    print("Accuracy-Constrained (fallback=reject):")
    for tau in QUALITY_FLOORS:
        k = f"tau_{tau:.2f}_reject"
        entry = dm["accuracy_constrained"][k]
        print(f"  tau={tau:.2f} ({entry['fallback_policy']}){'':<26} | {entry['hit_rate_pct']:<10.1f}% | {entry['quality_compliance_pct']:<10.1f}% | {entry['mean_miou']:<8.4f} (Rej: {entry['rejected_count']})")
    print("=" * 70)


if __name__ == "__main__":
    main()
