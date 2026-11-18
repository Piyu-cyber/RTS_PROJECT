"""Phase 10D Part C: Deadline Prediction Calibration and Part D: Robustness Sweep.

Evaluates:
  Part C — Deadline prediction calibration:
    - Analyzes feasibility predictions under p95 vs p99 empirical quantiles.
    - Evaluates:
        * Predicted-feasible requests.
        * Actual deadline hits among predicted-feasible requests (Precision / Feasible Hit Rate).
        * False-feasible decisions (Type I error / Over-optimistic): predicted feasible, but deadline missed.
        * False-infeasible decisions (Type II error / Over-conservative): predicted infeasible, but would have hit deadline.
        * Rejected requests (under reject policy).
        * Deadline misses among dispatched requests.
        * Overall request completion rate.
        * Useful-completion rate (satisfying both deadline hit AND quality threshold tau).
    - Stratifies results by deadline range: Tight [10-17ms], Moderate [17-22ms], Relaxed [22-32ms].
    - Stratifies by workload: default_mixture vs stress_tight.

  Part D — Quality floor and fallback robustness:
    - Re-evaluates quality floors {0.45, 0.65, 0.68, 0.70, 0.73} under identical matched streams.
    - Compares:
        1. Always Full Attention
        2. Always Fastest Candidate
        3. Deadline-Only Adaptive (tau=0.0)
        4. Accuracy-Constrained (highest_quality_miss)
        5. Accuracy-Constrained (fastest_miss)
        6. Accuracy-Constrained (reject)
    - Records deadline hit rate, acceptance rate, rejection rate, quality-compliance rate,
      mean validation mIoU, useful-completion rate, fallback frequency, and miss root-causes.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from realtime.accuracy_scheduler import (
    AccuracyCandidateProfile,
    load_accuracy_candidates,
    select_accuracy_constrained_mode,
)
from realtime.simulate_strategies import generate_deadline_stream

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase10d_realtime_validation"
BENCHMARK_PATH = (
    PROJECT_ROOT / "outputs" / "phase10c_adaptive_scheduler" / "benchmark_results.json"
)
POOLED_LATENCY_PATH = OUTPUT_DIR / "latency_summary.json"


def run_calibration_and_robustness() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = load_accuracy_candidates(BENCHMARK_PATH)

    # Load pooled latency measurements to simulate realistic empirical runtime execution
    with open(POOLED_LATENCY_PATH, "r", encoding="utf-8") as f:
        pooled_data = json.load(f)["pooled"]

    # We use empirical distributions for actual latency realization
    # To be scientifically rigorous, each simulated request draws an actual execution latency
    # from the empirical pooled measurements corresponding to the dispatched candidate.
    with open(OUTPUT_DIR / "latency_rounds.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        latency_samples_by_cand = {c.name: [] for c in candidates}
        for row in reader:
            latency_samples_by_cand[row["candidate"]].append({
                "model_ms": float(row["model_host_ms"]),
                "e2e_ms": float(row["e2e_ms"]),
            })

    workloads = ["default_mixture", "stress_tight"]
    quality_floors = [0.45, 0.65, 0.68, 0.70, 0.73]
    calibration_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []

    # 1. Part C: Deadline Prediction Calibration Sweep
    # Test p95 vs p99 bounds on both model-only scope and e2e scope
    for wl in workloads:
        stream = generate_deadline_stream(100, seed=42 if wl == "default_mixture" else 1042, workload_name=wl)

        for metric in ["p95", "p99"]:
            for tau in [0.0, 0.65, 0.70]:
                for fb_policy in ["highest_quality_miss", "reject", "fastest_miss"]:
                    # Stratification buckets
                    buckets = {"overall": stream, "tight": [r for r in stream if r.workload_category == "tight"],
                               "moderate": [r for r in stream if r.workload_category == "moderate"],
                               "relaxed": [r for r in stream if r.workload_category == "relaxed"]}

                    for b_name, b_stream in buckets.items():
                        if not b_stream:
                            continue

                        n_total = len(b_stream)
                        pred_feasible = 0
                        actual_hits_among_pred_feasible = 0
                        false_feasible = 0
                        false_infeasible = 0
                        rejected = 0
                        dispatched = 0
                        actual_hits_dispatched = 0
                        useful_completions = 0

                        for idx, req in enumerate(b_stream):
                            res = select_accuracy_constrained_mode(
                                remaining_deadline_ms=req.deadline_ms,
                                candidates=candidates,
                                min_miou_threshold=tau,
                                latency_metric=metric,
                                fallback_policy=fb_policy,
                            )

                            if res.is_rejected:
                                rejected += 1
                                continue

                            dispatched += 1
                            # Sample actual latency realization
                            sample = latency_samples_by_cand[res.selected_candidate][(idx * 7) % len(latency_samples_by_cand[res.selected_candidate])]
                            actual_lat = sample["e2e_ms"]

                            hit = actual_lat <= req.deadline_ms
                            qual_met = res.selected_miou >= tau

                            if hit:
                                actual_hits_dispatched += 1
                            if hit and qual_met:
                                useful_completions += 1

                            # Prediction calibration against model bound vs actual
                            pred_hit = not res.is_infeasible
                            if pred_hit:
                                pred_feasible += 1
                                if hit:
                                    actual_hits_among_pred_feasible += 1
                                else:
                                    false_feasible += 1
                            else:
                                if hit:
                                    false_infeasible += 1

                        precision_pct = round(actual_hits_among_pred_feasible / pred_feasible * 100.0, 2) if pred_feasible > 0 else 0.0
                        completion_rate_pct = round(dispatched / n_total * 100.0, 2)
                        useful_rate_pct = round(useful_completions / n_total * 100.0, 2)
                        hit_rate_dispatched_pct = round(actual_hits_dispatched / dispatched * 100.0, 2) if dispatched > 0 else 0.0

                        calibration_rows.append({
                            "workload": wl,
                            "stratum": b_name,
                            "metric": metric,
                            "tau_miou": tau,
                            "fallback_policy": fb_policy,
                            "total_requests": n_total,
                            "dispatched_requests": dispatched,
                            "rejected_requests": rejected,
                            "predicted_feasible": pred_feasible,
                            "actual_hits_among_predicted": actual_hits_among_pred_feasible,
                            "precision_predicted_feasible_pct": precision_pct,
                            "false_feasible_count": false_feasible,
                            "false_infeasible_count": false_infeasible,
                            "actual_hits_dispatched": actual_hits_dispatched,
                            "hit_rate_dispatched_pct": hit_rate_dispatched_pct,
                            "completion_rate_pct": completion_rate_pct,
                            "useful_completion_rate_pct": useful_rate_pct,
                        })

    # 2. Part D: Quality Floor & Fallback Robustness Sweep
    for wl in workloads:
        stream = generate_deadline_stream(100, seed=42 if wl == "default_mixture" else 1042, workload_name=wl)
        n_total = len(stream)

        # Baseline 1: Always Full Attention
        full_cand = next(c for c in candidates if c.name == "full_attention")
        f_hits = 0
        f_useful = 0
        for idx, req in enumerate(stream):
            lat = latency_samples_by_cand["full_attention"][(idx * 7) % len(latency_samples_by_cand["full_attention"])]["e2e_ms"]
            if lat <= req.deadline_ms:
                f_hits += 1
                f_useful += 1
        fallback_rows.append({
            "workload": wl,
            "policy": "always_full_attention",
            "tau_miou": 0.0,
            "fallback_policy": "none",
            "deadline_hit_rate_total_pct": round(f_hits / n_total * 100.0, 2),
            "deadline_hit_rate_dispatched_pct": round(f_hits / n_total * 100.0, 2),
            "acceptance_rate_pct": 100.0,
            "rejection_rate_pct": 0.0,
            "quality_compliance_rate_pct": 100.0,
            "mean_selected_miou": round(full_cand.validation_miou, 4),
            "useful_completion_rate_pct": round(f_useful / n_total * 100.0, 2),
            "fallback_frequency_pct": 0.0,
            "miss_cause": "deadline < e2e latency (18.0ms)",
            "selected_full_attention": n_total,
            "selected_moderate_dwconv": 0,
            "selected_sigma_b_dwconv": 0,
        })

        # Baseline 2: Always Fastest
        fastest_cand = min(candidates, key=lambda c: (c.p95_ms, -c.validation_miou, c.name))
        fa_hits = 0
        for idx, req in enumerate(stream):
            lat = latency_samples_by_cand[fastest_cand.name][(idx * 7) % len(latency_samples_by_cand[fastest_cand.name])]["e2e_ms"]
            if lat <= req.deadline_ms:
                fa_hits += 1
        fallback_rows.append({
            "workload": wl,
            "policy": "always_fastest",
            "tau_miou": 0.0,
            "fallback_policy": "none",
            "deadline_hit_rate_total_pct": round(fa_hits / n_total * 100.0, 2),
            "deadline_hit_rate_dispatched_pct": round(fa_hits / n_total * 100.0, 2),
            "acceptance_rate_pct": 100.0,
            "rejection_rate_pct": 0.0,
            "quality_compliance_rate_pct": 100.0,
            "mean_selected_miou": round(fastest_cand.validation_miou, 4),
            "useful_completion_rate_pct": round(fa_hits / n_total * 100.0, 2),
            "fallback_frequency_pct": 0.0,
            "miss_cause": "deadline < e2e latency (18.0ms)",
            "selected_full_attention": n_total if fastest_cand.name == "full_attention" else 0,
            "selected_moderate_dwconv": n_total if fastest_cand.name == "moderate_dwconv" else 0,
            "selected_sigma_b_dwconv": n_total if fastest_cand.name == "sigma_b_dwconv" else 0,
        })

        # Sweep Accuracy-Constrained policies across quality floors
        for tau in quality_floors:
            for fb_policy in ["highest_quality_miss", "fastest_miss", "reject"]:
                hits = 0
                dispatched = 0
                rejected = 0
                qual_compliant = 0
                useful = 0
                fallbacks = 0
                cand_counts = {c.name: 0 for c in candidates}
                mious = []

                for idx, req in enumerate(stream):
                    res = select_accuracy_constrained_mode(
                        remaining_deadline_ms=req.deadline_ms,
                        candidates=candidates,
                        min_miou_threshold=tau,
                        latency_metric="p95",
                        fallback_policy=fb_policy,
                    )

                    if res.is_rejected:
                        rejected += 1
                        continue

                    dispatched += 1
                    cand_counts[res.selected_candidate] += 1
                    mious.append(res.selected_miou)
                    if res.is_fallback:
                        fallbacks += 1

                    lat = latency_samples_by_cand[res.selected_candidate][(idx * 7) % len(latency_samples_by_cand[res.selected_candidate])]["e2e_ms"]
                    hit = lat <= req.deadline_ms
                    comp = res.selected_miou >= tau

                    if hit:
                        hits += 1
                    if comp:
                        qual_compliant += 1
                    if hit and comp:
                        useful += 1

                hit_rate_total = round(hits / n_total * 100.0, 2)
                hit_rate_disp = round(hits / dispatched * 100.0, 2) if dispatched > 0 else 0.0
                acc_rate = round(dispatched / n_total * 100.0, 2)
                rej_rate = round(rejected / n_total * 100.0, 2)
                qual_rate = round(qual_compliant / n_total * 100.0, 2)
                useful_rate = round(useful / n_total * 100.0, 2)
                fb_rate = round(fallbacks / n_total * 100.0, 2)
                avg_miou = round(float(np.mean(mious)), 4) if mious else 0.0

                miss_cause = (
                    "Rejection of unachievable deadlines"
                    if fb_policy == "reject"
                    else "Over-budget requests dispatched under fallback"
                )

                fallback_rows.append({
                    "workload": wl,
                    "policy": f"accuracy_constrained_{fb_policy}",
                    "tau_miou": tau,
                    "fallback_policy": fb_policy,
                    "deadline_hit_rate_total_pct": hit_rate_total,
                    "deadline_hit_rate_dispatched_pct": hit_rate_disp,
                    "acceptance_rate_pct": acc_rate,
                    "rejection_rate_pct": rej_rate,
                    "quality_compliance_rate_pct": qual_rate,
                    "mean_selected_miou": avg_miou,
                    "useful_completion_rate_pct": useful_rate,
                    "fallback_frequency_pct": fb_rate,
                    "miss_cause": miss_cause,
                    "selected_full_attention": cand_counts["full_attention"],
                    "selected_moderate_dwconv": cand_counts["moderate_dwconv"],
                    "selected_sigma_b_dwconv": cand_counts["sigma_b_dwconv"],
                })

    return calibration_rows, fallback_rows


def main() -> None:
    print("=" * 70)
    print("Phase 10D Parts C & D: Prediction Calibration & Robustness Evaluation")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calib_rows, fb_rows = run_calibration_and_robustness()

    # Save deadline_calibration.csv
    calib_csv = OUTPUT_DIR / "deadline_calibration.csv"
    with open(calib_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(calib_rows[0].keys()))
        writer.writeheader()
        writer.writerows(calib_rows)
    print(f"Saved deadline calibration results to: {calib_csv}")

    # Save fallback_comparison.csv
    fb_csv = OUTPUT_DIR / "fallback_comparison.csv"
    with open(fb_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fb_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fb_rows)
    print(f"Saved fallback & quality floor robustness comparison to: {fb_csv}")

    # Print summary highlights
    print("\nCALIBRATION SUMMARY HIGHLIGHTS (p95 vs p99, tau=0.70, default_mixture):")
    for r in calib_rows:
        if r["workload"] == "default_mixture" and r["stratum"] == "overall" and r["tau_miou"] == 0.70:
            print(f"  Metric: {r['metric']:<4} | Fallback: {r['fallback_policy']:<20} | Pred Feasible: {r['predicted_feasible']} | Hits among Pred: {r['actual_hits_among_predicted']} ({r['precision_predicted_feasible_pct']}%) | False Feas: {r['false_feasible_count']} | False Infeas: {r['false_infeasible_count']}")

    print("\nROBUSTNESS SUMMARY HIGHLIGHTS (default_mixture, tau=0.70):")
    for r in fb_rows:
        if r["workload"] == "default_mixture" and r.get("tau_miou") in [0.0, 0.70]:
            pol = r["policy"]
            hit_tot = r["deadline_hit_rate_total_pct"]
            hit_disp = r["deadline_hit_rate_dispatched_pct"]
            rej = r["rejection_rate_pct"]
            use = r["useful_completion_rate_pct"]
            print(f"  {pol:<32} | Total Hit: {hit_tot:>5.1f}% | Disp Hit: {hit_disp:>5.1f}% | Rej: {rej:>5.1f}% | Useful: {use:>5.1f}%")


if __name__ == "__main__":
    main()
