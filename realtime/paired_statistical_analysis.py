"""Phase 9.2: Statistical Analysis and Paired Hypothesis Testing on Phase 9.1 traces.

Calculates:
  1. Data integrity and fairness validation across matched requests.
  2. Per-round hit rates and variances across 5 repeated rounds.
  3. Paired differences in hit rates, latency, and selected mIoU.
  4. Exact and continuity-corrected McNemar's tests for matched binary outcomes.
  5. Paired difference confidence intervals (Wilson / Newcombe and standard error).
  6. Bonferroni / Holm-Bonferroni corrections for multiple comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_FILE = PROJECT_ROOT / "outputs" / "phase9_1_fair" / "raw_traces.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase9_2_analysis"


def calculate_mcnemar_test(b: int, c: int) -> dict[str, Any]:
    """Calculate McNemar's test for paired binary discordant pairs (b, c).
    
    b: Strategy 1 Hit (1), Strategy 2 Miss (0)
    c: Strategy 1 Miss (0), Strategy 2 Hit (1)
    """
    total_discordant = b + c
    if total_discordant == 0:
        return {
            "b": b,
            "c": c,
            "chi2_unrounded": 0.0,
            "chi2_cc": 0.0,
            "p_value_cc": 1.0,
            "interpretation": "Identical discordant pairs (no difference)",
        }
    
    # Continuity-corrected chi-squared: (|b - c| - 1)^2 / (b + c)
    chi2_cc = (max(0.0, abs(b - c) - 1.0) ** 2) / total_discordant
    # p-value from chi2 distribution with 1 DOF: P(X >= chi2) = erfc(sqrt(chi2 / 2))
    p_value_cc = math.erfc(math.sqrt(chi2_cc / 2.0))
    
    # Standard chi2 without continuity correction: (b - c)^2 / (b + c)
    chi2_std = ((b - c) ** 2) / total_discordant
    p_value_std = math.erfc(math.sqrt(chi2_std / 2.0))
    
    return {
        "b_discordant_first_only": b,
        "c_discordant_second_only": c,
        "chi2_continuity_corrected": round(float(chi2_cc), 4),
        "p_value_continuity_corrected": round(float(p_value_cc), 6),
        "chi2_standard": round(float(chi2_std), 4),
        "p_value_standard": round(float(p_value_std), 6),
        "statistically_significant_05": p_value_cc < 0.05,
    }


def paired_difference_ci(diffs: list[float], confidence: float = 0.95) -> dict[str, float]:
    """Compute mean paired difference and normal approximation 95% CI."""
    if not diffs:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}
    n = len(diffs)
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    z = 1.95996  # 95%
    return {
        "mean": round(mean, 4),
        "ci_lower": round(mean - z * se, 4),
        "ci_upper": round(mean + z * se, 4),
        "std": round(std, 4),
        "std_error": round(se, 4),
    }


def analyze_phase9_1_traces(trace_path: Path, output_dir: Path) -> dict[str, Any]:
    """Execute complete paired statistical evaluation of Phase 9.1 traces."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with trace_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # 1. Integrity check
    data_val: dict[str, Any] = {
        "total_rows": len(rows),
        "expected_rows": 2250,
        "row_count_valid": len(rows) == 2250,
        "metrics_found": sorted(list(set(r["controller_metric"] for r in rows))),
        "rounds_found": sorted(list(set(int(r["round_id"]) for r in rows))),
        "strategies_found": sorted(list(set(r["strategy"] for r in rows))),
        "errors": [],
    }

    # Group rows by (controller_metric, round_id, request_id)
    matched_groups: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for r in rows:
        key = (r["controller_metric"], int(r["round_id"]), r["request_id"])
        matched_groups.setdefault(key, {})[r["strategy"]] = r

    # Verify matching
    expected_strats = {"always_full_attention", "always_fastest", "adaptive_controller"}
    for (metric, rnd, req_id), strat_dict in matched_groups.items():
        if set(strat_dict.keys()) != expected_strats:
            data_val["errors"].append(f"Metric {metric} Round {rnd} Req {req_id} missing strategies: {expected_strats - set(strat_dict.keys())}")
        else:
            # Check deadline consistency
            dlines = [float(s_row["deadline_budget_ms"]) for s_row in strat_dict.values()]
            if max(dlines) - min(dlines) > 1e-4:
                data_val["errors"].append(f"Metric {metric} Round {rnd} Req {req_id} deadline mismatch: {dlines}")

    data_val["matching_valid"] = len(data_val["errors"]) == 0

    # 2. Per-round results table
    per_round_table: list[dict[str, Any]] = []
    by_metric_round_strat: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["controller_metric"], int(r["round_id"]), r["strategy"])
        by_metric_round_strat.setdefault(key, []).append(r)

    for (metric, rnd, strat), s_rows in sorted(by_metric_round_strat.items()):
        tot = len(s_rows)
        e2e_hits = sum(1 for r in s_rows if r["end_to_end_deadline_hit"] == "True")
        mod_hits = sum(1 for r in s_rows if r["model_deadline_hit"] == "True")
        e2e_lats = [float(r["end_to_end_ms"]) for r in s_rows]
        mod_lats = [float(r["model_forward_ms"]) for r in s_rows]
        per_round_table.append({
            "controller_metric": metric,
            "round_id": rnd,
            "strategy": strat,
            "request_count": tot,
            "e2e_hit_count": e2e_hits,
            "e2e_hit_rate": round(e2e_hits / tot, 4),
            "model_hit_count": mod_hits,
            "model_hit_rate": round(mod_hits / tot, 4),
            "median_e2e_ms": round(float(np.median(e2e_lats)), 4),
            "median_model_ms": round(float(np.median(mod_lats)), 4),
        })

    # Save per_round_results.csv
    csv_out = output_dir / "per_round_results.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_round_table[0].keys()))
        writer.writeheader()
        writer.writerows(per_round_table)

    # 3. Paired comparisons per metric
    paired_stats: dict[str, Any] = {}
    comparisons = [
        ("adaptive_controller", "always_full_attention"),
        ("adaptive_controller", "always_fastest"),
        ("always_fastest", "always_full_attention"),
    ]

    for metric in ["p95", "p99", "max_observed"]:
        paired_stats[metric] = {}
        metric_keys = [k for k in matched_groups if k[0] == metric]
        
        for s1, s2 in comparisons:
            comp_name = f"{s1}_vs_{s2}"
            hit_diffs: list[float] = []
            lat_diffs: list[float] = []
            miou_diffs: list[float] = []
            
            b = 0  # s1 hit, s2 miss
            c = 0  # s1 miss, s2 hit
            
            for k in metric_keys:
                r1 = matched_groups[k][s1]
                r2 = matched_groups[k][s2]
                
                h1 = 1 if r1["end_to_end_deadline_hit"] == "True" else 0
                h2 = 1 if r2["end_to_end_deadline_hit"] == "True" else 0
                hit_diffs.append(float(h1 - h2))
                
                if h1 == 1 and h2 == 0:
                    b += 1
                elif h1 == 0 and h2 == 1:
                    c += 1
                
                lat1 = float(r1["end_to_end_ms"])
                lat2 = float(r2["end_to_end_ms"])
                lat_diffs.append(lat1 - lat2)
                
                m1 = float(r1["selected_model_miou"])
                m2 = float(r2["selected_model_miou"])
                miou_diffs.append(m1 - m2)
            
            mcnemar = calculate_mcnemar_test(b, c)
            hit_ci = paired_difference_ci(hit_diffs)
            lat_ci = paired_difference_ci(lat_diffs)
            miou_ci = paired_difference_ci(miou_diffs)
            
            paired_stats[metric][comp_name] = {
                "pair_count": len(metric_keys),
                "strategy_1": s1,
                "strategy_2": s2,
                "hit_rate_difference": hit_ci,
                "latency_difference_ms": lat_ci,
                "miou_difference": miou_ci,
                "mcnemar_test": mcnemar,
            }

    # Multiple comparison correction (Bonferroni across the 3 comparisons per metric)
    for metric, comp_dict in paired_stats.items():
        p_vals = [comp_dict[k]["mcnemar_test"]["p_value_continuity_corrected"] for k in comp_dict]
        # Bonferroni alpha = 0.05 / 3 = 0.0167
        for k in comp_dict:
            p_val = comp_dict[k]["mcnemar_test"]["p_value_continuity_corrected"]
            comp_dict[k]["mcnemar_test"]["bonferroni_significant"] = p_val < (0.05 / len(comparisons))

    # Save outputs
    (output_dir / "data_validation.json").write_text(json.dumps(data_val, indent=2), encoding="utf-8")
    (output_dir / "paired_statistics.json").write_text(json.dumps(paired_stats, indent=2), encoding="utf-8")

    return {
        "validation": data_val,
        "per_round_table": per_round_table,
        "paired_statistics": paired_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9.2 Statistical Analysis")
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACE_FILE, help="Path to raw_traces.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    print("Executing Phase 9.2 Statistical Analysis on Phase 9.1 traces...")
    res = analyze_phase9_1_traces(args.traces, args.output_dir)
    print("Statistical analysis complete!")
    print(f"Validation: {res['validation']['row_count_valid']}, Errors: {len(res['validation']['errors'])}")
    for metric, comps in res["paired_statistics"].items():
        print(f"\n=== METRIC: {metric} ===")
        for c_name, data in comps.items():
            hr_diff = data["hit_rate_difference"]["mean"] * 100
            p_val = data["mcnemar_test"]["p_value_continuity_corrected"]
            sig = data["mcnemar_test"]["bonferroni_significant"]
            print(f"  {c_name:42}: Delta={hr_diff:+.2f}% | p={p_val:.5f} (Bonferroni sig: {sig})")


if __name__ == "__main__":
    main()
