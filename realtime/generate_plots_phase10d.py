"""Generate Phase 10D visualization plots using Pillow.

Produces:
  1. latency_stability_rounds.png: Per-round and pooled latency comparison across candidates.
  2. paired_differences_distribution.png: Paired latency difference distributions with 95% bootstrap CIs.
  3. prediction_calibration_roc.png: Precision and false-decision rates across p95 vs p99 quantiles.
  4. useful_completion_tradeoff.png: Useful completion rate vs rejection rate across quality floors.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase10d_realtime_validation"
LATENCY_SUM_PATH = OUTPUT_DIR / "latency_summary.json"
PAIRED_PATH = OUTPUT_DIR / "paired_latency_analysis.json"
CALIB_PATH = OUTPUT_DIR / "deadline_calibration.csv"
FALLBACK_PATH = OUTPUT_DIR / "fallback_comparison.csv"


def render_latency_stability_rounds() -> Path:
    with open(LATENCY_SUM_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    img = Image.new("RGB", (900, 520), color="#ffffff")
    draw = ImageDraw.Draw(img)

    draw.text((40, 25), "Phase 10D: Multi-Round Latency Stability Analysis", fill="#1e293b")
    draw.text((40, 50), "5 independent rounds x 30 randomized trials per candidate on RTX 4060 Laptop GPU", fill="#64748b")

    x_start, y_top, x_end, y_bottom = 80, 110, 840, 420
    draw.line([(x_start, y_bottom), (x_end, y_bottom)], fill="#94a3b8", width=2)
    draw.line([(x_start, y_top), (x_start, y_bottom)], fill="#94a3b8", width=2)

    y_max = 25.0
    for val in [0, 5, 10, 15, 20, 25]:
        y_pos = y_bottom - int((val / y_max) * (y_bottom - y_top))
        draw.line([(x_start, y_pos), (x_end, y_pos)], fill="#e2e8f0", width=1)
        draw.text((x_start - 35, y_pos - 8), f"{val}ms", fill="#64748b")

    rounds = [f"round_{i}" for i in range(1, 6)] + ["Pooled"]
    candidates = ["full_attention", "moderate_dwconv", "sigma_b_dwconv"]
    colors = ["#0284c7", "#0d9488", "#e11d48"]

    group_w = (x_end - x_start) / len(rounds)
    bw = 18

    for r_idx, r_name in enumerate(rounds):
        cx = x_start + r_idx * group_w + group_w / 2
        for c_idx, c_name in enumerate(candidates):
            if r_name == "Pooled":
                med = data["pooled"][c_name]["model_host_ms"]["median"]
            else:
                med = data["per_round"][c_name][r_name]["model_host_ms"]["median"]

            bar_h = int((med / y_max) * (y_bottom - y_top))
            bx0 = cx - 1.5 * bw + c_idx * bw
            bx1 = bx0 + bw - 3
            draw.rectangle([bx0, y_bottom - bar_h, bx1, y_bottom], fill=colors[c_idx])
            draw.text((bx0 - 2, y_bottom - bar_h - 14), f"{med:.1f}", fill="#334155")

        lbl = "Round " + str(r_idx + 1) if r_name != "Pooled" else "Pooled (150)"
        draw.text((cx - 28, y_bottom + 12), lbl, fill="#1e293b")

    # Legend
    leg_x = 300
    leg_labels = ["Full Attention", "Moderate DWConv", "Sigma-b DWConv"]
    for idx, (lbl, col) in enumerate(zip(leg_labels, colors)):
        draw.rectangle([leg_x + idx * 170, 75, leg_x + idx * 170 + 15, 90], fill=col)
        draw.text((leg_x + idx * 170 + 20, 75), lbl, fill="#334155")

    out_p = OUTPUT_DIR / "latency_stability_rounds.png"
    img.save(out_p)
    return out_p


def render_prediction_calibration_roc() -> Path:
    rows = []
    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["workload"] == "default_mixture" and r["stratum"] == "overall" and r["tau_miou"] == "0.7":
                rows.append(r)

    img = Image.new("RGB", (900, 520), color="#ffffff")
    draw = ImageDraw.Draw(img)

    draw.text((40, 25), "Deadline Feasibility Prediction Accuracy: p95 vs p99 Bounds", fill="#1e293b")
    draw.text((40, 50), "Precision, False-Feasible (Over-optimistic), and False-Infeasible (Over-conservative) Rates", fill="#64748b")

    x_start, y_top, x_end, y_bottom = 80, 110, 840, 420
    draw.line([(x_start, y_bottom), (x_end, y_bottom)], fill="#94a3b8", width=2)
    draw.line([(x_start, y_top), (x_start, y_bottom)], fill="#94a3b8", width=2)

    for val in [0, 20, 40, 60, 80, 100]:
        y_pos = y_bottom - int((val / 100.0) * (y_bottom - y_top))
        draw.line([(x_start, y_pos), (x_end, y_pos)], fill="#e2e8f0", width=1)
        draw.text((x_start - 35, y_pos - 8), f"{val}%", fill="#64748b")

    # Group by (metric, fallback_policy)
    group_w = (x_end - x_start) / len(rows)
    bw = 24
    c_prec = "#0284c7"
    c_false_feas = "#ef4444"
    c_false_infeas = "#f59e0b"

    for idx, r in enumerate(rows):
        cx = x_start + idx * group_w + group_w / 2
        prec = float(r["precision_predicted_feasible_pct"])
        ff = float(r["false_feasible_count"])  # out of 100
        fi = float(r["false_infeasible_count"])  # out of 100

        h_p = int((prec / 100.0) * (y_bottom - y_top))
        h_ff = int((ff / 100.0) * (y_bottom - y_top))
        h_fi = int((fi / 100.0) * (y_bottom - y_top))

        draw.rectangle([cx - 1.5 * bw, y_bottom - h_p, cx - 0.5 * bw - 2, y_bottom], fill=c_prec)
        draw.text((cx - 1.5 * bw, y_bottom - h_p - 15), f"{prec:.0f}%", fill=c_prec)

        draw.rectangle([cx - 0.5 * bw, y_bottom - h_ff, cx + 0.5 * bw - 2, y_bottom], fill=c_false_feas)
        draw.text((cx - 0.5 * bw, y_bottom - h_ff - 15), f"{ff:.0f}", fill=c_false_feas)

        draw.rectangle([cx + 0.5 * bw, y_bottom - h_fi, cx + 1.5 * bw - 2, y_bottom], fill=c_false_infeas)
        draw.text((cx + 0.5 * bw, y_bottom - h_fi - 15), f"{fi:.0f}", fill=c_false_infeas)

        lbl = f"{r['metric'].upper()}\n({r['fallback_policy'][:7]})"
        draw.text((cx - 25, y_bottom + 12), lbl, fill="#1e293b")

    # Legend
    draw.rectangle([300, 75, 315, 90], fill=c_prec)
    draw.text((320, 75), "Precision (Hits / Predicted Feasible)", fill="#334155")
    draw.rectangle([560, 75, 575, 90], fill=c_false_feas)
    draw.text((580, 75), "False Feasible (Count)", fill="#334155")
    draw.rectangle([740, 75, 755, 90], fill=c_false_infeas)
    draw.text((760, 75), "False Infeasible", fill="#334155")

    out_p = OUTPUT_DIR / "prediction_calibration_roc.png"
    img.save(out_p)
    return out_p


def render_useful_completion_tradeoff() -> Path:
    rows = []
    with open(FALLBACK_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["workload"] == "default_mixture" and "accuracy_constrained" in r["policy"]:
                rows.append(r)

    img = Image.new("RGB", (900, 520), color="#ffffff")
    draw = ImageDraw.Draw(img)

    draw.text((40, 25), "Useful Completion vs. Rejection Trade-off across Quality Floors", fill="#1e293b")
    draw.text((40, 50), "Comparing 'reject' policy (preserves GPU resources) vs 'highest_quality_miss' (dispatches despite miss)", fill="#64748b")

    x_start, y_top, x_end, y_bottom = 80, 110, 840, 420
    draw.line([(x_start, y_bottom), (x_end, y_bottom)], fill="#94a3b8", width=2)
    draw.line([(x_start, y_top), (x_start, y_bottom)], fill="#94a3b8", width=2)

    for val in [0, 20, 40, 60, 80, 100]:
        y_pos = y_bottom - int((val / 100.0) * (y_bottom - y_top))
        draw.line([(x_start, y_pos), (x_end, y_pos)], fill="#e2e8f0", width=1)
        draw.text((x_start - 35, y_pos - 8), f"{val}%", fill="#64748b")

    floors = ["0.45", "0.65", "0.68", "0.7", "0.73"]
    group_w = (x_end - x_start) / len(floors)
    bw = 24

    c_use_hq = "#0284c7"
    c_use_rej = "#10b981"
    c_rej_count = "#ef4444"

    for idx, tau in enumerate(floors):
        cx = x_start + idx * group_w + group_w / 2
        r_hq = next(r for r in rows if r["tau_miou"] == tau and r["fallback_policy"] == "highest_quality_miss")
        r_rej = next(r for r in rows if r["tau_miou"] == tau and r["fallback_policy"] == "reject")

        use_hq = float(r_hq["useful_completion_rate_pct"])
        use_rej = float(r_rej["useful_completion_rate_pct"])
        rej_rate = float(r_rej["rejection_rate_pct"])

        h_hq = int((use_hq / 100.0) * (y_bottom - y_top))
        h_rej = int((use_rej / 100.0) * (y_bottom - y_top))
        h_rr = int((rej_rate / 100.0) * (y_bottom - y_top))

        draw.rectangle([cx - 1.5 * bw, y_bottom - h_hq, cx - 0.5 * bw - 2, y_bottom], fill=c_use_hq)
        draw.text((cx - 1.5 * bw, y_bottom - h_hq - 15), f"{use_hq:.0f}%", fill=c_use_hq)

        draw.rectangle([cx - 0.5 * bw, y_bottom - h_rej, cx + 0.5 * bw - 2, y_bottom], fill=c_use_rej)
        draw.text((cx - 0.5 * bw, y_bottom - h_rej - 15), f"{use_rej:.0f}%", fill=c_use_rej)

        draw.rectangle([cx + 0.5 * bw, y_bottom - h_rr, cx + 1.5 * bw - 2, y_bottom], fill=c_rej_count)
        draw.text((cx + 0.5 * bw, y_bottom - h_rr - 15), f"{rej_rate:.0f}%", fill=c_rej_count)

        draw.text((cx - 20, y_bottom + 12), f"tau={tau}", fill="#1e293b")

    # Legend
    draw.rectangle([250, 75, 265, 90], fill=c_use_hq)
    draw.text((270, 75), "Useful Completions (HQ miss)", fill="#334155")
    draw.rectangle([480, 75, 495, 90], fill=c_use_rej)
    draw.text((500, 75), "Useful Completions (Reject)", fill="#334155")
    draw.rectangle([700, 75, 715, 90], fill=c_rej_count)
    draw.text((720, 75), "Rejection Rate", fill="#334155")

    out_p = OUTPUT_DIR / "useful_completion_tradeoff.png"
    img.save(out_p)
    return out_p


def main() -> None:
    p1 = render_latency_stability_rounds()
    p2 = render_prediction_calibration_roc()
    p3 = render_useful_completion_tradeoff()
    print(f"Generated plot: {p1}")
    print(f"Generated plot: {p2}")
    print(f"Generated plot: {p3}")


if __name__ == "__main__":
    main()
