"""Generate publication-quality visualizations for Phase 10C.

Produces:
  1. candidate_latency_comparison.png: Bar chart of Median, p90, p95, p99 latencies for the 3 candidates.
  2. quality_floor_sweep.png: Hit rate, compliance rate, and mean mIoU across tau_miou thresholds.
  3. accuracy_deadline_tradeoff.png: Scatter/bar representation of candidate mIoU vs measured p95 latency.
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase10c_adaptive_scheduler"
BENCHMARK_PATH = OUTPUT_DIR / "benchmark_results.json"
SIMULATION_PATH = OUTPUT_DIR / "simulation_results.json"


def render_candidate_latency_comparison() -> Path:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)["modes"]

    img = Image.new("RGB", (900, 520), color="#ffffff")
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((40, 25), "Phase 10C: Measured GPU Model Latency by Candidate Mode", fill="#1e293b")
    draw.text((40, 50), "Common protocol: 100 randomized rounds on NVIDIA RTX 4060 Laptop GPU (fp32, 224x224)", fill="#64748b")

    candidates = ["full_attention", "moderate_dwconv", "sigma_b_dwconv"]
    labels = ["Full Attention\n(mIoU 0.7430)", "Moderate DWConv\n(mIoU 0.7036)", "Sigma-b DWConv\n(mIoU 0.4801)"]
    metrics = ["median_ms", "p90_ms", "p95_ms", "p99_ms"]
    metric_labels = ["Median", "p90", "p95", "p99"]
    metric_colors = ["#0284c7", "#0d9488", "#f59e0b", "#ef4444"]

    # Chart bounds
    x_start, y_top, x_end, y_bottom = 80, 110, 840, 420
    draw.line([(x_start, y_bottom), (x_end, y_bottom)], fill="#94a3b8", width=2)
    draw.line([(x_start, y_top), (x_start, y_bottom)], fill="#94a3b8", width=2)

    # Gridlines (0 to 25 ms)
    y_max = 25.0
    for val in [0, 5, 10, 15, 20, 25]:
        y_pos = y_bottom - int((val / y_max) * (y_bottom - y_top))
        draw.line([(x_start, y_pos), (x_end, y_pos)], fill="#e2e8f0", width=1)
        draw.text((x_start - 35, y_pos - 8), f"{val}ms", fill="#64748b")

    # Bars
    group_width = (x_end - x_start) / len(candidates)
    bar_width = 32

    for c_idx, cand in enumerate(candidates):
        c_stats = data[cand]
        center_x = x_start + c_idx * group_width + group_width / 2

        for m_idx, (m_key, m_col) in enumerate(zip(metrics, metric_colors)):
            val = c_stats[m_key]
            bar_h = int((val / y_max) * (y_bottom - y_top))
            bx0 = center_x - 2 * bar_width + m_idx * bar_width + 4
            bx1 = bx0 + bar_width - 8
            by0 = y_bottom - bar_h
            by1 = y_bottom
            draw.rectangle([bx0, by0, bx1, by1], fill=m_col)
            draw.text((bx0 - 2, by0 - 15), f"{val:.1f}", fill="#334155")

        # Candidate label
        draw.text((center_x - 55, y_bottom + 12), labels[c_idx], fill="#1e293b")

    # Legend
    leg_x = 350
    for m_idx, (m_label, m_col) in enumerate(zip(metric_labels, metric_colors)):
        draw.rectangle([leg_x + m_idx * 110, 75, leg_x + m_idx * 110 + 15, 90], fill=m_col)
        draw.text((leg_x + m_idx * 110 + 20, 75), m_label, fill="#334155")

    out_path = OUTPUT_DIR / "candidate_latency_comparison.png"
    img.save(out_path)
    return out_path


def render_quality_floor_sweep() -> Path:
    with open(SIMULATION_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)["default_mixture"]["accuracy_constrained"]

    img = Image.new("RGB", (900, 520), color="#ffffff")
    draw = ImageDraw.Draw(img)

    draw.text((40, 25), "Quality Floor Sweep: Deadline Hit Rate vs Compliance Rate", fill="#1e293b")
    draw.text((40, 50), "Evaluation of Accuracy-Constrained Scheduler across tau_miou thresholds (default_mixture)", fill="#64748b")

    floors = [0.45, 0.65, 0.68, 0.70, 0.73]
    x_start, y_top, x_end, y_bottom = 80, 110, 840, 420
    draw.line([(x_start, y_bottom), (x_end, y_bottom)], fill="#94a3b8", width=2)
    draw.line([(x_start, y_top), (x_start, y_bottom)], fill="#94a3b8", width=2)

    # Gridlines (0% to 100%)
    for pct in [0, 20, 40, 60, 80, 100]:
        y_pos = y_bottom - int((pct / 100.0) * (y_bottom - y_top))
        draw.line([(x_start, y_pos), (x_end, y_pos)], fill="#e2e8f0", width=1)
        draw.text((x_start - 40, y_pos - 8), f"{pct}%", fill="#64748b")

    col_hit = "#0284c7"
    col_comp_hq = "#10b981"
    col_comp_rej = "#ef4444"

    group_w = (x_end - x_start) / len(floors)
    bw = 28

    for idx, tau in enumerate(floors):
        center_x = x_start + idx * group_w + group_w / 2
        entry_hq = data[f"tau_{tau:.2f}_highest_quality_miss"]
        entry_rej = data[f"tau_{tau:.2f}_reject"]

        # Hit rate
        hr = entry_hq["hit_rate_pct"]
        h_h = int((hr / 100.0) * (y_bottom - y_top))
        draw.rectangle([center_x - bw * 1.5, y_bottom - h_h, center_x - bw * 0.5 - 4, y_bottom], fill=col_hit)
        draw.text((center_x - bw * 1.5, y_bottom - h_h - 16), f"{hr:.0f}%", fill="#0284c7")

        # Compliance (HQ fallback)
        cq = entry_hq["quality_compliance_pct"]
        c_h = int((cq / 100.0) * (y_bottom - y_top))
        draw.rectangle([center_x - bw * 0.5, y_bottom - c_h, center_x + bw * 0.5 - 4, y_bottom], fill=col_comp_hq)
        draw.text((center_x - bw * 0.5, y_bottom - c_h - 16), f"{cq:.0f}%", fill="#10b981")

        # Compliance (Reject fallback)
        cr = entry_rej["quality_compliance_pct"]
        cr_h = int((cr / 100.0) * (y_bottom - y_top))
        draw.rectangle([center_x + bw * 0.5, y_bottom - cr_h, center_x + bw * 1.5 - 4, y_bottom], fill=col_comp_rej)
        draw.text((center_x + bw * 0.5, y_bottom - cr_h - 16), f"{cr:.0f}%", fill="#ef4444")

        draw.text((center_x - 22, y_bottom + 12), f"tau={tau:.2f}", fill="#1e293b")

    # Legend
    draw.rectangle([340, 75, 355, 90], fill=col_hit)
    draw.text((360, 75), "Hit Rate (70%)", fill="#334155")
    draw.rectangle([500, 75, 515, 90], fill=col_comp_hq)
    draw.text((520, 75), "Compliance (HQ miss)", fill="#334155")
    draw.rectangle([680, 75, 695, 90], fill=col_comp_rej)
    draw.text((700, 75), "Compliance (Reject)", fill="#334155")

    out_path = OUTPUT_DIR / "quality_floor_sweep.png"
    img.save(out_path)
    return out_path


def render_accuracy_deadline_tradeoff() -> Path:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        modes = json.load(f)["modes"]

    img = Image.new("RGB", (900, 520), color="#ffffff")
    draw = ImageDraw.Draw(img)

    draw.text((40, 25), "Pareto Frontier: Segmentation Quality vs. GPU Latency Bound", fill="#1e293b")
    draw.text((40, 50), "Empirical validation on RTX 4060 Laptop GPU: Full Attention strictly dominates in both quality & latency", fill="#64748b")

    x_start, y_top, x_end, y_bottom = 100, 110, 820, 420
    draw.line([(x_start, y_bottom), (x_end, y_bottom)], fill="#94a3b8", width=2)
    draw.line([(x_start, y_top), (x_start, y_bottom)], fill="#94a3b8", width=2)

    # X axis: Latency bound (14.0 to 18.0 ms)
    # Y axis: mIoU (0.40 to 0.80)
    for lat in [14.0, 15.0, 16.0, 17.0, 18.0]:
        x_pos = x_start + int(((lat - 14.0) / 4.0) * (x_end - x_start))
        draw.line([(x_pos, y_top), (x_pos, y_bottom)], fill="#e2e8f0", width=1)
        draw.text((x_pos - 15, y_bottom + 8), f"{lat:.1f}ms", fill="#64748b")
    draw.text(((x_start + x_end) // 2 - 40, y_bottom + 30), "p95 Latency Bound (ms)", fill="#1e293b")

    for miou in [0.40, 0.50, 0.60, 0.70, 0.80]:
        y_pos = y_bottom - int(((miou - 0.40) / 0.40) * (y_bottom - y_top))
        draw.line([(x_start, y_pos), (x_end, y_pos)], fill="#e2e8f0", width=1)
        draw.text((x_start - 45, y_pos - 8), f"{miou:.2f}", fill="#64748b")
    draw.text((x_start - 90, (y_top + y_bottom) // 2 - 10), "mIoU", fill="#1e293b")

    candidates_info = [
        ("full_attention", "#0284c7", "Full Attention (Phase 4)\n[mIoU=0.7430, p95=15.65ms]\n*Strictly Dominates*"),
        ("moderate_dwconv", "#0d9488", "Moderate DWConv (Phase 6B)\n[mIoU=0.7036, p95=16.41ms]"),
        ("sigma_b_dwconv", "#e11d48", "Sigma-b DWConv (Phase 10B)\n[mIoU=0.4801, p95=17.14ms]"),
    ]

    for name, col, note in candidates_info:
        cand = modes[name]
        p95 = cand["p95_ms"]
        miou = cand["validation_miou"]

        cx = x_start + int(((p95 - 14.0) / 4.0) * (x_end - x_start))
        cy = y_bottom - int(((miou - 0.40) / 0.40) * (y_bottom - y_top))

        r = 10
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline="#0f172a", width=2)
        draw.text((cx + 15, cy - 20), note, fill=col)

    out_path = OUTPUT_DIR / "accuracy_deadline_tradeoff.png"
    img.save(out_path)
    return out_path


def main() -> None:
    p1 = render_candidate_latency_comparison()
    p2 = render_quality_floor_sweep()
    p3 = render_accuracy_deadline_tradeoff()
    print(f"Generated plot: {p1}")
    print(f"Generated plot: {p2}")
    print(f"Generated plot: {p3}")


if __name__ == "__main__":
    main()
