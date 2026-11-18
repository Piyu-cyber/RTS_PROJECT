"""Plotting module for Phase 9.2 Statistical Validation and GPU Diagnosis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from realtime.simulate_strategies import (
    _render_bar_chart_png,
    _render_bar_chart_svg,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "phase9_2_analysis"


def generate_phase9_2_plots(
    paired_stats: dict[str, Any],
    gpu_profile: dict[str, Any],
    output_dir: Path = ANALYSIS_DIR,
) -> list[Path]:
    """Generate publication-quality SVG and PNG charts for Phase 9.2."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    # 1. Steady-state forward pass latency comparison
    modes = ["full_attention", "moderate_dwconv", "aggressive_dwconv"]
    mode_labels = ["Full Attn", "Mod DWConv", "Agg DWConv"]
    med_lats = [gpu_profile["steady_state_latency_ms"][m]["median_ms"] for m in modes]
    colors_hex = ["#0284c7", "#0d9488", "#e11d48"]
    colors_rgb = [(2, 132, 199), (13, 148, 136), (225, 29, 72)]

    svg_lat = _render_bar_chart_svg(
        title="Steady-State GPU Forward Pass Latency (100 Iterations)",
        categories=mode_labels,
        values=med_lats,
        y_label="Median GPU Latency (ms)",
        y_max=round(max(med_lats) * 1.3, 1),
        colors=colors_hex,
    )
    p_lat_svg = output_dir / "gpu_steady_state_latency.svg"
    p_lat_svg.write_text(svg_lat, encoding="utf-8")
    created.append(p_lat_svg)

    p_lat_png = output_dir / "gpu_steady_state_latency.png"
    _render_bar_chart_png(
        title="Steady-State GPU Forward Pass Latency (100 Iterations)",
        categories=mode_labels,
        values=med_lats,
        y_label="Median GPU Latency (ms)",
        y_max=round(max(med_lats) * 1.3, 1),
        colors_rgb=colors_rgb,
        output_path=p_lat_png,
    )
    created.append(p_lat_png)

    # 2. Paired Hit Rate Differences (Delta %)
    # Comparison: Adaptive vs Full Attention across metrics
    metrics = ["p95", "p99", "max_observed"]
    metric_labels = ["p95 Bound", "p99 Bound", "max_observed"]
    diff_vals = [paired_stats[m]["adaptive_controller_vs_always_full_attention"]["hit_rate_difference"]["mean"] * 100 for m in metrics]
    
    # Delta bar chart
    svg_diff = _render_bar_chart_svg(
        title="Paired Hit Rate Delta: Adaptive vs Full Attention (%)",
        categories=metric_labels,
        values=[abs(v) for v in diff_vals],  # Magnitude for bar chart
        y_label="|Hit Rate Delta| (%) (Full Attn Higher)",
        y_max=15.0,
        colors=["#f59e0b", "#ef4444", "#f97316"],
    )
    p_diff_svg = output_dir / "paired_hit_rate_delta.svg"
    p_diff_svg.write_text(svg_diff, encoding="utf-8")
    created.append(p_diff_svg)

    p_diff_png = output_dir / "paired_hit_rate_delta.png"
    _render_bar_chart_png(
        title="Paired Hit Rate Delta: Adaptive vs Full Attention (%)",
        categories=metric_labels,
        values=[abs(v) for v in diff_vals],
        y_label="|Hit Rate Delta| (%) (Full Attn Higher)",
        y_max=15.0,
        colors_rgb=[(245, 158, 11), (239, 68, 68), (249, 115, 22)],
        output_path=p_diff_png,
    )
    created.append(p_diff_png)

    return created


if __name__ == "__main__":
    with open(ANALYSIS_DIR / "paired_statistics.json") as f:
        ps = json.load(f)
    with open(ANALYSIS_DIR / "gpu_profile_summary.json") as f:
        gp = json.load(f)
    plots = generate_phase9_2_plots(ps, gp, ANALYSIS_DIR)
    print(f"Generated {len(plots)} diagnostic plots in {ANALYSIS_DIR}")
