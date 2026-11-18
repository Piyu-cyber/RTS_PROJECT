"""Simulation and comparative evaluation of deadline-scheduling strategies.

Compares three strategies on identical generated synthetic deadline streams:
  1. always_full_attention
  2. always_fastest
  3. adaptive_controller (Phase 7 controller)

The simulation is offline and deterministic: it uses empirical latency bounds
and validation mIoU from the Phase 6B benchmark without running neural network
inference or retraining models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from realtime.controller import (
    SUPPORTED_METRICS,
    ControllerError,
    ModeProfile,
    SelectionResult,
    load_benchmark_modes,
    select_mode,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "outputs" / "inference_benchmark" / "phase6_finetuned_results.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase8_simulation"

# ---------------------------------------------------------------------------
# Workload Configurations
# ---------------------------------------------------------------------------
# Measured Phase 6B statistics for orientation:
#   full_attention:    p95=16.619 ms, p99=21.334 ms, max=27.563 ms
#   moderate_dwconv:   p95=18.650 ms, p99=20.828 ms, max=23.999 ms
#   aggressive_dwconv: p95=18.061 ms, p99=23.270 ms, max=25.654 ms
#
# Default mixture partitions the space around these empirical boundaries:
#   tight:    [10.0, 17.0] ms — stresses modes near or below full_attention p95
#   moderate: [17.0, 22.0] ms — spans differences between p95 and p99 thresholds
#   relaxed:  [22.0, 32.0] ms — permits all or most modes under p95/p99/max
WORKLOAD_CONFIGS: dict[str, dict[str, Any]] = {
    "default_mixture": {
        "name": "default_mixture",
        "description": (
            "Synthetic mixture informed by Phase 6B latency statistics "
            "(30% tight [10, 17ms], 40% moderate [17, 22ms], 30% relaxed [22, 32ms])."
        ),
        "categories": {
            "tight": {"range_ms": (10.0, 17.0), "weight": 0.30},
            "moderate": {"range_ms": (17.0, 22.0), "weight": 0.40},
            "relaxed": {"range_ms": (22.0, 32.0), "weight": 0.30},
        },
    },
    "stress_tight": {
        "name": "stress_tight",
        "description": (
            "Stress workload with predominantly tight deadlines "
            "(60% tight [10, 17ms], 30% moderate [17, 22ms], 10% relaxed [22, 32ms])."
        ),
        "categories": {
            "tight": {"range_ms": (10.0, 17.0), "weight": 0.60},
            "moderate": {"range_ms": (17.0, 22.0), "weight": 0.30},
            "relaxed": {"range_ms": (22.0, 32.0), "weight": 0.10},
        },
    },
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticRequest:
    """A single synthetic inference request in the evaluation stream."""

    request_id: str
    deadline_ms: float
    workload_category: str


@dataclass(frozen=True)
class RequestDecision:
    """Decision made by a strategy for a single request."""

    request_id: str
    deadline_ms: float
    workload_category: str
    strategy: str
    selected_mode: str
    selected_miou: float
    selected_latency_bound_ms: float
    empirical_hit: bool
    feasible_modes: list[str]
    fallback_invoked: bool


# ---------------------------------------------------------------------------
# Deadline Stream Generator
# ---------------------------------------------------------------------------
def generate_deadline_stream(
    n_requests: int,
    seed: int,
    workload_name: str = "default_mixture",
) -> list[SyntheticRequest]:
    """Generate a reproducible stream of synthetic deadline requests.

    Validation rules:
      - n_requests must be a positive integer.
      - seed must be an integer.
      - workload_name must be a known configuration in WORKLOAD_CONFIGS.
    """
    if not isinstance(n_requests, int) or n_requests <= 0:
        raise ControllerError(
            f"n_requests must be a positive integer, got {n_requests!r}"
        )
    if not isinstance(seed, int):
        raise ControllerError(f"seed must be an integer, got {seed!r}")
    if workload_name not in WORKLOAD_CONFIGS:
        raise ControllerError(
            f"Unknown workload: {workload_name!r}. "
            f"Supported: {sorted(WORKLOAD_CONFIGS)}"
        )

    config = WORKLOAD_CONFIGS[workload_name]
    categories = config["categories"]
    cat_names = list(categories.keys())
    cat_weights = [categories[c]["weight"] for c in cat_names]

    rng = random.Random(seed)
    stream: list[SyntheticRequest] = []

    for idx in range(1, n_requests + 1):
        chosen_cat = rng.choices(cat_names, weights=cat_weights, k=1)[0]
        range_min, range_max = categories[chosen_cat]["range_ms"]
        deadline = rng.uniform(range_min, range_max)
        stream.append(
            SyntheticRequest(
                request_id=f"req_{idx:05d}",
                deadline_ms=round(deadline, 4),
                workload_category=chosen_cat,
            )
        )

    return stream


# ---------------------------------------------------------------------------
# Strategy Implementations
# ---------------------------------------------------------------------------
def evaluate_always_full_attention(
    requests: list[SyntheticRequest],
    modes: list[ModeProfile],
    latency_metric: str,
) -> list[RequestDecision]:
    """Strategy A: Always select 'full_attention' regardless of deadline."""
    full_mode = next((m for m in modes if m.name == "full_attention"), None)
    if full_mode is None:
        raise ControllerError(
            "Mode 'full_attention' is required for Strategy A but not present in benchmark."
        )

    bound = full_mode.latency_for_metric(latency_metric)
    miou = full_mode.validation_miou

    decisions: list[RequestDecision] = []
    for req in requests:
        hit = bound <= req.deadline_ms
        decisions.append(
            RequestDecision(
                request_id=req.request_id,
                deadline_ms=req.deadline_ms,
                workload_category=req.workload_category,
                strategy="always_full_attention",
                selected_mode="full_attention",
                selected_miou=miou,
                selected_latency_bound_ms=bound,
                empirical_hit=hit,
                feasible_modes=["full_attention"] if hit else [],
                fallback_invoked=False,
            )
        )
    return decisions


def evaluate_always_fastest(
    requests: list[SyntheticRequest],
    modes: list[ModeProfile],
    latency_metric: str,
) -> list[RequestDecision]:
    """Strategy B: Always select the mode with the lowest configured latency bound."""
    if not modes:
        raise ControllerError("Mode collection must not be empty.")

    # Smallest latency bound, tie-breaking by higher mIoU, then lexicographical name
    fastest_mode = min(
        modes,
        key=lambda m: (
            m.latency_for_metric(latency_metric),
            -m.validation_miou,
            m.name,
        ),
    )
    bound = fastest_mode.latency_for_metric(latency_metric)
    miou = fastest_mode.validation_miou

    decisions: list[RequestDecision] = []
    for req in requests:
        hit = bound <= req.deadline_ms
        decisions.append(
            RequestDecision(
                request_id=req.request_id,
                deadline_ms=req.deadline_ms,
                workload_category=req.workload_category,
                strategy="always_fastest",
                selected_mode=fastest_mode.name,
                selected_miou=miou,
                selected_latency_bound_ms=bound,
                empirical_hit=hit,
                feasible_modes=[fastest_mode.name] if hit else [],
                fallback_invoked=False,
            )
        )
    return decisions


def evaluate_adaptive_controller(
    requests: list[SyntheticRequest],
    modes: list[ModeProfile],
    latency_metric: str,
) -> list[RequestDecision]:
    """Strategy C: Call the authoritative Phase 7 controller for each request."""
    decisions: list[RequestDecision] = []
    for req in requests:
        result = select_mode(req.deadline_ms, modes, latency_metric)
        hit = not result.likely_deadline_miss
        decisions.append(
            RequestDecision(
                request_id=req.request_id,
                deadline_ms=req.deadline_ms,
                workload_category=req.workload_category,
                strategy="adaptive_controller",
                selected_mode=result.selected_mode,
                selected_miou=result.selected_miou,
                selected_latency_bound_ms=result.selected_empirical_latency_bound_ms,
                empirical_hit=hit,
                feasible_modes=result.feasible_modes,
                fallback_invoked=result.likely_deadline_miss,
            )
        )
    return decisions


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------
def compute_strategy_metrics(
    decisions: list[RequestDecision],
    all_mode_names: list[str],
) -> dict[str, Any]:
    """Compute aggregate and per-category metrics from a decision list."""
    total = len(decisions)
    if total == 0:
        raise ControllerError("Cannot compute metrics on empty decision list.")

    hits = sum(1 for d in decisions if d.empirical_hit)
    misses = total - hits
    hit_rate = hits / total
    miss_rate = misses / total
    avg_miou = sum(d.selected_miou for d in decisions) / total
    avg_bound = sum(d.selected_latency_bound_ms for d in decisions) / total
    fallbacks = sum(1 for d in decisions if d.fallback_invoked)

    # Mode frequencies
    counts: dict[str, int] = {m: 0 for m in all_mode_names}
    for d in decisions:
        counts[d.selected_mode] = counts.get(d.selected_mode, 0) + 1

    mode_frequencies: dict[str, dict[str, Any]] = {}
    for m in sorted(counts.keys()):
        c = counts[m]
        mode_frequencies[m] = {
            "count": c,
            "percentage": round((c / total) * 100.0, 2),
        }

    # Per-category metrics
    by_cat: dict[str, list[RequestDecision]] = {}
    for d in decisions:
        by_cat.setdefault(d.workload_category, []).append(d)

    per_category: dict[str, dict[str, Any]] = {}
    for cat_name in sorted(by_cat.keys()):
        cat_decs = by_cat[cat_name]
        cat_total = len(cat_decs)
        cat_hits = sum(1 for d in cat_decs if d.empirical_hit)
        cat_fallbacks = sum(1 for d in cat_decs if d.fallback_invoked)
        per_category[cat_name] = {
            "request_count": cat_total,
            "empirical_hits": cat_hits,
            "empirical_hit_rate": round(cat_hits / cat_total, 6),
            "empirical_miss_rate": round((cat_total - cat_hits) / cat_total, 6),
            "average_selected_miou": round(
                sum(d.selected_miou for d in cat_decs) / cat_total, 6
            ),
            "average_selected_latency_bound_ms": round(
                sum(d.selected_latency_bound_ms for d in cat_decs) / cat_total, 4
            ),
            "fallback_count": cat_fallbacks,
        }

    return {
        "total_requests": total,
        "empirical_hit_count": hits,
        "empirical_miss_count": misses,
        "empirical_deadline_hit_rate": round(hit_rate, 6),
        "empirical_deadline_miss_rate": round(miss_rate, 6),
        "average_selected_miou": round(avg_miou, 6),
        "average_selected_latency_bound_ms": round(avg_bound, 4),
        "adaptive_fallback_count": fallbacks,
        "mode_selection_frequency": mode_frequencies,
        "per_category_metrics": per_category,
    }


# ---------------------------------------------------------------------------
# Plot Generation (Pure Python / Pillow & SVG)
# ---------------------------------------------------------------------------
def _render_bar_chart_svg(
    title: str,
    categories: list[str],
    values: list[float],
    y_label: str,
    y_max: float,
    colors: list[str],
) -> str:
    """Generate a clean SVG bar chart."""
    width, height = 700, 450
    margin_left, margin_right = 100, 50
    margin_top, margin_bottom = 80, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    n_bars = len(categories)
    bar_width = plot_width / (n_bars * 1.6)
    gap = plot_width / n_bars

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#fcfcfc" />',
        f'<text x="{width / 2}" y="45" font-family="Segoe UI, sans-serif" font-size="20" font-weight="bold" text-anchor="middle" fill="#1e293b">{title}</text>',
    ]

    # Gridlines & Y-ticks
    num_ticks = 5
    for i in range(num_ticks + 1):
        tick_val = y_max * (i / num_ticks)
        y_pos = margin_top + plot_height - (plot_height * (i / num_ticks))
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y_pos}" x2="{width - margin_right}" y2="{y_pos}" stroke="#e2e8f0" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{margin_left - 12}" y="{y_pos + 4}" font-family="Segoe UI, sans-serif" font-size="12" text-anchor="end" fill="#64748b">{tick_val:.2f}</text>'
        )

    # Y-axis label
    svg_parts.append(
        f'<text x="25" y="{margin_top + plot_height / 2}" font-family="Segoe UI, sans-serif" font-size="14" font-weight="bold" text-anchor="middle" fill="#334155" transform="rotate(-90 25 {margin_top + plot_height / 2})">{y_label}</text>'
    )

    # Bars
    for idx, (cat, val, col) in enumerate(zip(categories, values, colors)):
        x_pos = margin_left + idx * gap + (gap - bar_width) / 2
        bar_h = (val / y_max) * plot_height if y_max > 0 else 0
        y_pos = margin_top + plot_height - bar_h

        svg_parts.append(
            f'<rect x="{x_pos}" y="{y_pos}" width="{bar_width}" height="{bar_h}" rx="4" fill="{col}" />'
        )
        # Value label on bar
        svg_parts.append(
            f'<text x="{x_pos + bar_width / 2}" y="{y_pos - 8}" font-family="Segoe UI, sans-serif" font-size="12" font-weight="bold" text-anchor="middle" fill="#1e293b">{val:.4f}</text>'
        )
        # Category label under axis
        svg_parts.append(
            f'<text x="{x_pos + bar_width / 2}" y="{margin_top + plot_height + 25}" font-family="Segoe UI, sans-serif" font-size="13" font-weight="500" text-anchor="middle" fill="#334155">{cat}</text>'
        )

    # X-axis
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#94a3b8" stroke-width="1.5" />'
    )
    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _render_bar_chart_png(
    title: str,
    categories: list[str],
    values: list[float],
    y_label: str,
    y_max: float,
    colors_rgb: list[tuple[int, int, int]],
    output_path: Path,
) -> None:
    """Generate a clean standalone PNG bar chart using Pillow."""
    width, height = 700, 450
    margin_left, margin_right = 100, 50
    margin_top, margin_bottom = 80, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), (252, 252, 252))
    draw = ImageDraw.Draw(image)

    # Draw title
    draw.text((width // 2, 35), title, fill=(30, 41, 59), anchor="mm")

    # Gridlines and Y ticks
    num_ticks = 5
    for i in range(num_ticks + 1):
        tick_val = y_max * (i / num_ticks)
        y_pos = int(margin_top + plot_height - (plot_height * (i / num_ticks)))
        draw.line([(margin_left, y_pos), (width - margin_right, y_pos)], fill=(226, 232, 240), width=1)
        draw.text((margin_left - 12, y_pos), f"{tick_val:.2f}", fill=(100, 116, 139), anchor="rm")

    # Y label
    draw.text((25, margin_top + plot_height // 2), y_label, fill=(51, 65, 85), anchor="mm")

    # Bars
    n_bars = len(categories)
    bar_width = plot_width / (n_bars * 1.6)
    gap = plot_width / n_bars

    for idx, (cat, val, col) in enumerate(zip(categories, values, colors_rgb)):
        x_pos = int(margin_left + idx * gap + (gap - bar_width) / 2)
        bar_h = int((val / y_max) * plot_height) if y_max > 0 else 0
        y_pos = int(margin_top + plot_height - bar_h)

        draw.rectangle([x_pos, y_pos, x_pos + int(bar_width), margin_top + plot_height], fill=col)
        draw.text((x_pos + int(bar_width) // 2, y_pos - 12), f"{val:.4f}", fill=(30, 41, 59), anchor="mm")
        draw.text((x_pos + int(bar_width) // 2, margin_top + plot_height + 25), cat, fill=(51, 65, 85), anchor="mm")

    # Axis line
    draw.line([(margin_left, margin_top + plot_height), (width - margin_right, margin_top + plot_height)], fill=(148, 163, 184), width=2)

    image.save(output_path, "PNG")


def generate_simulation_plots(
    summary_data: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Generate PNG and SVG comparison charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = list(summary_data["strategies"].keys())
    labels = [
        "Always Full" if s == "always_full_attention" else
        "Always Fastest" if s == "always_fastest" else
        "Adaptive" for s in strategies
    ]
    colors_hex = ["#3b82f6", "#10b981", "#8b5cf6"]
    colors_rgb = [(59, 130, 246), (16, 185, 129), (139, 92, 246)]

    hit_rates = [summary_data["strategies"][s]["empirical_deadline_hit_rate"] for s in strategies]
    avg_mious = [summary_data["strategies"][s]["average_selected_miou"] for s in strategies]

    created_files: list[Path] = []

    # 1. Hit Rate Plot
    svg_hit = _render_bar_chart_svg(
        title="Empirical Deadline-Hit Rate by Strategy",
        categories=labels,
        values=hit_rates,
        y_label="Empirical Hit Rate",
        y_max=1.0,
        colors=colors_hex,
    )
    svg_hit_path = output_dir / "hit_rate_by_strategy.svg"
    svg_hit_path.write_text(svg_hit, encoding="utf-8")
    created_files.append(svg_hit_path)

    png_hit_path = output_dir / "hit_rate_by_strategy.png"
    _render_bar_chart_png(
        title="Empirical Deadline-Hit Rate by Strategy",
        categories=labels,
        values=hit_rates,
        y_label="Empirical Hit Rate",
        y_max=1.0,
        colors_rgb=colors_rgb,
        output_path=png_hit_path,
    )
    created_files.append(png_hit_path)

    # 2. Average mIoU Plot
    svg_miou = _render_bar_chart_svg(
        title="Average Selected Validation mIoU by Strategy",
        categories=labels,
        values=avg_mious,
        y_label="Mean mIoU",
        y_max=1.0,
        colors=colors_hex,
    )
    svg_miou_path = output_dir / "avg_miou_by_strategy.svg"
    svg_miou_path.write_text(svg_miou, encoding="utf-8")
    created_files.append(svg_miou_path)

    png_miou_path = output_dir / "avg_miou_by_strategy.png"
    _render_bar_chart_png(
        title="Average Selected Validation mIoU by Strategy",
        categories=labels,
        values=avg_mious,
        y_label="Mean mIoU",
        y_max=1.0,
        colors_rgb=colors_rgb,
        output_path=png_miou_path,
    )
    created_files.append(png_miou_path)

    return created_files


# ---------------------------------------------------------------------------
# Full Simulation Pipeline
# ---------------------------------------------------------------------------
def run_simulation(
    n_requests: int = 1000,
    seed: int = 42,
    metric: str = "p95",
    benchmark_path: Path = DEFAULT_BENCHMARK,
    workload_name: str = "default_mixture",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Execute the three-strategy simulation on a synthetic deadline stream.

    Returns the comprehensive summary dictionary and writes all required artifacts:
      - summary.json
      - request_trace.csv
      - deadline_stream.json
      - visual PNG/SVG plots
    """
    if metric not in SUPPORTED_METRICS:
        raise ControllerError(
            f"Unsupported metric: {metric!r}. Supported: {sorted(SUPPORTED_METRICS)}"
        )

    # 1. Load benchmark modes
    modes = load_benchmark_modes(benchmark_path)
    all_mode_names = sorted(m.name for m in modes)

    # 2. Generate deadline stream
    stream = generate_deadline_stream(n_requests, seed, workload_name)

    # 3. Run the three strategies on the EXACT SAME stream
    decisions_full = evaluate_always_full_attention(stream, modes, metric)
    decisions_fastest = evaluate_always_fastest(stream, modes, metric)
    decisions_adaptive = evaluate_adaptive_controller(stream, modes, metric)

    # 4. Verify request order invariance and completeness
    for idx, req in enumerate(stream):
        d_full, d_fast, d_adapt = decisions_full[idx], decisions_fastest[idx], decisions_adaptive[idx]
        if not (d_full.request_id == d_fast.request_id == d_adapt.request_id == req.request_id):
            raise ControllerError(f"Request ID mismatch at index {idx}")
        if not (d_full.deadline_ms == d_fast.deadline_ms == d_adapt.deadline_ms == req.deadline_ms):
            raise ControllerError(f"Deadline mismatch at index {idx}")

    # 5. Compute metrics
    metrics_full = compute_strategy_metrics(decisions_full, all_mode_names)
    metrics_fastest = compute_strategy_metrics(decisions_fastest, all_mode_names)
    metrics_adaptive = compute_strategy_metrics(decisions_adaptive, all_mode_names)

    # 6. Mode-level latency bounds dictionary for summary
    mode_bounds = {m.name: m.latency_for_metric(metric) for m in modes}
    mode_mious = {m.name: m.validation_miou for m in modes}

    summary_data = {
        "simulation_task": "Phase 8 — Simulate and Compare Deadline-Scheduling Strategies",
        "configuration": {
            "n_requests": n_requests,
            "seed": seed,
            "latency_metric": metric,
            "workload": WORKLOAD_CONFIGS[workload_name],
            "benchmark_source": str(benchmark_path.resolve()),
        },
        "mode_profiles": {
            m: {
                f"{metric}_latency_ms": mode_bounds[m],
                "validation_miou": mode_mious[m],
            }
            for m in all_mode_names
        },
        "strategies": {
            "always_full_attention": metrics_full,
            "always_fastest": metrics_fastest,
            "adaptive_controller": metrics_adaptive,
        },
        "scientific_interpretation_notes": {
            "empirical_nature_of_bounds": (
                "Empirical deadline-hit rate reflects whether the configured statistical "
                f"bound ({metric}) was <= deadline. It is not formal WCET or measured wall-clock success."
            ),
            "fastest_mode_by_metric": (
                f"Under {metric}, the mode with the lowest empirical bound is "
                f"{min(modes, key=lambda m: m.latency_for_metric(metric)).name}."
            ),
        },
    }

    # 7. Write artifacts
    output_dir.mkdir(parents=True, exist_ok=True)

    # summary.json
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # deadline_stream.json
    stream_path = output_dir / "deadline_stream.json"
    stream_dict = [asdict(r) for r in stream]
    stream_path.write_text(json.dumps(stream_dict, indent=2), encoding="utf-8")

    # request_trace.csv
    trace_path = output_dir / "request_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "request_id",
            "deadline_ms",
            "workload_category",
            "strategy",
            "selected_mode",
            "selected_miou",
            "selected_latency_bound_ms",
            "empirical_hit",
            "feasible_modes",
            "fallback_invoked",
        ])
        for decisions_list in [decisions_full, decisions_fastest, decisions_adaptive]:
            for d in decisions_list:
                writer.writerow([
                    d.request_id,
                    f"{d.deadline_ms:.4f}",
                    d.workload_category,
                    d.strategy,
                    d.selected_mode,
                    f"{d.selected_miou:.6f}",
                    f"{d.selected_latency_bound_ms:.4f}",
                    d.empirical_hit,
                    ";".join(d.feasible_modes),
                    d.fallback_invoked,
                ])

    # Plots
    plot_files = generate_simulation_plots(summary_data, output_dir)

    summary_data["output_artifacts"] = {
        "summary_json": str(summary_path.resolve()),
        "deadline_stream_json": str(stream_path.resolve()),
        "request_trace_csv": str(trace_path.resolve()),
        "plots": [str(p.resolve()) for p in plot_files],
    }

    return summary_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 8: Simulate and Compare Deadline-Scheduling Strategies."
    )
    parser.add_argument(
        "--n-requests",
        type=int,
        default=1000,
        help="Number of synthetic requests to generate (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic deadline stream generation (default: 42).",
    )
    parser.add_argument(
        "--metric",
        choices=["p95", "p99", "max_observed"],
        default="p95",
        help="Empirical latency metric to evaluate (default: p95).",
    )
    parser.add_argument(
        "--workload",
        choices=sorted(WORKLOAD_CONFIGS),
        default="default_mixture",
        help="Workload distribution configuration (default: default_mixture).",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help=f"Path to Phase 6B benchmark JSON (default: {DEFAULT_BENCHMARK}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for simulation artifacts (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser


def format_cli_summary(summary: dict[str, Any]) -> str:
    """Format an informative terminal summary report."""
    cfg = summary["configuration"]
    strat = summary["strategies"]
    lines = [
        "=" * 78,
        "  Phase 8 — Real-Time Inference Scheduling Simulation Report",
        "=" * 78,
        f"  Workload:        {cfg['workload']['name']} ({cfg['n_requests']:,} requests, seed={cfg['seed']})",
        f"  Latency Metric:  {cfg['latency_metric']}",
        f"  Benchmark File:  {cfg['benchmark_source']}",
        "",
        "  Mode Profiles under this metric:",
    ]
    for mode_name, prof in summary["mode_profiles"].items():
        bound = prof[f"{cfg['latency_metric']}_latency_ms"]
        miou = prof["validation_miou"]
        lines.append(f"    - {mode_name:22s} Bound: {bound:7.3f} ms | mIoU: {miou:.6f}")

    lines.extend([
        "",
        "-" * 78,
        f"  {'Strategy':25s} | {'Hit Rate':10s} | {'Mean mIoU':11s} | {'Mean Bound':12s} | {'Fallbacks':9s}",
        "-" * 78,
    ])
    for s_name, s_data in strat.items():
        hr = s_data["empirical_deadline_hit_rate"]
        miou = s_data["average_selected_miou"]
        bound = s_data["average_selected_latency_bound_ms"]
        fb = s_data["adaptive_fallback_count"]
        lines.append(
            f"  {s_name:25s} | {hr * 100.0:8.2f}% | {miou:11.6f} | {bound:9.3f} ms | {fb:9d}"
        )

    lines.extend([
        "-" * 78,
        "",
        "  Mode Selection Frequencies:",
    ])
    for s_name, s_data in strat.items():
        freqs = ", ".join(
            f"{m}: {d['count']} ({d['percentage']}%)"
            for m, d in s_data["mode_selection_frequency"].items()
        )
        lines.append(f"    {s_name:25s} -> {freqs}")

    lines.extend([
        "",
        "=" * 78,
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_simulation(
            n_requests=args.n_requests,
            seed=args.seed,
            metric=args.metric,
            benchmark_path=args.benchmark,
            workload_name=args.workload,
            output_dir=args.output_dir,
        )
    except ControllerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(format_cli_summary(summary))
    print(f"Artifacts saved to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
