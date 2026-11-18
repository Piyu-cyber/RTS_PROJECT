"""CLI Experiment Runner for Phase 9 Runtime Deadline-Aware Inference Prototype.

Executes runtime evaluation across the three scheduling strategies on GPU:
  - always_full_attention
  - always_fastest
  - adaptive_controller

Generates full runtime traces, statistical summaries, prediction masks, and plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from realtime.controller import (
    SUPPORTED_METRICS,
    ControllerError,
    load_benchmark_modes,
)
from realtime.runtime_inference import (
    DEFAULT_BENCHMARK,
    RuntimeModelRegistry,
    RuntimeRequestResult,
    preprocess_image,
    process_runtime_request,
)
from realtime.simulate_strategies import (
    WORKLOAD_CONFIGS,
    SyntheticRequest,
    _render_bar_chart_png,
    _render_bar_chart_svg,
    generate_deadline_stream,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase9_runtime"
PET_DATASET_DIR = PROJECT_ROOT / "datasets" / "oxford-iiit-pet" / "images"


def compute_runtime_statistics(values: list[float]) -> dict[str, float]:
    """Compute standard summary statistics (mean, median, p95, p99, max, std) for a latency list."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "std": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "max": round(float(np.max(arr)), 4),
        "std": round(float(np.std(arr)), 4),
    }


def compute_strategy_runtime_summary(
    results: list[RuntimeRequestResult],
    all_mode_names: list[str],
) -> dict[str, Any]:
    """Compute aggregate performance and breakdown statistics for a runtime strategy."""
    total = len(results)
    actual_hits = sum(1 for r in results if r.actual_deadline_hit)
    actual_misses = total - actual_hits
    predicted_hits = sum(1 for r in results if r.predicted_empirical_feasibility)
    predicted_misses = total - predicted_hits
    fallbacks = sum(1 for r in results if r.likely_deadline_miss)

    # Latencies
    e2e_latencies = [r.total_request_latency_ms for r in results]
    model_latencies = [r.model_execution_ms for r in results]
    cuda_event_latencies = [r.model_cuda_event_ms for r in results]
    ctrl_latencies = [r.controller_latency_ms for r in results]
    prep_latencies = [r.preprocessing_latency_ms for r in results]
    trans_latencies = [r.transfer_latency_ms for r in results]
    post_latencies = [r.postprocessing_latency_ms for r in results]

    # Mode frequencies
    counts: dict[str, int] = {m: 0 for m in all_mode_names}
    for r in results:
        counts[r.selected_mode] = counts.get(r.selected_mode, 0) + 1

    mode_frequencies = {
        m: {
            "count": counts[m],
            "percentage": round((counts[m] / total) * 100.0, 2),
        }
        for m in sorted(counts.keys())
    }

    # Per-category metrics
    by_cat: dict[str, list[RuntimeRequestResult]] = {}
    for r in results:
        by_cat.setdefault(r.workload_category, []).append(r)

    per_category: dict[str, dict[str, Any]] = {}
    for cat_name, cat_res in sorted(by_cat.items()):
        c_tot = len(cat_res)
        c_hits = sum(1 for r in cat_res if r.actual_deadline_hit)
        c_pred = sum(1 for r in cat_res if r.predicted_empirical_feasibility)
        per_category[cat_name] = {
            "request_count": c_tot,
            "actual_deadline_hits": c_hits,
            "actual_deadline_hit_rate": round(c_hits / c_tot, 6),
            "predicted_feasibility_rate": round(c_pred / c_tot, 6),
            "average_selected_miou": round(sum(r.selected_miou for r in cat_res) / c_tot, 6),
            "average_model_latency_ms": round(sum(r.model_execution_ms for r in cat_res) / c_tot, 4),
            "average_e2e_latency_ms": round(sum(r.total_request_latency_ms for r in cat_res) / c_tot, 4),
        }

    return {
        "total_requests": total,
        "actual_deadline_hit_count": actual_hits,
        "actual_deadline_miss_count": actual_misses,
        "actual_deadline_hit_rate": round(actual_hits / total, 6),
        "actual_deadline_miss_rate": round(actual_misses / total, 6),
        "predicted_empirical_feasibility_count": predicted_hits,
        "predicted_empirical_feasibility_rate": round(predicted_hits / total, 6),
        "average_selected_miou": round(sum(r.selected_miou for r in results) / total, 6),
        "adaptive_fallback_count": fallbacks,
        "mode_selection_frequency": mode_frequencies,
        "latency_statistics_ms": {
            "end_to_end": compute_runtime_statistics(e2e_latencies),
            "model_forward_host": compute_runtime_statistics(model_latencies),
            "model_forward_cuda_event": compute_runtime_statistics(cuda_event_latencies),
            "controller_decision": compute_runtime_statistics(ctrl_latencies),
            "preprocessing": compute_runtime_statistics(prep_latencies),
            "host_to_device_transfer": compute_runtime_statistics(trans_latencies),
            "postprocessing": compute_runtime_statistics(post_latencies),
        },
        "per_category_metrics": per_category,
    }


def generate_runtime_plots(
    summary_data: dict[str, Any],
    all_results: dict[str, list[RuntimeRequestResult]],
    output_dir: Path,
) -> list[Path]:
    """Create publication-quality PNG and SVG plots for runtime evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = list(summary_data["strategies"].keys())
    labels = [
        "Always Full" if s == "always_full_attention" else
        "Always Fastest" if s == "always_fastest" else
        "Adaptive" for s in strategies
    ]
    colors_hex = ["#3b82f6", "#10b981", "#8b5cf6"]
    colors_rgb = [(59, 130, 246), (16, 185, 129), (139, 92, 246)]

    created_plots: list[Path] = []

    # 1. Actual E2E Deadline-Hit Rate
    hit_rates = [summary_data["strategies"][s]["actual_deadline_hit_rate"] for s in strategies]
    svg_hit = _render_bar_chart_svg(
        title="Actual End-to-End Deadline-Hit Rate by Strategy",
        categories=labels,
        values=hit_rates,
        y_label="Actual Hit Rate",
        y_max=1.0,
        colors=colors_hex,
    )
    svg_hit_p = output_dir / "actual_hit_rate_by_strategy.svg"
    svg_hit_p.write_text(svg_hit, encoding="utf-8")
    created_plots.append(svg_hit_p)

    png_hit_p = output_dir / "actual_hit_rate_by_strategy.png"
    _render_bar_chart_png(
        title="Actual End-to-End Deadline-Hit Rate by Strategy",
        categories=labels,
        values=hit_rates,
        y_label="Actual Hit Rate",
        y_max=1.0,
        colors_rgb=colors_rgb,
        output_path=png_hit_p,
    )
    created_plots.append(png_hit_p)

    # 2. Average Selected mIoU
    avg_mious = [summary_data["strategies"][s]["average_selected_miou"] for s in strategies]
    svg_miou = _render_bar_chart_svg(
        title="Average Selected Validation mIoU by Strategy",
        categories=labels,
        values=avg_mious,
        y_label="Mean mIoU",
        y_max=1.0,
        colors=colors_hex,
    )
    svg_miou_p = output_dir / "avg_miou_by_strategy.svg"
    svg_miou_p.write_text(svg_miou, encoding="utf-8")
    created_plots.append(svg_miou_p)

    png_miou_p = output_dir / "avg_miou_by_strategy.png"
    _render_bar_chart_png(
        title="Average Selected Validation mIoU by Strategy",
        categories=labels,
        values=avg_mious,
        y_label="Mean mIoU",
        y_max=1.0,
        colors_rgb=colors_rgb,
        output_path=png_miou_p,
    )
    created_plots.append(png_miou_p)

    # 3. Model Execution Latency (Median) by Mode across all recorded requests
    mode_latencies: dict[str, list[float]] = {}
    for res_list in all_results.values():
        for r in res_list:
            mode_latencies.setdefault(r.selected_mode, []).append(r.model_execution_ms)

    sorted_modes = sorted(mode_latencies.keys())
    if sorted_modes:
        median_latencies = [float(np.median(mode_latencies[m])) for m in sorted_modes]
        mode_labels = [
            "Full Attn" if m == "full_attention" else
            "Mod DWConv" if m == "moderate_dwconv" else
            "Agg DWConv" for m in sorted_modes
        ]
        mode_colors_hex = ["#0284c7", "#0d9488", "#e11d48"][:len(sorted_modes)]
        mode_colors_rgb = [(2, 132, 199), (13, 148, 136), (225, 29, 72)][:len(sorted_modes)]

        max_lat = max(median_latencies) * 1.3 if median_latencies else 30.0
        svg_lat = _render_bar_chart_svg(
            title="Measured Median Model Execution Latency by Mode",
            categories=mode_labels,
            values=median_latencies,
            y_label="Median GPU Latency (ms)",
            y_max=round(max_lat, 1),
            colors=mode_colors_hex,
        )
        svg_lat_p = output_dir / "model_latency_by_mode.svg"
        svg_lat_p.write_text(svg_lat, encoding="utf-8")
        created_plots.append(svg_lat_p)

        png_lat_p = output_dir / "model_latency_by_mode.png"
        _render_bar_chart_png(
            title="Measured Median Model Execution Latency by Mode",
            categories=mode_labels,
            values=median_latencies,
            y_label="Median GPU Latency (ms)",
            y_max=round(max_lat, 1),
            colors_rgb=mode_colors_rgb,
            output_path=png_lat_p,
        )
        created_plots.append(png_lat_p)

    return created_plots


def run_runtime_experiment(
    n_requests: int = 50,
    seed: int = 42,
    metric: str = "p95",
    warmup_count: int = 5,
    benchmark_path: Path = DEFAULT_BENCHMARK,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    save_predictions_count: int = 5,
) -> dict[str, Any]:
    """Execute the Phase 9 runtime evaluation pipeline."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Phase 9 runtime execution.")

    device = torch.device("cuda:0")
    modes_metadata = load_benchmark_modes(benchmark_path)
    all_mode_names = sorted(m.name for m in modes_metadata)

    # 1. Initialize & Preload Models
    registry = RuntimeModelRegistry(device)
    print("Preloading all 3 inference models into GPU memory...")
    registry.preload_all()
    print(
        f"Preload complete. Allocated VRAM: "
        f"{torch.cuda.memory_allocated(device)/1024**2:.1f} MiB"
    )

    # 2. Warm up models
    print(f"Running {warmup_count} warm-up iterations per model...")
    registry.warmup(warmup_count)

    # 3. Generate Request Stream
    stream = generate_deadline_stream(n_requests, seed, "default_mixture")

    # 4. Find available dataset images for real image inference
    image_paths: list[Path] = []
    if PET_DATASET_DIR.is_dir():
        image_paths = sorted(PET_DATASET_DIR.glob("*.jpg"))
    if not image_paths:
        print("Notice: Oxford-IIIT Pet images directory not found; using synthetic standard RGB inputs.")

    # 5. Execute each strategy over the exact same stream
    strategies = ["always_full_attention", "always_fastest", "adaptive_controller"]
    all_results: dict[str, list[RuntimeRequestResult]] = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    print(f"Executing runtime evaluation ({n_requests} requests per strategy)...")
    for strat in strategies:
        strat_results: list[RuntimeRequestResult] = []
        for idx, req in enumerate(stream):
            img_source: Any
            if image_paths:
                img_source = image_paths[idx % len(image_paths)]
            else:
                img_source = Image.new("RGB", (224, 224), color=(100 + (idx * 5) % 150, 120, 140))

            save_path: Path | None = None
            if idx < save_predictions_count:
                save_path = preds_dir / f"{strat}_{req.request_id}.png"

            res = process_runtime_request(
                request_id=req.request_id,
                strategy=strat,
                deadline_ms=req.deadline_ms,
                workload_category=req.workload_category,
                image_source=img_source,
                registry=registry,
                modes_metadata=modes_metadata,
                latency_metric=metric,
                save_prediction_path=save_path,
            )
            strat_results.append(res)
        all_results[strat] = strat_results

    # 6. Aggregate Summary
    strategy_summaries: dict[str, Any] = {}
    for strat in strategies:
        strategy_summaries[strat] = compute_strategy_runtime_summary(
            all_results[strat], all_mode_names
        )

    summary_data: dict[str, Any] = {
        "prototype_task": "Phase 9 — Runtime Deadline-Aware Adaptive Inference Prototype",
        "environment": {
            "device": torch.cuda.get_device_name(device),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "gpu_allocated_vram_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 1),
            "gpu_reserved_vram_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 1),
        },
        "configuration": {
            "n_requests": n_requests,
            "seed": seed,
            "latency_metric": metric,
            "warmup_iterations": warmup_count,
            "benchmark_source": str(benchmark_path.resolve()),
        },
        "strategies": strategy_summaries,
        "timing_definitions": {
            "controller_decision": "Host monotonic elapsed time to invoke Phase 7 select_mode",
            "preprocessing": "Host monotonic elapsed time to decode, resize, and normalize input image",
            "host_to_device_transfer": "Synchronized time to transfer float32 tensor to CUDA",
            "model_execution": "Synchronized time for backbone feature extraction + head forward pass",
            "total_request_latency": "Host monotonic time from request arrival to prediction completion",
            "actual_deadline_hit": "total_request_latency <= deadline_ms",
            "predicted_empirical_feasibility": "configured_empirical_latency_bound <= deadline_ms",
        },
    }

    # 7. Write outputs
    # JSON Summary
    summary_path = output_dir / "runtime_summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # Request stream JSON
    stream_path = output_dir / "request_stream.json"
    stream_dicts = [asdict(r) for r in stream]
    stream_path.write_text(json.dumps(stream_dicts, indent=2), encoding="utf-8")

    # CSV Trace
    trace_path = output_dir / "runtime_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "request_id",
            "strategy",
            "deadline_ms",
            "workload_category",
            "selected_mode",
            "selected_miou",
            "configured_latency_bound_ms",
            "controller_latency_ms",
            "preprocessing_latency_ms",
            "transfer_latency_ms",
            "model_execution_ms",
            "model_cuda_event_ms",
            "postprocessing_latency_ms",
            "total_request_latency_ms",
            "predicted_empirical_feasibility",
            "likely_deadline_miss",
            "actual_deadline_hit",
            "prediction_saved_path",
        ])
        for strat in strategies:
            for r in all_results[strat]:
                writer.writerow([
                    r.request_id,
                    r.strategy,
                    f"{r.deadline_ms:.4f}",
                    r.workload_category,
                    r.selected_mode,
                    f"{r.selected_miou:.6f}",
                    f"{r.configured_latency_bound_ms:.4f}",
                    f"{r.controller_latency_ms:.4f}",
                    f"{r.preprocessing_latency_ms:.4f}",
                    f"{r.transfer_latency_ms:.4f}",
                    f"{r.model_execution_ms:.4f}",
                    f"{r.model_cuda_event_ms:.4f}",
                    f"{r.postprocessing_latency_ms:.4f}",
                    f"{r.total_request_latency_ms:.4f}",
                    r.predicted_empirical_feasibility,
                    r.likely_deadline_miss,
                    r.actual_deadline_hit,
                    r.prediction_saved_path,
                ])

    # Plots
    plot_files = generate_runtime_plots(summary_data, all_results, output_dir)
    summary_data["output_artifacts"] = {
        "summary_json": str(summary_path.resolve()),
        "trace_csv": str(trace_path.resolve()),
        "stream_json": str(stream_path.resolve()),
        "plots": [str(p.resolve()) for p in plot_files],
    }

    return summary_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 9: Runtime Deadline-Aware Adaptive Inference Prototype."
    )
    parser.add_argument(
        "--n-requests",
        type=int,
        default=50,
        help="Number of runtime requests per strategy (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for request stream generation (default: 42).",
    )
    parser.add_argument(
        "--metric",
        choices=["p95", "p99", "max_observed"],
        default="p95",
        help="Latency metric used by the controller (default: p95).",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=5,
        help="Number of warm-up iterations per model (default: 5).",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="Path to Phase 6B benchmark JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser


def format_cli_summary(summary: dict[str, Any]) -> str:
    env = summary["environment"]
    cfg = summary["configuration"]
    lines = [
        "=" * 82,
        "  Phase 9 — Runtime Deadline-Aware Adaptive Inference Prototype Report",
        "=" * 82,
        f"  Device:       {env['device']} (VRAM allocated: {env['gpu_allocated_vram_mib']} MiB)",
        f"  Workload:     {cfg['n_requests']} requests per strategy (seed={cfg['seed']}, metric={cfg['latency_metric']})",
        "",
        "-" * 82,
        f"  {'Strategy':24s} | {'Actual Hit':10s} | {'Pred Feas':10s} | {'Mean mIoU':10s} | {'Med Model ms':12s} | {'Med E2E ms':10s}",
        "-" * 82,
    ]
    for s_name, s_data in summary["strategies"].items():
        act_hr = s_data["actual_deadline_hit_rate"] * 100.0
        pred_hr = s_data["predicted_empirical_feasibility_rate"] * 100.0
        miou = s_data["average_selected_miou"]
        med_mod = s_data["latency_statistics_ms"]["model_forward_host"]["median"]
        med_e2e = s_data["latency_statistics_ms"]["end_to_end"]["median"]
        lines.append(
            f"  {s_name:24s} | {act_hr:8.2f}%  | {pred_hr:8.2f}%  | {miou:10.6f} | {med_mod:10.3f} ms | {med_e2e:8.3f} ms"
        )
    lines.extend([
        "-" * 82,
        "",
        "  Component Latency Breakdown (Adaptive Controller Median):",
        f"    - Controller decision: {summary['strategies']['adaptive_controller']['latency_statistics_ms']['controller_decision']['median']:.4f} ms",
        f"    - Preprocessing:       {summary['strategies']['adaptive_controller']['latency_statistics_ms']['preprocessing']['median']:.4f} ms",
        f"    - Host->Device copy:   {summary['strategies']['adaptive_controller']['latency_statistics_ms']['host_to_device_transfer']['median']:.4f} ms",
        f"    - Model forward:       {summary['strategies']['adaptive_controller']['latency_statistics_ms']['model_forward_host']['median']:.4f} ms",
        f"    - Postprocessing:      {summary['strategies']['adaptive_controller']['latency_statistics_ms']['postprocessing']['median']:.4f} ms",
        f"    - Total End-to-End:    {summary['strategies']['adaptive_controller']['latency_statistics_ms']['end_to_end']['median']:.4f} ms",
        "=" * 82,
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_runtime_experiment(
            n_requests=args.n_requests,
            seed=args.seed,
            metric=args.metric,
            warmup_count=args.warmups,
            benchmark_path=args.benchmark,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(format_cli_summary(summary))
    print(f"Artifacts successfully written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
