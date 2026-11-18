"""Fair, Repeated Runtime Evaluation Runner for Adaptive Vision Transformers.

Phase 9.1:
  - Multi-round matched evaluation (5 rounds x 50 requests x 3 strategies = 750 requests per metric).
  - Explicit warm-cache priming (filesystem read & decode) before timed rounds.
  - Full GPU warm-up across all models.
  - Per-round independent randomization of strategy execution order with recorded seeds.
  - Strict matched-request comparisons with Wilson 95% confidence intervals.
  - Component-level latency recording and model-only vs. end-to-end deadline tracking.
  - Re-computes summary statistics directly from raw traces with automated fairness validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from realtime.controller import (
    SUPPORTED_METRICS,
    ControllerError,
    ModeProfile,
    load_benchmark_modes,
    select_mode,
)
from realtime.runtime_inference import (
    DEFAULT_BENCHMARK,
    RuntimeModelRegistry,
    execute_model_forward,
    preprocess_image,
)
from realtime.simulate_strategies import (
    WORKLOAD_CONFIGS,
    SyntheticRequest,
    _render_bar_chart_png,
    _render_bar_chart_svg,
    generate_deadline_stream,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PET_DATASET_DIR = PROJECT_ROOT / "datasets" / "oxford-iiit-pet" / "images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase9_1_fair"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


@dataclass(frozen=True)
class FairRequestRecord:
    """Detailed record for a single request in a fair evaluation round."""

    round_id: int
    request_id: str
    strategy: str
    controller_metric: str
    selected_model: str
    deadline_budget_ms: float
    workload_category: str
    controller_ms: float
    preprocessing_ms: float
    host_to_device_ms: float
    model_forward_ms: float
    model_cuda_event_ms: float
    postprocessing_ms: float
    end_to_end_ms: float
    model_deadline_hit: bool
    end_to_end_deadline_hit: bool
    predicted_feasible: bool
    likely_deadline_miss: bool
    selected_model_miou: float
    execution_order_index: int
    round_order_seed: int
    error_message: str = ""


def compute_wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute 95% Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z**2) / total
    centre = (p + (z**2) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z**2) / (4.0 * (total**2))))
    return round(max(0.0, centre - margin), 6), round(min(1.0, centre + margin), 6)


def warm_cache_images(image_paths: list[Path]) -> None:
    """Read and decode every image from disk once into memory to warm OS and PIL caches."""
    for p in image_paths:
        with Image.open(p) as img:
            img.load()


def process_fair_request(
    round_id: int,
    request_id: str,
    strategy: str,
    controller_metric: str,
    deadline_ms: float,
    workload_category: str,
    image_source: Image.Image | Path | str | torch.Tensor,
    registry: RuntimeModelRegistry,
    modes_metadata: list[ModeProfile],
    execution_order_index: int,
    round_order_seed: int,
) -> FairRequestRecord:
    """Execute a single request with monotonic host timers and CUDA event synchronization."""
    t_start = time.perf_counter()

    # 1. Controller selection
    t_ctrl_0 = time.perf_counter()
    if strategy == "always_full_attention":
        selected_mode = "full_attention"
        prof = next(m for m in modes_metadata if m.name == "full_attention")
        selected_miou = prof.validation_miou
        bound_ms = prof.latency_for_metric(controller_metric)
        predicted_feasible = bound_ms <= deadline_ms
        likely_miss = not predicted_feasible
    elif strategy == "always_fastest":
        fastest_prof = min(
            modes_metadata,
            key=lambda m: (m.latency_for_metric(controller_metric), -m.validation_miou, m.name),
        )
        selected_mode = fastest_prof.name
        selected_miou = fastest_prof.validation_miou
        bound_ms = fastest_prof.latency_for_metric(controller_metric)
        predicted_feasible = bound_ms <= deadline_ms
        likely_miss = not predicted_feasible
    elif strategy == "adaptive_controller":
        res = select_mode(deadline_ms, modes_metadata, controller_metric)
        selected_mode = res.selected_mode
        selected_miou = res.selected_miou
        bound_ms = res.selected_empirical_latency_bound_ms
        predicted_feasible = not res.likely_deadline_miss
        likely_miss = res.likely_deadline_miss
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    t_ctrl_1 = time.perf_counter()
    ctrl_ms = (t_ctrl_1 - t_ctrl_0) * 1000.0

    # 2. Preprocessing
    t_prep_0 = time.perf_counter()
    image_tensor = preprocess_image(image_source)
    t_prep_1 = time.perf_counter()
    prep_ms = (t_prep_1 - t_prep_0) * 1000.0

    # 3. Host-to-device transfer
    t_trans_0 = time.perf_counter()
    image_cuda = image_tensor.to(registry.device, non_blocking=False)
    torch.cuda.synchronize(registry.device)
    t_trans_1 = time.perf_counter()
    trans_ms = (t_trans_1 - t_trans_0) * 1000.0

    # 4. Model forward pass on GPU
    backbone, head = registry.models[selected_mode]
    preds_cpu, model_host_ms, model_cuda_ms = execute_model_forward(
        backbone, head, image_cuda
    )

    # 5. Postprocessing
    t_post_0 = time.perf_counter()
    _ = preds_cpu.numpy()
    t_post_1 = time.perf_counter()
    post_ms = (t_post_1 - t_post_0) * 1000.0

    # 6. Total end-to-end duration
    t_end = time.perf_counter()
    e2e_ms = (t_end - t_start) * 1000.0

    # 7. Deadline evaluations
    e2e_hit = e2e_ms <= deadline_ms
    model_hit = model_host_ms <= deadline_ms

    return FairRequestRecord(
        round_id=round_id,
        request_id=request_id,
        strategy=strategy,
        controller_metric=controller_metric,
        selected_model=selected_mode,
        deadline_budget_ms=deadline_ms,
        workload_category=workload_category,
        controller_ms=ctrl_ms,
        preprocessing_ms=prep_ms,
        host_to_device_ms=trans_ms,
        model_forward_ms=model_host_ms,
        model_cuda_event_ms=model_cuda_ms,
        postprocessing_ms=post_ms,
        end_to_end_ms=e2e_ms,
        model_deadline_hit=model_hit,
        end_to_end_deadline_hit=e2e_hit,
        predicted_feasible=predicted_feasible,
        likely_deadline_miss=likely_miss,
        selected_model_miou=selected_miou,
        execution_order_index=execution_order_index,
        round_order_seed=round_order_seed,
    )


def compute_distribution_stats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for latency lists."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "std": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "std": round(float(np.std(arr)), 4),
    }


def compute_aggregate_metrics(
    records: list[FairRequestRecord],
    all_mode_names: list[str],
) -> dict[str, Any]:
    """Aggregate metrics across requests for a single strategy."""
    total = len(records)
    if total == 0:
        return {"total_requests": 0}

    e2e_hits = sum(1 for r in records if r.end_to_end_deadline_hit)
    model_hits = sum(1 for r in records if r.model_deadline_hit)
    pred_feas = sum(1 for r in records if r.predicted_feasible)

    e2e_hit_ci = compute_wilson_ci(e2e_hits, total)
    model_hit_ci = compute_wilson_ci(model_hits, total)
    pred_feas_ci = compute_wilson_ci(pred_feas, total)

    # Confusion matrix: Predicted Feasible (Pos) vs. End-to-End Hit (Pos)
    tp = sum(1 for r in records if r.predicted_feasible and r.end_to_end_deadline_hit)
    fp = sum(1 for r in records if r.predicted_feasible and not r.end_to_end_deadline_hit)
    fn = sum(1 for r in records if not r.predicted_feasible and r.end_to_end_deadline_hit)
    tn = sum(1 for r in records if not r.predicted_feasible and not r.end_to_end_deadline_hit)

    e2e_latencies = [r.end_to_end_ms for r in records]
    model_latencies = [r.model_forward_ms for r in records]
    ctrl_latencies = [r.controller_ms for r in records]
    prep_latencies = [r.preprocessing_ms for r in records]
    trans_latencies = [r.host_to_device_ms for r in records]

    # Model mode frequencies
    mode_counts: dict[str, int] = {m: 0 for m in all_mode_names}
    for r in records:
        mode_counts[r.selected_model] = mode_counts.get(r.selected_model, 0) + 1

    mode_frequencies = {
        m: {
            "count": mode_counts[m],
            "percentage": round((mode_counts[m] / total) * 100.0, 2),
        }
        for m in sorted(mode_counts.keys())
    }

    # Per-round metrics
    by_round: dict[int, list[FairRequestRecord]] = {}
    for r in records:
        by_round.setdefault(r.round_id, []).append(r)

    round_metrics = {}
    for rnd_id, rnd_recs in sorted(by_round.items()):
        r_tot = len(rnd_recs)
        r_e2e_h = sum(1 for r in rnd_recs if r.end_to_end_deadline_hit)
        r_mod_h = sum(1 for r in rnd_recs if r.model_deadline_hit)
        round_metrics[f"round_{rnd_id}"] = {
            "request_count": r_tot,
            "e2e_hit_count": r_e2e_h,
            "e2e_hit_rate": round(r_e2e_h / r_tot, 6),
            "model_hit_count": r_mod_h,
            "model_hit_rate": round(r_mod_h / r_tot, 6),
            "median_e2e_ms": round(float(np.median([r.end_to_end_ms for r in rnd_recs])), 4),
            "median_model_ms": round(float(np.median([r.model_forward_ms for r in rnd_recs])), 4),
            "execution_order_index": rnd_recs[0].execution_order_index,
        }

    return {
        "total_requests": total,
        "end_to_end_deadline_hit_count": e2e_hits,
        "end_to_end_deadline_hit_rate": round(e2e_hits / total, 6),
        "end_to_end_deadline_hit_ci_95": list(e2e_hit_ci),
        "model_deadline_hit_count": model_hits,
        "model_deadline_hit_rate": round(model_hits / total, 6),
        "model_deadline_hit_ci_95": list(model_hit_ci),
        "predicted_feasibility_count": pred_feas,
        "predicted_feasibility_rate": round(pred_feas / total, 6),
        "predicted_feasibility_ci_95": list(pred_feas_ci),
        "mean_selected_model_miou": round(sum(r.selected_model_miou for r in records) / total, 6),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "latency_statistics_ms": {
            "end_to_end": compute_distribution_stats(e2e_latencies),
            "model_forward": compute_distribution_stats(model_latencies),
            "controller": compute_distribution_stats(ctrl_latencies),
            "preprocessing": compute_distribution_stats(prep_latencies),
            "host_to_device": compute_distribution_stats(trans_latencies),
        },
        "mode_selection_frequency": mode_frequencies,
        "per_round_metrics": round_metrics,
    }


def validate_fair_traces(
    records: list[FairRequestRecord],
    expected_rounds: int,
    expected_requests_per_round: int,
    expected_strategies: list[str],
    modes_metadata: list[ModeProfile],
) -> dict[str, Any]:
    """Strict fairness and validity validator."""
    errors: list[str] = []

    # Total count
    expected_total = expected_rounds * expected_requests_per_round * len(expected_strategies)
    if len(records) != expected_total:
        errors.append(f"Expected {expected_total} records, got {len(records)}")

    # Check finite and non-negative latencies
    for idx, r in enumerate(records):
        for lat_name, val in [
            ("controller_ms", r.controller_ms),
            ("preprocessing_ms", r.preprocessing_ms),
            ("host_to_device_ms", r.host_to_device_ms),
            ("model_forward_ms", r.model_forward_ms),
            ("end_to_end_ms", r.end_to_end_ms),
        ]:
            if not math.isfinite(val) or val < 0.0:
                errors.append(f"Row {idx} ({r.request_id}) has invalid {lat_name}: {val}")

    # Check request and deadline matching across strategies within each round
    by_round: dict[int, dict[str, list[FairRequestRecord]]] = {}
    for r in records:
        by_round.setdefault(r.round_id, {}).setdefault(r.strategy, []).append(r)

    observed_orders: list[list[str]] = []
    for rnd_id, strat_map in by_round.items():
        if set(strat_map.keys()) != set(expected_strategies):
            errors.append(f"Round {rnd_id} missing strategies: {set(expected_strategies) - set(strat_map.keys())}")
            continue

        base_strat = expected_strategies[0]
        base_recs = strat_map[base_strat]
        if len(base_recs) != expected_requests_per_round:
            errors.append(f"Round {rnd_id} strategy {base_strat} has {len(base_recs)} requests, expected {expected_requests_per_round}")

        # Check matched deadlines and request IDs
        base_req_tuples = [(r.request_id, r.deadline_budget_ms) for r in base_recs]
        for other_strat in expected_strategies[1:]:
            other_recs = strat_map[other_strat]
            other_tuples = [(r.request_id, r.deadline_budget_ms) for r in other_recs]
            if base_req_tuples != other_tuples:
                errors.append(f"Round {rnd_id}: Mismatch in request IDs or deadlines between {base_strat} and {other_strat}")

        # Record strategy execution order
        ordered = sorted(expected_strategies, key=lambda s: strat_map[s][0].execution_order_index)
        observed_orders.append(ordered)

    # Check execution order variation across rounds
    unique_orders = len(set(tuple(o) for o in observed_orders))
    order_varied = unique_orders > 1

    # Check controller model selection conformance
    for idx, r in enumerate(records):
        if r.strategy == "always_full_attention":
            if r.selected_model != "full_attention":
                errors.append(f"Row {idx}: always_full_attention selected {r.selected_model}")
        elif r.strategy == "always_fastest":
            fastest_prof = min(
                modes_metadata,
                key=lambda m: (m.latency_for_metric(r.controller_metric), -m.validation_miou, m.name),
            )
            if r.selected_model != fastest_prof.name:
                errors.append(f"Row {idx}: always_fastest selected {r.selected_model}, expected {fastest_prof.name}")
        elif r.strategy == "adaptive_controller":
            sel = select_mode(r.deadline_budget_ms, modes_metadata, r.controller_metric)
            if r.selected_model != sel.selected_mode:
                errors.append(f"Row {idx}: adaptive_controller selected {r.selected_model}, expected {sel.selected_mode}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "total_records_checked": len(records),
        "observed_orders": observed_orders,
        "unique_orders_count": unique_orders,
        "strategy_order_varied": order_varied,
    }


def run_fair_experiment_for_metric(
    metric: str,
    n_rounds: int,
    n_requests_per_round: int,
    base_seed: int,
    registry: RuntimeModelRegistry,
    modes_metadata: list[ModeProfile],
    image_paths: list[Path],
) -> tuple[list[FairRequestRecord], dict[str, Any]]:
    """Execute multi-round fair evaluation for a specific controller metric."""
    strategies = ["always_full_attention", "always_fastest", "adaptive_controller"]
    all_mode_names = sorted([m.name for m in modes_metadata])
    records: list[FairRequestRecord] = []

    for rnd_id in range(1, n_rounds + 1):
        # 1. Deterministic request stream for this round
        round_seed = base_seed + (rnd_id * 1000)
        stream = generate_deadline_stream(n_requests_per_round, round_seed)

        # 2. Randomize strategy execution order for this round
        round_order_seed = round_seed + 777
        rnd_gen = random.Random(round_order_seed)
        shuffled_strategies = list(strategies)
        rnd_gen.shuffle(shuffled_strategies)

        # 3. Execute strategies in shuffled order
        for exec_idx, strat in enumerate(shuffled_strategies):
            for req_idx, req in enumerate(stream):
                img_path = image_paths[req_idx % len(image_paths)]
                rec = process_fair_request(
                    round_id=rnd_id,
                    request_id=req.request_id,
                    strategy=strat,
                    controller_metric=metric,
                    deadline_ms=req.deadline_ms,
                    workload_category=req.workload_category,
                    image_source=img_path,
                    registry=registry,
                    modes_metadata=modes_metadata,
                    execution_order_index=exec_idx,
                    round_order_seed=round_order_seed,
                )
                records.append(rec)

    # Aggregate summaries
    summaries = {}
    for strat in strategies:
        strat_recs = [r for r in records if r.strategy == strat]
        summaries[strat] = compute_aggregate_metrics(strat_recs, all_mode_names)

    return records, summaries


def generate_fair_plots(
    summaries_by_metric: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Generate comparative PNG and SVG charts for end-to-end and model-only hit rates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    labels = ["Always Full", "Always Fastest", "Adaptive"]
    strategies = ["always_full_attention", "always_fastest", "adaptive_controller"]
    colors_hex = ["#3b82f6", "#10b981", "#8b5cf6"]
    colors_rgb = [(59, 130, 246), (16, 185, 129), (139, 92, 246)]

    for metric, strats in summaries_by_metric.items():
        # E2E Hit Rate
        e2e_vals = [strats[s]["end_to_end_deadline_hit_rate"] for s in strategies]
        svg_e2e = _render_bar_chart_svg(
            title=f"Fair End-to-End Hit Rate ({metric.upper()}, 5 Rounds Warm-Cache)",
            categories=labels,
            values=e2e_vals,
            y_label="End-to-End Hit Rate",
            y_max=1.0,
            colors=colors_hex,
        )
        p_svg = output_dir / f"fair_e2e_hit_rate_{metric}.svg"
        p_svg.write_text(svg_e2e, encoding="utf-8")
        created.append(p_svg)

        p_png = output_dir / f"fair_e2e_hit_rate_{metric}.png"
        _render_bar_chart_png(
            title=f"Fair End-to-End Hit Rate ({metric.upper()}, 5 Rounds Warm-Cache)",
            categories=labels,
            values=e2e_vals,
            y_label="End-to-End Hit Rate",
            y_max=1.0,
            colors_rgb=colors_rgb,
            output_path=p_png,
        )
        created.append(p_png)

        # Model-Only Hit Rate
        mod_vals = [strats[s]["model_deadline_hit_rate"] for s in strategies]
        svg_mod = _render_bar_chart_svg(
            title=f"Model-Only Hit Rate ({metric.upper()}, 5 Rounds Warm-Cache)",
            categories=labels,
            values=mod_vals,
            y_label="Model-Only Hit Rate",
            y_max=1.0,
            colors=colors_hex,
        )
        p_mod_svg = output_dir / f"fair_model_hit_rate_{metric}.svg"
        p_mod_svg.write_text(svg_mod, encoding="utf-8")
        created.append(p_mod_svg)

        p_mod_png = output_dir / f"fair_model_hit_rate_{metric}.png"
        _render_bar_chart_png(
            title=f"Model-Only Hit Rate ({metric.upper()}, 5 Rounds Warm-Cache)",
            categories=labels,
            values=mod_vals,
            y_label="Model-Only Hit Rate",
            y_max=1.0,
            colors_rgb=colors_rgb,
            output_path=p_mod_png,
        )
        created.append(p_mod_png)

    return created


def run_experiment(
    n_rounds: int = 5,
    n_requests_per_round: int = 50,
    seed: int = 42,
    warmup_count: int = 5,
    benchmark_path: Path = DEFAULT_BENCHMARK,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Main experiment coordinator across all three controller metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Starting Phase 9.1 Fair Runtime Evaluation ===")
    print(f"Rounds: {n_rounds}, Requests per round: {n_requests_per_round}, Seed: {seed}")
    print(f"Output directory: {output_dir}")

    # 1. Hardware verification
    require(torch.cuda.is_available(), "CUDA device required for runtime evaluation")
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    # 2. Load benchmark metadata
    modes_metadata = load_benchmark_modes(benchmark_path)

    # 3. Preload all three models into GPU
    registry = RuntimeModelRegistry(device)
    print("Preloading all 3 models into GPU memory...")
    registry.preload_all()
    print(f"Preload complete. VRAM allocated: {torch.cuda.memory_allocated(device)/1024**2:.1f} MiB")

    # 4. GPU warm-up
    print(f"Executing {warmup_count} GPU warm-up iterations per model...")
    registry.warmup(warmup_count)

    # 5. Filesystem cache warming
    img_paths = sorted(list(PET_DATASET_DIR.glob("*.jpg")))[:n_requests_per_round]
    print(f"Warming filesystem cache on {len(img_paths)} sample images...")
    warm_cache_images(img_paths)
    print("Filesystem cache primed.")

    # 6. Execute fair experiments across all three metrics
    all_traces: list[FairRequestRecord] = []
    summaries_by_metric: dict[str, dict[str, Any]] = {}
    validations_by_metric: dict[str, dict[str, Any]] = {}

    expected_strats = ["always_full_attention", "always_fastest", "adaptive_controller"]

    for metric in ["p95", "p99", "max_observed"]:
        print(f"\n--- Running evaluation for controller metric: {metric} ---")
        records, summaries = run_fair_experiment_for_metric(
            metric=metric,
            n_rounds=n_rounds,
            n_requests_per_round=n_requests_per_round,
            base_seed=seed,
            registry=registry,
            modes_metadata=modes_metadata,
            image_paths=img_paths,
        )
        all_traces.extend(records)
        summaries_by_metric[metric] = summaries

        # Validate fairness and consistency
        val_res = validate_fair_traces(
            records=records,
            expected_rounds=n_rounds,
            expected_requests_per_round=n_requests_per_round,
            expected_strategies=expected_strats,
            modes_metadata=modes_metadata,
        )
        validations_by_metric[metric] = val_res
        if not val_res["valid"]:
            print(f"VALIDATION WARNING for {metric}: {val_res['errors']}")
        else:
            print(f"Validation PASS for {metric} (Unique strategy orders: {val_res['unique_orders_count']})")

    # 7. Write raw_traces.csv
    trace_path = output_dir / "raw_traces.csv"
    print(f"\nWriting raw traces to {trace_path} ({len(all_traces)} rows)...")
    fieldnames = [
        "round_id",
        "request_id",
        "strategy",
        "controller_metric",
        "selected_model",
        "deadline_budget_ms",
        "workload_category",
        "controller_ms",
        "preprocessing_ms",
        "host_to_device_ms",
        "model_forward_ms",
        "model_cuda_event_ms",
        "postprocessing_ms",
        "end_to_end_ms",
        "model_deadline_hit",
        "end_to_end_deadline_hit",
        "predicted_feasible",
        "likely_deadline_miss",
        "selected_model_miou",
        "execution_order_index",
        "round_order_seed",
        "error_message",
    ]
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_traces:
            writer.writerow(asdict(r))

    # 8. Write summary.json
    summary_data = {
        "experiment": "Phase 9.1 Fair, Repeated Runtime Evaluation",
        "environment": {
            "device": torch.cuda.get_device_name(device),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "vram_allocated_mib": round(torch.cuda.memory_allocated(device) / 1024**2, 1),
            "vram_reserved_mib": round(torch.cuda.memory_reserved(device) / 1024**2, 1),
        },
        "config": {
            "n_rounds": n_rounds,
            "n_requests_per_round": n_requests_per_round,
            "total_requests_per_strategy_per_metric": n_rounds * n_requests_per_round,
            "base_seed": seed,
            "warmup_count": warmup_count,
            "benchmark_source": str(benchmark_path.resolve()),
        },
        "metrics_summary": summaries_by_metric,
        "timing_definitions": {
            "controller_ms": "Host monotonic elapsed time for select_mode call",
            "preprocessing_ms": "Host monotonic elapsed time for resize, convert RGB, and normalize",
            "host_to_device_ms": "Synchronized time to copy float tensor to CUDA device",
            "model_forward_ms": "Host monotonic duration of backbone + head forward pass",
            "model_cuda_event_ms": "CUDA event recorded GPU duration of forward pass",
            "postprocessing_ms": "Argmax prediction extraction time",
            "end_to_end_ms": "Single measured host monotonic interval from request arrival to prediction completion",
            "model_deadline_hit": "model_forward_ms <= deadline_budget_ms",
            "end_to_end_deadline_hit": "end_to_end_ms <= deadline_budget_ms",
            "predicted_feasible": "configured_empirical_latency_bound <= deadline_budget_ms",
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # 9. Write validation.json
    val_path = output_dir / "validation.json"
    val_path.write_text(json.dumps(validations_by_metric, indent=2), encoding="utf-8")

    # 10. Write experiment_config.json
    config_path = output_dir / "experiment_config.json"
    config_path.write_text(json.dumps(summary_data["config"], indent=2), encoding="utf-8")

    # 11. Generate plots
    print("Generating comparative plots...")
    generate_fair_plots(summaries_by_metric, output_dir)

    print("\nPhase 9.1 fair evaluation complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9.1 Fair Runtime Evaluation Runner")
    parser.add_argument("--rounds", type=int, default=5, help="Number of matched experimental rounds (default: 5)")
    parser.add_argument("--requests-per-round", type=int, default=50, help="Requests per strategy per round (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--warmups", type=int, default=5, help="GPU warm-up iterations (default: 5)")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK, help="Benchmark JSON path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()
    run_experiment(
        n_rounds=args.rounds,
        n_requests_per_round=args.requests_per_round,
        seed=args.seed,
        warmup_count=args.warmups,
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
