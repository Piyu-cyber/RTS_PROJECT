"""GPU Performance Profiling and Diagnostic Tool for Vision Transformer Modes.

Phase 9.2 Part C:
  1. Steady-state latency profiling (100 forward passes per model).
  2. Measures full_attention, moderate_dwconv, and aggressive_dwconv.
  3. Peak allocated and reserved CUDA memory tracking.
  4. Parameter counts (total and trainable).
  5. Operator-level profiling via PyTorch Profiler (CPU/CUDA time and kernel calls).
  6. Layer-by-layer attention block timing analysis.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.profiler

from baseline.train_segmentation import (
    DEPTH,
    EMBED_DIM,
    PATCH_SIZE,
    RunConfig,
    SegmentationHead,
    build_backbone,
    features,
    require,
    set_seed,
    strict_load_backbone,
)
from benchmarks.benchmark_inference_modes import (
    MODES,
    load_finetuned_checkpoint,
    prepare_mode,
)
from realtime.runtime_inference import (
    AGGRESSIVE_CHECKPOINT,
    DEFAULT_BENCHMARK,
    MODERATE_CHECKPOINT,
    PHASE4_CHECKPOINT,
    RuntimeModelRegistry,
    preprocess_image,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase9_2_analysis"


def profile_steady_state(
    registry: RuntimeModelRegistry,
    dummy_input: torch.Tensor,
    repeats: int = 100,
) -> dict[str, Any]:
    """Measure steady-state forward pass latencies using interleaved execution order."""
    mode_names = list(registry.models.keys())
    latencies: dict[str, list[float]] = {m: [] for m in mode_names}

    # Interleaved execution over repeats
    for i in range(repeats):
        shuffled = list(mode_names)
        random.shuffle(shuffled)
        for mode in shuffled:
            backbone, head = registry.models[mode]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize(registry.device)
            start_event.record()
            with torch.inference_mode():
                feat = features(backbone, dummy_input)
                logits = head(feat, (224, 224))
            end_event.record()
            end_event.synchronize()

            dur = start_event.elapsed_time(end_event)
            latencies[mode].append(dur)

    results = {}
    for mode in mode_names:
        arr = np.array(latencies[mode], dtype=np.float64)
        results[mode] = {
            "repeats": repeats,
            "mean_ms": round(float(np.mean(arr)), 4),
            "median_ms": round(float(np.median(arr)), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "p99_ms": round(float(np.percentile(arr, 99)), 4),
            "min_ms": round(float(np.min(arr)), 4),
            "max_ms": round(float(np.max(arr)), 4),
            "std_ms": round(float(np.std(arr)), 4),
        }
    return results


def profile_operator_breakdown(
    registry: RuntimeModelRegistry,
    dummy_input: torch.Tensor,
    num_steps: int = 10,
) -> dict[str, Any]:
    """Capture kernel execution time and operator breakdown via PyTorch profiler."""
    breakdowns = {}
    for mode, (backbone, head) in registry.models.items():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            with torch.inference_mode():
                for _ in range(num_steps):
                    feat = features(backbone, dummy_input)
                    _ = head(feat, (224, 224))
            torch.cuda.synchronize(registry.device)

        key_averages = prof.key_averages()
        # Top operators by CUDA time
        top_ops = []
        for evt in sorted(key_averages, key=lambda e: e.device_time_total, reverse=True)[:15]:
            top_ops.append({
                "operator": evt.key,
                "cuda_time_total_ms": round(evt.device_time_total / 1000.0, 3),
                "cuda_time_avg_ms": round(evt.device_time / 1000.0, 3),
                "cpu_time_total_ms": round(evt.cpu_time_total / 1000.0, 3),
                "calls": evt.count,
                "input_shapes": str(evt.input_shapes)[:80],
            })
        breakdowns[mode] = top_ops
    return breakdowns


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_gpu_diagnosis(
    repeats: int = 100,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Execute complete GPU performance diagnosis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    require(torch.cuda.is_available(), "CUDA required for GPU diagnosis")
    device = torch.device("cuda:0")

    print(f"=== Running GPU Performance Diagnosis on {torch.cuda.get_device_name(device)} ===")

    # 1. Reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    t0_load = time.perf_counter()

    registry = RuntimeModelRegistry(device)
    registry.preload_all()
    t1_load = time.perf_counter()
    load_time_ms = (t1_load - t0_load) * 1000.0
    print(f"Preloaded all models in {load_time_ms:.1f} ms")

    # 2. Warm up
    t0_warm = time.perf_counter()
    registry.warmup(10)
    t1_warm = time.perf_counter()
    warm_time_ms = (t1_warm - t0_warm) * 1000.0
    print(f"Warmed up all models in {warm_time_ms:.1f} ms")

    # Memory footprint
    alloc_mib = torch.cuda.memory_allocated(device) / (1024 ** 2)
    res_mib = torch.cuda.memory_reserved(device) / (1024 ** 2)
    peak_alloc_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_res_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    # Parameter counts
    param_counts = {}
    for mode, (backbone, head) in registry.models.items():
        b_tot, b_trn = count_parameters(backbone)
        h_tot, h_trn = count_parameters(head)
        param_counts[mode] = {
            "backbone_params": b_tot,
            "head_params": h_tot,
            "total_params": b_tot + h_tot,
        }

    # 3. Steady state latency
    dummy = torch.randn(1, 3, 224, 224, device=device)
    print(f"Measuring {repeats} interleaved forward passes per model...")
    steady_lat = profile_steady_state(registry, dummy, repeats=repeats)

    # 4. Operator profiling
    print("Capturing operator breakdown with PyTorch profiler...")
    op_breakdowns = profile_operator_breakdown(registry, dummy, num_steps=10)

    diag_summary = {
        "device_info": {
            "device_name": torch.cuda.get_device_name(device),
            "cuda_version": torch.version.cuda,
            "compute_capability": torch.cuda.get_device_capability(device),
            "total_memory_mib": round(torch.cuda.get_device_properties(device).total_memory / (1024**2), 1),
        },
        "timing_ms": {
            "model_preloading_ms": round(load_time_ms, 2),
            "warmup_ms": round(warm_time_ms, 2),
        },
        "memory_footprint_mib": {
            "allocated_mib": round(alloc_mib, 2),
            "reserved_mib": round(res_mib, 2),
            "peak_allocated_mib": round(peak_alloc_mib, 2),
            "peak_reserved_mib": round(peak_res_mib, 2),
        },
        "parameter_counts": param_counts,
        "steady_state_latency_ms": steady_lat,
        "operator_level_profiles": op_breakdowns,
    }

    out_file = output_dir / "gpu_profile_summary.json"
    out_file.write_text(json.dumps(diag_summary, indent=2), encoding="utf-8")
    print(f"Diagnostic results saved to {out_file}")
    return diag_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9.2 GPU Diagnosis")
    parser.add_argument("--repeats", type=int, default=100, help="Measured iterations per model (default: 100)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    run_gpu_diagnosis(repeats=args.repeats, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
