"""Multi-run benchmarking suite for DeiT and EViT models on NVIDIA RTX 4060."""

import os
import sys
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn

try:
    from torchprofile import profile_macs
except ImportError:
    profile_macs = None

from .models import MODEL_CONFIGS, build_model, load_official_checkpoint


def measure_macs(model: nn.Module, device: str = "cuda") -> float:
    """Measure model MACs in Giga-MACs (G)."""
    if profile_macs is None:
        return -1.0
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    try:
        with torch.no_grad():
            macs = profile_macs(model, dummy)
        return float(macs) / 1e9
    except Exception as e:
        print(f"Warning: MACs calculation failed: {e}")
        return -1.0


def inspect_token_sequence(model: nn.Module, device: str = "cuda") -> List[int]:
    """Inspect retained tokens across all Transformer blocks."""
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _, tokens, _ = model.forward_features(dummy)
    return tokens


def benchmark_eager_single_run(
    model: nn.Module,
    device: str = "cuda",
    batch_size: int = 1,
    warmup_iters: int = 50,
    eval_iters: int = 100,
) -> Dict[str, float]:
    """Benchmark eager inference latency using synchronized CUDA events."""
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(eval_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(eval_iters)]

    with torch.no_grad():
        for i in range(eval_iters):
            start_events[i].record()
            _ = model(x)
            end_events[i].record()
        torch.cuda.synchronize()

    timings = np.array([start_events[i].elapsed_time(end_events[i]) for i in range(eval_iters)])

    return {
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "std_ms": float(np.std(timings)),
    }


def benchmark_cuda_graph_single_run(
    model: nn.Module,
    device: str = "cuda",
    batch_size: int = 1,
    warmup_iters: int = 20,
    eval_iters: int = 100,
) -> Dict[str, Any]:
    """Benchmark latency using captured CUDA Graph replay."""
    model.eval()
    static_input = torch.randn(batch_size, 3, 224, 224, device=device)

    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(10):
                _ = model(static_input)
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _ = model(static_input)

        # Warmup replay
        for _ in range(warmup_iters):
            graph.replay()
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(eval_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(eval_iters)]

        for i in range(eval_iters):
            start_events[i].record()
            graph.replay()
            end_events[i].record()
        torch.cuda.synchronize()

        timings = np.array([start_events[i].elapsed_time(end_events[i]) for i in range(eval_iters)])

        return {
            "status": "success",
            "mean_ms": float(np.mean(timings)),
            "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "p99_ms": float(np.percentile(timings, 99)),
            "min_ms": float(np.min(timings)),
            "max_ms": float(np.max(timings)),
            "std_ms": float(np.std(timings)),
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def benchmark_throughput_single_run(
    model: nn.Module,
    device: str = "cuda",
    batch_size: int = 16,
    warmup_iters: int = 20,
    eval_iters: int = 50,
) -> float:
    """Measure batched inference throughput in images/sec."""
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device)

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(eval_iters):
            _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return float((batch_size * eval_iters) / elapsed)


def measure_peak_memory_mb(model: nn.Module, device: str = "cuda") -> float:
    """Measure peak allocated GPU memory in MB during forward pass (batch size 1)."""
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x)
        torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def run_multi_run_benchmark(
    config_name: str,
    device: str = "cuda",
    num_runs: int = 3,
) -> Dict[str, Any]:
    """Execute at least 3 independent benchmark runs for eager, graph, and throughput modes."""
    model = build_model(config_name, num_classes=1000, pretrained=False)
    model, load_info = load_official_checkpoint(model, config_name, device=device)
    model.to(device)

    macs_g = measure_macs(model, device=device)
    param_count = sum(p.numel() for p in model.parameters())
    tokens = inspect_token_sequence(model, device=device)
    peak_mem = measure_peak_memory_mb(model, device=device)

    # 3 independent runs for Eager Mode (bs=1)
    eager_runs = []
    for r in range(num_runs):
        res = benchmark_eager_single_run(model, device=device, batch_size=1)
        res["run_id"] = r + 1
        eager_runs.append(res)

    eager_mean_of_means = float(np.mean([r["mean_ms"] for r in eager_runs]))
    eager_mean_of_medians = float(np.mean([r["median_ms"] for r in eager_runs]))
    eager_mean_of_p95 = float(np.mean([r["p95_ms"] for r in eager_runs]))
    eager_mean_of_p99 = float(np.mean([r["p99_ms"] for r in eager_runs]))

    # 3 independent runs for CUDA Graph Mode (bs=1)
    graph_runs = []
    for r in range(num_runs):
        res = benchmark_cuda_graph_single_run(model, device=device, batch_size=1)
        res["run_id"] = r + 1
        graph_runs.append(res)

    if graph_runs[0].get("status") == "success":
        graph_mean_of_means = float(np.mean([r["mean_ms"] for r in graph_runs]))
        graph_mean_of_medians = float(np.mean([r["median_ms"] for r in graph_runs]))
        graph_mean_of_p95 = float(np.mean([r["p95_ms"] for r in graph_runs]))
        graph_mean_of_p99 = float(np.mean([r["p99_ms"] for r in graph_runs]))
    else:
        graph_mean_of_means = None
        graph_mean_of_medians = None
        graph_mean_of_p95 = None
        graph_mean_of_p99 = None

    # 3 independent runs for Batched Throughput (bs=16)
    throughput_runs = []
    for r in range(num_runs):
        tp = benchmark_throughput_single_run(model, device=device, batch_size=16)
        throughput_runs.append(tp)
    mean_throughput = float(np.mean(throughput_runs))

    return {
        "config_name": config_name,
        "parameters": param_count,
        "parameters_m": round(param_count / 1e6, 2),
        "macs_g": round(macs_g, 2),
        "token_sequence": tokens,
        "peak_memory_mb": round(peak_mem, 2),
        "eager_runs": eager_runs,
        "eager_summary": {
            "mean_ms": round(eager_mean_of_means, 3),
            "median_ms": round(eager_mean_of_medians, 3),
            "p95_ms": round(eager_mean_of_p95, 3),
            "p99_ms": round(eager_mean_of_p99, 3),
        },
        "cuda_graph_runs": graph_runs,
        "cuda_graph_summary": {
            "mean_ms": round(graph_mean_of_means, 3) if graph_mean_of_means is not None else "N/A",
            "median_ms": round(graph_mean_of_medians, 3) if graph_mean_of_medians is not None else "N/A",
            "p95_ms": round(graph_mean_of_p95, 3) if graph_mean_of_p95 is not None else "N/A",
            "p99_ms": round(graph_mean_of_p99, 3) if graph_mean_of_p99 is not None else "N/A",
        },
        "throughput_runs_fps": [round(tp, 1) for tp in throughput_runs],
        "throughput_summary_fps": round(mean_throughput, 1),
    }
