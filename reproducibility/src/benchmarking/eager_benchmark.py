import time
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn


def benchmark_eager_latency(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    img_size: int = 224,
    warmup: int = 50,
    iterations: int = 200,
) -> Dict[str, float]:
    """Benchmark eager PyTorch execution latency using CUDA Events.
    
    Args:
        model: PyTorch model.
        device: CUDA device.
        batch_size: Batch size (default 1).
        img_size: Square input resolution (default 224).
        warmup: Number of warmup passes.
        iterations: Number of measured passes.
        
    Returns:
        Dictionary of mean, median, p95, p99 latency in ms.
    """
    model.eval()
    model.to(device)

    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times_ms = []
    with torch.no_grad():
        for _ in range(iterations):
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                _ = model(dummy_input)
                end_event.record()
                torch.cuda.synchronize()
                times_ms.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "raw_times": times_ms,
    }


# Alias
benchmark_model_latency = benchmark_eager_latency

