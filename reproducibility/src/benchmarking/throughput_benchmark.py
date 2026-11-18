"""Throughput benchmark module for batch inference in images/sec (FPS)."""

from __future__ import annotations

import time
from typing import Dict
import torch
import torch.nn as nn
from reproducibility.src.utils.logging import get_logger

logger = get_logger("throughput_benchmark")


def benchmark_model_throughput(
    model: nn.Module,
    batch_size: int = 64,
    iterations: int = 50,
    warmup_runs: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure batch inference throughput in frames per second (FPS).

    Args:
        model: Evaluated PyTorch neural network.
        batch_size: Batch size of input images (default 64).
        iterations: Number of timed batches.
        warmup_runs: Number of warmup batches.
        device: Device to benchmark on.

    Returns:
        Dict containing fps, mean_batch_time_ms, and batch_size.
    """
    dev = torch.device(device)
    model = model.eval().to(dev)

    dummy_input = torch.zeros(batch_size, 3, 224, 224, device=dev)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_runs):
            _ = model(dummy_input)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

        # Timed iterations
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = model(dummy_input)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        total_time_s = time.perf_counter() - start_time

    total_images = batch_size * iterations
    fps = total_images / total_time_s
    mean_batch_time_ms = (total_time_s / iterations) * 1000.0

    logger.info(f"Throughput (batch_size={batch_size}): {fps:.2f} FPS ({mean_batch_time_ms:.2f} ms/batch)")
    return {
        "throughput_fps": float(fps),
        "mean_batch_time_ms": float(mean_batch_time_ms),
        "batch_size": batch_size,
        "total_images": total_images,
        "total_time_seconds": float(total_time_s),
    }
