"""CUDA Graph latency benchmark module for batch-1 real-time inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from reproducibility.src.models.deit import build_deit_small
from reproducibility.src.models.evit import build_evit_small
from reproducibility.src.utils.logging import get_logger

logger = get_logger("cuda_graph_benchmark")


class CUDAGraphBenchmarkRunner:
    """Captures and benchmarks static CUDA Graphs for batch-1 vision transformer inference."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        img_size: int = 224,
        num_classes: int = 6,
        warmup_runs: int = 50,
    ):
        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("CUDA is required for CUDA Graph benchmarking.")

        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.num_classes = num_classes
        self.img_size = img_size
        self.warmup_runs = warmup_runs

        # Allocate static input and output buffers
        self.static_input = torch.zeros(1, 3, img_size, img_size, device=self.device)
        self.static_output = torch.zeros(1, num_classes, device=self.device)

        # Warm up on dedicated stream
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(self.warmup_runs):
                self.static_output = self.model(self.static_input)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        # Capture static CUDA Graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)

    def replay(self, input_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Replay the captured CUDA graph."""
        if input_tensor is not None:
            self.static_input.copy_(input_tensor)
        self.graph.replay()
        return self.static_output.clone()

    def benchmark(self, iterations: int = 200) -> Dict[str, float]:
        """Benchmark CUDA Graph replay latency using CUDA events."""
        latencies_ms: List[float] = []

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Additional warm-up passes
        for _ in range(10):
            self.graph.replay()
        torch.cuda.synchronize(self.device)

        for _ in range(iterations):
            start_event.record()
            self.graph.replay()
            end_event.record()
            torch.cuda.synchronize(self.device)
            elapsed_ms = start_event.elapsed_time(end_event)
            latencies_ms.append(float(elapsed_ms))

        arr = np.array(latencies_ms, dtype=np.float64)
        return {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "median_ms": float(np.median(arr)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_observed_ms": float(np.max(arr)),
            "min_observed_ms": float(np.min(arr)),
            "iterations": iterations,
            "raw_latencies_ms": latencies_ms,
        }


def benchmark_model_cuda_graph(
    model: nn.Module,
    iterations: int = 200,
    warmup_runs: int = 50,
    device: str = "cuda",
) -> Dict[str, float]:
    """Convenience function to benchmark a model using CUDA Graphs."""
    runner = CUDAGraphBenchmarkRunner(
        model=model,
        device=device,
        warmup_runs=warmup_runs,
    )
    return runner.benchmark(iterations=iterations)
