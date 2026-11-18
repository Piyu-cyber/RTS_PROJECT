"""Phase 23A EViT Feasibility Audit and Baseline Reproduction package."""

from .models import build_evit_model, CONFIGS
from .checkpoint_manager import load_checkpoint_into_model, get_loaded_model
from .benchmark import (
    benchmark_gpu_latency,
    benchmark_gpu_throughput,
    measure_peak_memory,
    inspect_token_counts,
    compute_macs,
    verify_real_image_pipeline,
)

__all__ = [
    "build_evit_model",
    "CONFIGS",
    "load_checkpoint_into_model",
    "get_loaded_model",
    "benchmark_gpu_latency",
    "benchmark_gpu_throughput",
    "measure_peak_memory",
    "inspect_token_counts",
    "compute_macs",
    "verify_real_image_pipeline",
]
