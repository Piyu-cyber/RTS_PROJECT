"""GPU VRAM memory footprint benchmark module."""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
from reproducibility.src.utils.logging import get_logger

logger = get_logger("memory_benchmark")


def benchmark_model_memory(
    model: nn.Module,
    batch_size: int = 1,
    img_size: int = 224,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure peak allocated and reserved GPU memory in MB during forward pass.

    Args:
        model: Evaluated PyTorch neural network.
        batch_size: Batch size (default 1 for batch-1 inference).
        img_size: Image height and width (default 224).
        device: Target CUDA device.

    Returns:
        Dict containing peak_allocated_mb, peak_reserved_mb, and param_count_m.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        return {
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
            "param_count_m": sum(p.numel() for p in model.parameters()) / 1e6,
        }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    model = model.eval().to(dev)
    dummy_input = torch.zeros(batch_size, 3, img_size, img_size, device=dev)

    with torch.no_grad():
        _ = model(dummy_input)
        torch.cuda.synchronize(dev)

    peak_allocated_bytes = torch.cuda.max_memory_allocated(dev)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(dev)

    peak_allocated_mb = peak_allocated_bytes / (1024.0 * 1024.0)
    peak_reserved_mb = peak_reserved_bytes / (1024.0 * 1024.0)
    param_count_m = sum(p.numel() for p in model.parameters()) / 1e6

    logger.info(
        f"Memory Profile (batch={batch_size}): Peak Allocated: {peak_allocated_mb:.2f} MB, "
        f"Peak Reserved: {peak_reserved_mb:.2f} MB, Params: {param_count_m:.2f} M"
    )

    return {
        "peak_allocated_mb": float(peak_allocated_mb),
        "peak_reserved_mb": float(peak_reserved_mb),
        "param_count_m": float(param_count_m),
    }
