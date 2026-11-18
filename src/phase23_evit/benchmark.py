"""Benchmarking suite for DeiT-Small and EViT models."""

import os
import sys
import time
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

try:
    from torchprofile import profile_macs
except ImportError:
    profile_macs = None

from .models import CONFIGS
from .checkpoint_manager import get_loaded_model


def get_deit_val_transform(img_size: int = 224) -> transforms.Compose:
    """Standard DeiT / ImageNet validation transform."""
    size = int((256 / 224) * img_size)
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def inspect_token_counts(model: nn.Module, device: str = "cuda") -> List[int]:
    """Inspect and return token count at each layer of the model."""
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _, left_tokens, _ = model.forward_features(dummy)
    return left_tokens


def compute_macs(model: nn.Module, device: str = "cuda") -> float:
    """Compute MACs (GFLOPs) using torchprofile if available."""
    if profile_macs is None:
        return -1.0
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)
    try:
        with torch.no_grad():
            macs = profile_macs(model, dummy)
        return float(macs) / 1e9  # in Giga-MACs
    except Exception as e:
        print(f"Warning: MACs computation failed: {e}")
        return -1.0


def benchmark_gpu_latency(
    model: nn.Module,
    device: str = "cuda",
    batch_size: int = 1,
    warmup_iters: int = 50,
    eval_iters: int = 200,
) -> Dict[str, Any]:
    """Measure inference latency using synchronized CUDA events."""
    if not torch.cuda.is_available() or device != "cuda":
        raise RuntimeError("CUDA is required for accurate GPU latency benchmarking.")

    model.to("cuda")
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device="cuda")

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
        torch.cuda.synchronize()

    # Latency timing with CUDA events
    timings_ms = []
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(eval_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(eval_iters)]

    with torch.no_grad():
        for i in range(eval_iters):
            start_events[i].record()
            _ = model(x)
            end_events[i].record()
        torch.cuda.synchronize()

    for i in range(eval_iters):
        timings_ms.append(start_events[i].elapsed_time(end_events[i]))

    timings = np.array(timings_ms)
    per_image_timings = timings / batch_size

    return {
        "batch_size": batch_size,
        "warmup_iters": warmup_iters,
        "eval_iters": eval_iters,
        "batch_latency_ms": {
            "mean": float(np.mean(timings)),
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
            "p99": float(np.percentile(timings, 99)),
            "min": float(np.min(timings)),
            "max": float(np.max(timings)),
            "std": float(np.std(timings)),
        },
        "per_image_latency_ms": {
            "mean": float(np.mean(per_image_timings)),
            "median": float(np.median(per_image_timings)),
            "p95": float(np.percentile(per_image_timings, 95)),
            "p99": float(np.percentile(per_image_timings, 99)),
            "min": float(np.min(per_image_timings)),
            "max": float(np.max(per_image_timings)),
            "std": float(np.std(per_image_timings)),
        },
    }


def benchmark_gpu_throughput(
    model: nn.Module,
    device: str = "cuda",
    batch_size: int = 16,
    warmup_iters: int = 20,
    eval_iters: int = 50,
) -> Dict[str, Any]:
    """Measure batch throughput (images / second) on GPU."""
    model.to("cuda")
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device="cuda")

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
        torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(eval_iters):
            _ = model(x)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time

    total_images = batch_size * eval_iters
    throughput_fps = total_images / total_time

    return {
        "batch_size": batch_size,
        "eval_iters": eval_iters,
        "total_images": total_images,
        "total_time_s": total_time,
        "throughput_img_per_s": float(throughput_fps),
    }


def measure_peak_memory(
    model: nn.Module,
    device: str = "cuda",
    batch_size: int = 1,
) -> float:
    """Measure peak GPU memory allocated during a forward pass in MB."""
    model.to("cuda")
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device="cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x)
        torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated()
    return float(peak_bytes / (1024 * 1024))  # MB


def verify_real_image_pipeline(
    model: nn.Module,
    image_dir: str = "data/bdd100k/images/10k/test",
    num_samples: int = 20,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run real images through the evaluation pipeline to verify inference sanity."""
    if not os.path.isdir(image_dir):
        return {
            "status": "dataset_missing",
            "message": f"Image directory not found: {image_dir}",
            "tested_samples": 0,
        }

    transform = get_deit_val_transform(224)
    image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ][:num_samples]

    if not image_files:
        return {
            "status": "no_images_found",
            "message": "No valid image files found in directory",
            "tested_samples": 0,
        }

    model.to(device)
    model.eval()

    tensors = []
    for f in image_files:
        try:
            img = Image.open(f).convert("RGB")
            t = transform(img)
            tensors.append(t)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not tensors:
        return {"status": "image_loading_failed", "tested_samples": 0}

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=-1)
        top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)

    return {
        "status": "pipeline_verified",
        "tested_samples": len(tensors),
        "output_shape": list(logits.shape),
        "mean_top1_confidence": float(top5_probs[:, 0].mean().item()),
    }
