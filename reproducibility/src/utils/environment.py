"""Diagnostic utility to inspect and record runtime hardware and software environments."""

import platform
import sys
import torch
import torchvision
import numpy as np


def collect_environment_info() -> dict:
    """Collect full hardware, OS, Python, and PyTorch diagnostic information."""
    info = {
        "os": platform.platform(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["gpu_compute_capability"] = torch.cuda.get_device_capability(0)
        info["total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    else:
        info["cuda_version"] = None
        info["device_name"] = "CPU only"
        info["total_memory_gb"] = 0.0
    return info


# Alias for backward compatibility
get_environment_info = collect_environment_info


def print_environment_summary(info: dict = None) -> None:
    """Print formatted environment diagnostic summary."""
    if info is None:
        info = collect_environment_info()

    print(f"OS Platform:          {info['os']}")
    print(f"Python Version:       {info['python_version'].split()[0]}")
    print(f"PyTorch Version:      {info['pytorch_version']}")
    print(f"TorchVision Version:  {info['torchvision_version']}")
    print(f"NumPy Version:        {info['numpy_version']}")
    print(f"CUDA Available:       {info['cuda_available']}")
    if info.get("cuda_available"):
        print(f"CUDA Build:           {info['cuda_version']}")
        print(f"cuDNN Version:        {info['cudnn_version']}")
        print(f"GPU Device:           {info['device_name']}")
        print(f"Compute Capability:   {info['gpu_compute_capability']}")
        print(f"Total VRAM:           {info['total_memory_gb']:.2f} GB")
