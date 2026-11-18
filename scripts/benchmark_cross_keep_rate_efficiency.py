"""Benchmark hardware efficiency (latency, throughput, memory, GMACs) across keep rates on RTX 4060."""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.phase23b.models import build_model
from scripts.run_cross_keep_rate_experiments import STUDY_DIR


def measure_efficiency(model_name: str, arch_config: str, keep_rate: float, gmacs: float, device: str = "cuda"):
    dev = torch.device(device)
    model = build_model(arch_config, num_classes=6, pretrained=False).to(dev)
    model.eval()

    # Warmup
    dummy_bs1 = torch.randn(1, 3, 224, 224, device=dev)
    dummy_bs64 = torch.randn(64, 3, 224, 224, device=dev)

    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_bs1)
            _ = model(dummy_bs64)
    torch.cuda.synchronize()

    # Batch Size 1 Latency (100 repetitions)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    latencies_bs1 = []

    for _ in range(100):
        starter.record()
        with torch.no_grad():
            _ = model(dummy_bs1)
        ender.record()
        torch.cuda.synchronize()
        latencies_bs1.append(starter.elapsed_time(ender))

    lat_bs1_mean = float(np.mean(latencies_bs1))
    lat_bs1_p95 = float(np.percentile(latencies_bs1, 95))
    lat_bs1_p99 = float(np.percentile(latencies_bs1, 99))

    # Batch Size 64 Throughput and Memory
    torch.cuda.reset_peak_memory_stats(dev)
    latencies_bs64 = []

    for _ in range(50):
        starter.record()
        with torch.no_grad():
            _ = model(dummy_bs64)
        ender.record()
        torch.cuda.synchronize()
        latencies_bs64.append(starter.elapsed_time(ender))

    lat_bs64_mean = float(np.mean(latencies_bs64))
    throughput_bs64 = 64.0 / (lat_bs64_mean / 1000.0)
    peak_vram_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

    return {
        "model": model_name,
        "keep_rate": keep_rate,
        "gmacs": gmacs,
        "latency_bs1_mean_ms": round(lat_bs1_mean, 2),
        "latency_bs1_p95_ms": round(lat_bs1_p95, 2),
        "latency_bs1_p99_ms": round(lat_bs1_p99, 2),
        "latency_bs64_mean_ms": round(lat_bs64_mean, 2),
        "throughput_bs64_fps": round(throughput_bs64, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    }


def main():
    benchmarks = [
        ("DeiT-Small", "deit_small_baseline", 1.0, 4.60),
        ("EViT k09", "evit_small_k09_fuse", 0.9, 4.03),
        ("EViT k07", "evit_small_k07_fuse", 0.7, 3.03),
        ("EViT k05", "evit_small_k05_fuse", 0.5, 2.31),
    ]

    records = []
    print("Running hardware efficiency benchmarks on RTX 4060 Laptop GPU...")
    for name, arch, kr, macs in benchmarks:
        print(f"Benchmarking {name} ({arch})...")
        rec = measure_efficiency(name, arch, kr, macs)
        records.append(rec)
        print(f"  --> BS=1 Latency: {rec['latency_bs1_mean_ms']} ms | BS=64 Throughput: {rec['throughput_bs64_fps']} fps | VRAM: {rec['peak_vram_mb']} MB")

    df = pd.DataFrame(records)
    out_csv = os.path.join(STUDY_DIR, "efficiency_benchmark.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved efficiency benchmark to: {out_csv}")
    print(df.to_string())


if __name__ == "__main__":
    main()
