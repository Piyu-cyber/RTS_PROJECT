"""Remeasure physical GPU latency profiles for the 4 DeiT/EViT models on BDD100K.

Measures:
- Static CUDA Graph latency (batch_size=1, FP32, RTX 4060)
- Eager PyTorch latency
- Throughput (batch_size=16)
- Accurate p95, median, mean, p99 timing distributions

Generates:
- outputs/bdd100k_migration/model_profiles.json
- outputs/bdd100k_migration/latency_profiles.csv
"""

from __future__ import annotations

import os
import sys
import json
import csv
import time
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.phase23b.models import MODEL_CONFIGS, build_model
from src.phase23b.bdd_dataset import NUM_BDD_CLASSES, BDD_CLASSES
from src.phase24.gpu_trace_replay import CUDAGraphRunner

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")


def measure_latencies(
    config_name: str,
    device: str = "cuda",
    warmup: int = 50,
    iterations: int = 200,
) -> Dict[str, Any]:
    """Measure eager and CUDA Graph latency on GPU using torch.cuda.Event."""
    ckpt_path = os.path.join(CKPT_DIR, f"{config_name}_bdd_best.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")

    model = build_model(config_name, num_classes=NUM_BDD_CLASSES, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    dummy_bs1 = torch.zeros(1, 3, 224, 224, device=device)

    # 1. Eager Latency
    for _ in range(warmup):
        _ = model(dummy_bs1)
    torch.cuda.synchronize()

    eager_times = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(iterations):
            start_ev.record()
            _ = model(dummy_bs1)
            end_ev.record()
            torch.cuda.synchronize()
            eager_times.append(start_ev.elapsed_time(end_ev))

    # 2. CUDA Graph Latency
    graph_runner = CUDAGraphRunner(model, device=device, num_classes=NUM_BDD_CLASSES)

    for _ in range(warmup):
        _ = graph_runner.run(dummy_bs1)
    torch.cuda.synchronize()

    graph_times = []
    with torch.no_grad():
        for _ in range(iterations):
            start_ev.record()
            _ = graph_runner.run(dummy_bs1)
            end_ev.record()
            torch.cuda.synchronize()
            graph_times.append(start_ev.elapsed_time(end_ev))

    # 3. Batched Throughput (bs=16)
    dummy_bs16 = torch.zeros(16, 3, 224, 224, device=device)
    for _ in range(20):
        _ = model(dummy_bs16)
    torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_bs16)
            torch.cuda.synchronize()
    total_imgs = 16 * 50
    fps = round(total_imgs / (time.time() - t0), 1)

    def stats(arr: List[float]) -> Dict[str, float]:
        return {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "std": round(float(np.std(arr)), 3),
        }

    return {
        "config_name": config_name,
        "eager": stats(eager_times),
        "cuda_graph": stats(graph_times),
        "throughput_fps": fps,
        "val_top1": ckpt.get("best_val_acc", 0.0),
        "checkpoint_path": os.path.relpath(ckpt_path, PROJECT_ROOT).replace("\\", "/"),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA device required for physical hardware latency benchmarking.")
        return

    print("================================================================================")
    print(" Benchmarking Hardware Latency Profiles on NVIDIA RTX 4060 (BDD100K 6 Classes)")
    print("================================================================================")

    results = {}
    csv_rows = []

    # Token sequences
    token_sequences = {
        "deit_small_baseline": [196] * 12,
        "evit_small_k09_fuse": [196, 196, 196, 178, 178, 178, 162, 162, 162, 147, 147, 147],
        "evit_small_k07_fuse": [196, 196, 196, 139, 139, 139, 99, 99, 99, 71, 71, 71],
        "evit_small_k05_fuse": [196, 196, 196, 99, 99, 99, 51, 51, 51, 27, 27, 27],
    }
    params_m = {
        "deit_small_baseline": 22.05,
        "evit_small_k09_fuse": 22.05,
        "evit_small_k07_fuse": 22.05,
        "evit_small_k05_fuse": 22.05,
    }
    macs_g = {
        "deit_small_baseline": 4.60,
        "evit_small_k09_fuse": 4.03,
        "evit_small_k07_fuse": 3.03,
        "evit_small_k05_fuse": 2.31,
    }
    display_names = {
        "deit_small_baseline": "DeiT-Small Baseline",
        "evit_small_k09_fuse": "EViT-Small 0.9",
        "evit_small_k07_fuse": "EViT-Small 0.7",
        "evit_small_k05_fuse": "EViT-Small 0.5",
    }

    # Load BDD fine-tuning test accuracies
    test_accs = {}
    finetune_json = os.path.join(OUTPUT_DIR, "bdd_finetuning_results.json")
    if os.path.isfile(finetune_json):
        with open(finetune_json, "r") as f:
            ft_data = json.load(f)
        for k, v in ft_data.items():
            test_accs[k] = (v.get("test_top1_accuracy", 0.0), v.get("test_macro_f1", 0.0))

    profiles_payload = {
        "metadata": {
            "dataset": "BDD100K",
            "task": "driving_scene_classification",
            "num_classes": NUM_BDD_CLASSES,
            "classes": BDD_CLASSES,
            "device": torch.cuda.get_device_name(0),
            "measurement_iterations": 200,
            "warmup_iterations": 50,
            "precision": "FP32",
            "batch_size": 1,
        },
        "profiles": {},
    }

    for cfg_name in MODEL_CONFIGS.keys():
        print(f"\nBenchmarking {cfg_name} (200 iterations)...", flush=True)
        m = measure_latencies(cfg_name, device=device)
        results[cfg_name] = m

        t_top1, t_f1 = test_accs.get(cfg_name, (m["val_top1"], 0.0))

        prof = {
            "config_id": cfg_name,
            "display_name": display_names[cfg_name],
            "keep_rate": MODEL_CONFIGS[cfg_name]["keep_rate"],
            "fuse_token": MODEL_CONFIGS[cfg_name]["fuse_token"],
            "parameters_m": params_m[cfg_name],
            "macs_g": macs_g[cfg_name],
            "token_sequence": token_sequences[cfg_name],
            "val_top1_accuracy": m["val_top1"],
            "test_top1_accuracy": t_top1,
            "test_macro_f1": t_f1,
            "cuda_graph_latency_ms": m["cuda_graph"],
            "eager_latency_ms": m["eager"],
            "throughput_fps": m["throughput_fps"],
            "checkpoint_path": m["checkpoint_path"],
            "official_checkpoint": MODEL_CONFIGS[cfg_name]["checkpoint_filename"],
        }
        profiles_payload["profiles"][cfg_name] = prof

        print(f"  CUDA Graph p95: {m['cuda_graph']['p95']} ms | Median: {m['cuda_graph']['median']} ms | Eager Mean: {m['eager']['mean']} ms | Throughput: {m['throughput_fps']} fps")

        csv_rows.append({
            "model_id": cfg_name,
            "keep_rate": prof["keep_rate"],
            "macs_g": prof["macs_g"],
            "bdd_val_top1": prof["val_top1_accuracy"],
            "bdd_test_top1": prof["test_top1_accuracy"],
            "cuda_graph_p95_ms": m["cuda_graph"]["p95"],
            "cuda_graph_median_ms": m["cuda_graph"]["median"],
            "cuda_graph_mean_ms": m["cuda_graph"]["mean"],
            "eager_mean_ms": m["eager"]["mean"],
            "throughput_fps": m["throughput_fps"],
        })

    profiles_path = os.path.join(OUTPUT_DIR, "model_profiles.json")
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles_payload, f, indent=2)
    print(f"\nSaved updated BDD100K model profiles to: {profiles_path}")

    csv_path = os.path.join(OUTPUT_DIR, "latency_profiles.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved latency profiles CSV to: {csv_path}")


if __name__ == "__main__":
    main()
