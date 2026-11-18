"""Main evaluation runner for Phase 23A baseline reproduction."""

import os
import sys
import json
import csv
import datetime
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.phase23_evit.models import CONFIGS
from src.phase23_evit.checkpoint_manager import get_loaded_model, verify_checkpoint_exists
from src.phase23_evit.benchmark import (
    benchmark_gpu_latency,
    benchmark_gpu_throughput,
    measure_peak_memory,
    inspect_token_counts,
    compute_macs,
    verify_real_image_pipeline,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase23_evit_migration")


def run_evaluation() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("======================================================================")
    print(" Phase 23A: EViT Feasibility Audit and Baseline Reproduction")
    print(f" Device: {device_name} ({device})")
    print(f" PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print("======================================================================")

    all_results = []
    latency_profiles = {
        "metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "device": device_name,
            "device_type": device,
            "precision": "FP32",
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "batch_size_latency": 1,
            "warmup_iters_latency": 50,
            "eval_iters_latency": 200,
            "batch_size_throughput": 16,
            "eval_iters_throughput": 50,
            "measurement_units": {
                "latency": "milliseconds (ms)",
                "throughput": "images per second (img/s)",
                "peak_memory": "megabytes (MB)",
                "macs": "Giga-MACs (G)",
                "params": "Millions (M)",
            },
        },
        "models": {},
    }

    for config_name, cfg in CONFIGS.items():
        print(f"\nEvaluating: {config_name} ({cfg['description']})...")
        ckpt_exists = verify_checkpoint_exists(config_name)
        if not ckpt_exists:
            print(f"  [SKIPPED] Missing checkpoint for {config_name}")
            result_row = {
                "model_id": config_name,
                "model_name": cfg["model_fn"],
                "keep_rate": cfg["keep_rate"],
                "fuse_token": cfg["fuse_token"],
                "checkpoint": cfg["checkpoint_filename"],
                "dataset": "ImageNet-1k",
                "sample_count": 0,
                "accuracy_top1": "MISSING_CHECKPOINT",
                "accuracy_top5": "MISSING_CHECKPOINT",
                "latency_mean_ms": "N/A",
                "latency_median_ms": "N/A",
                "latency_p95_ms": "N/A",
                "latency_p99_ms": "N/A",
                "throughput_fps": "N/A",
                "peak_gpu_mem_mb": "N/A",
                "macs_g": "N/A",
                "params_m": "N/A",
                "token_sequence": "N/A",
                "experiment_status": "FAILED_MISSING_CHECKPOINT",
            }
            all_results.append(result_row)
            continue

        # Load model and weights
        model, load_info = get_loaded_model(config_name, device=device)
        params_m = load_info["param_count"] / 1e6
        print(f"  Parameters: {params_m:.2f}M | Keys matched: {load_info['is_perfect_match']}")

        # Token progression
        token_counts = inspect_token_counts(model, device=device)
        print(f"  Token counts per layer: {token_counts}")

        # MACs
        macs_g = compute_macs(model, device=device)
        print(f"  MACs: {macs_g:.2f}G")

        # Peak Memory
        peak_mem_mb = measure_peak_memory(model, device=device, batch_size=1)
        print(f"  Peak GPU Memory (bs=1): {peak_mem_mb:.2f} MB")

        # Latency (Batch size 1, per-frame real-time)
        print("  Benchmarking per-frame latency (bs=1, 50 warmup, 200 iters)...")
        lat_res = benchmark_gpu_latency(
            model,
            device=device,
            batch_size=1,
            warmup_iters=50,
            eval_iters=200,
        )
        batch_lat = lat_res["batch_latency_ms"]
        print(f"  Latency (ms) - Mean: {batch_lat['mean']:.2f}, Median: {batch_lat['median']:.2f}, "
              f"P95: {batch_lat['p95']:.2f}, P99: {batch_lat['p99']:.2f}")

        # Throughput (Batch size 16)
        print("  Benchmarking batch throughput (bs=16)...")
        tp_res = benchmark_gpu_throughput(
            model,
            device=device,
            batch_size=16,
            warmup_iters=20,
            eval_iters=50,
        )
        throughput_fps = tp_res["throughput_img_per_s"]
        print(f"  Throughput: {throughput_fps:.1f} img/s")

        # Real Image Pipeline Test
        pipe_res = verify_real_image_pipeline(
            model,
            image_dir=os.path.join(PROJECT_ROOT, "data", "bdd100k", "images", "10k", "test"),
            num_samples=20,
            device=device,
        )
        print(f"  Real-image pipeline status: {pipe_res['status']} ({pipe_res.get('tested_samples', 0)} samples)")

        # Record profile
        model_profile = {
            "config_name": config_name,
            "description": cfg["description"],
            "model_function": cfg["model_fn"],
            "keep_rate": cfg["keep_rate"],
            "fuse_token": cfg["fuse_token"],
            "checkpoint_filename": cfg["checkpoint_filename"],
            "checkpoint_epoch": load_info.get("checkpoint_epoch"),
            "parameter_count": load_info["param_count"],
            "parameter_count_millions": round(params_m, 2),
            "macs_giga": round(macs_g, 2),
            "layer_token_counts": token_counts,
            "peak_memory_mb": round(peak_mem_mb, 2),
            "throughput_img_per_s": round(throughput_fps, 1),
            "latency_ms": lat_res,
            "throughput_benchmark": tp_res,
            "real_image_pipeline_test": pipe_res,
            "accuracy_status": {
                "imagenet_val_available": False,
                "reason": "ImageNet validation dataset (~50k images) is not available in local workspace.",
                "reported_paper_acc1": {
                    "deit_small_baseline": 79.9,
                    "evit_small_k09_fuse": 79.8,
                    "evit_small_k07_fuse": 79.5,
                    "evit_small_k05_fuse": 78.5,
                }.get(config_name, None),
                "reported_paper_acc5": {
                    "deit_small_baseline": 95.0,
                    "evit_small_k09_fuse": 95.0,
                    "evit_small_k07_fuse": 94.8,
                    "evit_small_k05_fuse": 94.2,
                }.get(config_name, None),
            },
        }
        latency_profiles["models"][config_name] = model_profile

        # CSV row
        result_row = {
            "model_id": config_name,
            "model_name": cfg["model_fn"],
            "keep_rate": cfg["keep_rate"],
            "fuse_token": cfg["fuse_token"],
            "checkpoint": cfg["checkpoint_filename"],
            "dataset": "ImageNet-1k (val absent locally)",
            "sample_count": pipe_res.get("tested_samples", 0),
            "accuracy_top1": "N/A (Dataset unavailable)",
            "accuracy_top5": "N/A (Dataset unavailable)",
            "latency_mean_ms": f"{batch_lat['mean']:.3f}",
            "latency_median_ms": f"{batch_lat['median']:.3f}",
            "latency_p95_ms": f"{batch_lat['p95']:.3f}",
            "latency_p99_ms": f"{batch_lat['p99']:.3f}",
            "throughput_fps": f"{throughput_fps:.1f}",
            "peak_gpu_mem_mb": f"{peak_mem_mb:.2f}",
            "macs_g": f"{macs_g:.2f}",
            "params_m": f"{params_m:.2f}",
            "token_sequence": str(token_counts),
            "experiment_status": "SUCCESS_MEASURED",
        }
        all_results.append(result_row)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved CSV: {csv_path}")

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "latency_profiles.json")
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(latency_profiles, f, indent=2)
    print(f"Saved JSON: {json_path}")

    return all_results, latency_profiles


if __name__ == "__main__":
    run_evaluation()
