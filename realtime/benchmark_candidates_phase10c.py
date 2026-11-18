"""Candidate registry and device benchmarking for Phase 10C.

Evaluates and benchmarks the 3 validated candidate models:
  1. full_attention (Phase 4 Baseline)
  2. moderate_dwconv (Phase 6B Moderate, blocks [2, 5, 8, 11])
  3. sigma_b_dwconv (Phase 10B Run 3, blocks [1, 2, 3, 4])

Uses a fair common protocol on the GPU with randomized round-robin execution,
recording model-only and end-to-end component latencies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ["XFORMERS_DISABLED"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "reference"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

import numpy as np
import torch
from PIL import Image

from baseline.train_segmentation import (
    DEPTH,
    EMBED_DIM,
    PATCH_SIZE,
    RunConfig,
    SegmentationHead,
    build_backbone,
    features,
    require,
    set_seed,
    strict_load_backbone,
)
from benchmarks.benchmark_inference_modes import (
    capture_transfer_sources,
    make_patch_config,
    verify_patch,
)
from lib.patching.patch import patch_dinov2_convatt
from realtime.runtime_inference import execute_model_forward, preprocess_image

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase10c_adaptive_scheduler"
AUDIT_INVENTORY_PATH = (
    PROJECT_ROOT / "outputs" / "phase10b_1_integrity_audit" / "checkpoint_inventory.json"
)
RECONCILED_RESULTS_PATH = (
    PROJECT_ROOT / "outputs" / "phase10b_1_integrity_audit" / "reconciled_results.csv"
)

CANDIDATE_DEFS = {
    "full_attention": {
        "display_name": "Full Attention (Phase 4 Baseline)",
        "checkpoint_path": PROJECT_ROOT / "outputs" / "pet_experiment" / "best_checkpoint.pt",
        "blocks": [],
        "source_phase": "Phase 4",
        "fine_tuned_jointly": False,
    },
    "moderate_dwconv": {
        "display_name": "Moderate DWConv (Manual Interleaved [2,5,8,11])",
        "checkpoint_path": PROJECT_ROOT
        / "outputs"
        / "finetune_moderate"
        / "multiepoch_10"
        / "best_checkpoint.pt",
        "blocks": [2, 5, 8, 11],
        "source_phase": "Phase 6B",
        "fine_tuned_jointly": True,
    },
    "sigma_b_dwconv": {
        "display_name": "Sigma-b DWConv (Contiguous [1,2,3,4] Run 3)",
        "checkpoint_path": PROJECT_ROOT
        / "outputs"
        / "phase10b_two_stage"
        / "run3_sigma_selected_joint"
        / "best_checkpoint.pt",
        "blocks": [1, 2, 3, 4],
        "source_phase": "Phase 10B Run 3",
        "fine_tuned_jointly": True,
    },
}


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "std": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "max": round(float(np.max(arr)), 4),
        "std": round(float(np.std(arr)), 4),
    }


def load_candidate_models(device: torch.device) -> dict[str, tuple[torch.nn.Module, SegmentationHead]]:
    models: dict[str, tuple[torch.nn.Module, SegmentationHead]] = {}
    p4_ckpt_path = CANDIDATE_DEFS["full_attention"]["checkpoint_path"]
    p4_payload = torch.load(p4_ckpt_path, map_location="cpu", weights_only=False)
    cfg = RunConfig(**p4_payload["run_config"])

    for name, cdef in CANDIDATE_DEFS.items():
        ckpt_path = cdef["checkpoint_path"]
        require(ckpt_path.is_file(), f"Checkpoint missing: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        set_seed(cfg.seed)
        backbone = build_backbone()
        strict_load_backbone(backbone)

        blocks = cdef["blocks"]
        if blocks:
            expected = capture_transfer_sources(backbone, blocks)
            patch_dinov2_convatt(backbone, make_patch_config(blocks, cfg.image_size))
            backbone = backbone.to(device).eval()
            verify_patch(backbone, blocks, expected, device)
        else:
            backbone = backbone.to(device).eval()

        head = SegmentationHead().to(device).eval()

        if name == "full_attention":
            head.load_state_dict(payload["head_state_dict"], strict=True)
        else:
            backbone.load_state_dict(payload["backbone_state_dict"], strict=True)
            head.load_state_dict(payload["head_state_dict"], strict=True)

        models[name] = (backbone, head)

    return models


def create_candidate_registry() -> dict[str, Any]:
    with open(AUDIT_INVENTORY_PATH, "r", encoding="utf-8") as f:
        inv = json.load(f)

    reconciled_mious = {
        "full_attention": 0.743031,
        "moderate_dwconv": 0.703628,
        "sigma_b_dwconv": 0.480098,
    }

    registry: dict[str, Any] = {}
    for name, cdef in CANDIDATE_DEFS.items():
        ckpt_path = cdef["checkpoint_path"]
        sha256 = compute_sha256(ckpt_path)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        inv_entry = (
            inv["phase4_baseline"]
            if name == "full_attention"
            else (inv["phase6b_moderate"] if name == "moderate_dwconv" else inv["phase10b_run3"])
        )
        require(sha256 == inv_entry["sha256"], f"SHA256 mismatch for {name}: {sha256} vs {inv_entry['sha256']}")

        val_metrics = inv_entry["metrics"]["validation"]
        registry[name] = {
            "name": name,
            "display_name": cdef["display_name"],
            "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256,
            "blocks": cdef["blocks"],
            "num_dwconv_blocks": len(cdef["blocks"]),
            "source_phase": cdef["source_phase"],
            "audited_validation_miou": reconciled_mious[name],
            "checkpoint_validation_miou": round(val_metrics["mean_iou"], 6),
            "background_iou": round(val_metrics["per_class_iou"]["background"], 6),
            "pet_iou": round(val_metrics["per_class_iou"]["pet"], 6),
            "best_epoch": inv_entry["epoch"],
            "trainable_param_count": inv_entry.get("trainable_param_count", 0),
        }

    return registry


def run_benchmark(
    models: dict[str, tuple[torch.nn.Module, SegmentationHead]],
    device: torch.device,
    sample_images: list[Path],
    num_rounds: int = 100,
    warmup_rounds: int = 10,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    candidate_names = list(models.keys())

    # Pre-warmup
    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.inference_mode():
        for name in candidate_names:
            bb, hd = models[name]
            for _ in range(warmup_rounds):
                feat = features(bb, dummy)
                _ = hd(feat, (224, 224))
            torch.cuda.synchronize(device)

    # Prepare benchmark inputs
    raw_measurements: list[dict[str, Any]] = []
    times_by_model: dict[str, dict[str, list[float]]] = {
        name: {"model_cuda_ms": [], "model_host_ms": [], "prep_ms": [], "trans_ms": [], "post_ms": [], "e2e_ms": []}
        for name in candidate_names
    }

    # Run randomized rounds
    for round_idx in range(num_rounds):
        round_order = list(candidate_names)
        rng.shuffle(round_order)
        img_path = sample_images[round_idx % len(sample_images)]

        for name in round_order:
            bb, hd = models[name]

            # Measure full request steps
            t_req_0 = time.perf_counter()

            # Preprocess
            t_prep_0 = time.perf_counter()
            img_tensor = preprocess_image(img_path)
            t_prep_1 = time.perf_counter()
            prep_ms = (t_prep_1 - t_prep_0) * 1000.0

            # Transfer
            t_trans_0 = time.perf_counter()
            img_cuda = img_tensor.to(device, non_blocking=False)
            torch.cuda.synchronize(device)
            t_trans_1 = time.perf_counter()
            trans_ms = (t_trans_1 - t_trans_0) * 1000.0

            # Model forward
            preds_cpu, model_host_ms, model_cuda_ms = execute_model_forward(bb, hd, img_cuda)

            # Postprocess
            t_post_0 = time.perf_counter()
            _ = preds_cpu.byte().numpy()
            t_post_1 = time.perf_counter()
            post_ms = (t_post_1 - t_post_0) * 1000.0

            t_req_1 = time.perf_counter()
            e2e_ms = (t_req_1 - t_req_0) * 1000.0

            times_by_model[name]["model_cuda_ms"].append(model_cuda_ms)
            times_by_model[name]["model_host_ms"].append(model_host_ms)
            times_by_model[name]["prep_ms"].append(prep_ms)
            times_by_model[name]["trans_ms"].append(trans_ms)
            times_by_model[name]["post_ms"].append(post_ms)
            times_by_model[name]["e2e_ms"].append(e2e_ms)

            raw_measurements.append({
                "round_idx": round_idx,
                "model_name": name,
                "image_name": img_path.name,
                "prep_ms": round(prep_ms, 4),
                "trans_ms": round(trans_ms, 4),
                "model_host_ms": round(model_host_ms, 4),
                "model_cuda_ms": round(model_cuda_ms, 4),
                "post_ms": round(post_ms, 4),
                "e2e_ms": round(e2e_ms, 4),
            })

    summary_by_model: dict[str, Any] = {}
    for name in candidate_names:
        summary_by_model[name] = {
            "model_cuda_ms": compute_summary_stats(times_by_model[name]["model_cuda_ms"]),
            "model_host_ms": compute_summary_stats(times_by_model[name]["model_host_ms"]),
            "prep_ms": compute_summary_stats(times_by_model[name]["prep_ms"]),
            "trans_ms": compute_summary_stats(times_by_model[name]["trans_ms"]),
            "post_ms": compute_summary_stats(times_by_model[name]["post_ms"]),
            "e2e_ms": compute_summary_stats(times_by_model[name]["e2e_ms"]),
            # Benchmark profile fields for controller consumption
            "p90_ms": compute_summary_stats(times_by_model[name]["model_host_ms"])["p90"],
            "p95_ms": compute_summary_stats(times_by_model[name]["model_host_ms"])["p95"],
            "p99_ms": compute_summary_stats(times_by_model[name]["model_host_ms"])["p99"],
            "max_observed_ms": compute_summary_stats(times_by_model[name]["model_host_ms"])["max"],
            "median_ms": compute_summary_stats(times_by_model[name]["model_host_ms"])["median"],
        }

    return summary_by_model, raw_measurements


def main() -> None:
    print("=" * 70)
    print("Phase 10C: Candidate Registry & Device Latency Benchmarking")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "CUDA GPU required for device benchmarking")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    # 1. Build Candidate Registry
    print("\n--- Constructing Candidate Registry ---")
    registry = create_candidate_registry()
    registry_path = OUTPUT_DIR / "candidate_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"Saved Candidate Registry to: {registry_path}")

    # 2. Preload Models
    print("\n--- Preloading Candidate Models on GPU ---")
    models = load_candidate_models(device)
    print("All 3 candidate models loaded and verified successfully.")

    # 3. Locate Sample Images
    images_dir = PROJECT_ROOT / "datasets" / "oxford-iiit-pet" / "images"
    sample_images = sorted(images_dir.glob("*.jpg"))[:32]
    require(len(sample_images) > 0, "No sample images found for benchmarking")
    print(f"Using {len(sample_images)} test images across 100 randomized rounds.")

    # 4. Run Benchmark
    print("\n--- Executing Common-Protocol Device Benchmark (100 rounds, randomized) ---")
    t0 = time.time()
    summary, raw = run_benchmark(models, device, sample_images, num_rounds=100, warmup_rounds=10, seed=42)
    t1 = time.time()
    print(f"Benchmarking complete in {t1 - t0:.2f}s.")

    # 5. Combine with Candidate Registry
    benchmark_modes: dict[str, Any] = {}
    for name, stats in summary.items():
        candidate_meta = registry[name]
        benchmark_modes[name] = {
            "name": name,
            "display_name": candidate_meta["display_name"],
            "validation_miou": candidate_meta["audited_validation_miou"],
            "background_iou": candidate_meta["background_iou"],
            "pet_iou": candidate_meta["pet_iou"],
            "blocks": candidate_meta["blocks"],
            "model_host_latency": stats["model_host_ms"],
            "model_cuda_event_latency": stats["model_cuda_ms"],
            "e2e_latency": stats["e2e_ms"],
            "p90_ms": stats["p90_ms"],
            "p95_ms": stats["p95_ms"],
            "p99_ms": stats["p99_ms"],
            "max_observed_ms": stats["max_observed_ms"],
            "median_ms": stats["median_ms"],
        }

    benchmark_output = {
        "metadata": {
            "device": torch.cuda.get_device_name(device),
            "num_rounds": 100,
            "warmup_rounds": 10,
            "protocol": "randomized round-robin with synchronized events",
            "evaluated_candidates": list(CANDIDATE_DEFS.keys()),
        },
        "modes": benchmark_modes,
    }

    # Save benchmark outputs
    benchmark_json_path = OUTPUT_DIR / "benchmark_results.json"
    with open(benchmark_output_json := benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_output, f, indent=2)
    print(f"Saved benchmark summary to: {benchmark_output_json}")

    csv_path = OUTPUT_DIR / "benchmark_raw_measurements.csv"
    if raw:
        fieldnames = list(raw[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw)
        print(f"Saved raw trace measurements to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (RTX 4060 Laptop GPU):")
    print(f"{'Mode':<18} | {'mIoU':<8} | {'Median':<8} | {'p90':<8} | {'p95':<8} | {'p99':<8} | {'Max':<8}")
    print("-" * 70)
    for name, mode_info in benchmark_modes.items():
        miou = mode_info["validation_miou"]
        med = mode_info["median_ms"]
        p90 = mode_info["p90_ms"]
        p95 = mode_info["p95_ms"]
        p99 = mode_info["p99_ms"]
        mx = mode_info["max_observed_ms"]
        print(f"{name:<18} | {miou:<8.4f} | {med:<8.2f} | {p90:<8.2f} | {p95:<8.2f} | {p99:<8.2f} | {mx:<8.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
