"""Driver script to compute Sigma_b scores and execute Phase 10A comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "reference"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

from baseline.train_segmentation import (
    DEPTH,
    EMBED_DIM,
    IGNORE_INDEX,
    PATCH_SIZE,
    RunConfig,
    SegmentationHead,
    build_backbone,
    build_datasets,
    evaluate,
    features,
    require,
    set_seed,
    strict_load_backbone,
)
from benchmarks.benchmark_inference_modes import (
    load_finetuned_checkpoint,
    measure_latency,
    prepare_mode,
)
from finetuning.finetune_dwconv import (
    FineTuneConfig,
    configure_trainable_parameters,
    save_checkpoint,
    verify_finite_gradients,
)
from realtime.runtime_inference import PHASE4_CHECKPOINT
from realtime.sigma_b_selection import (
    DEFAULT_OUTPUT_DIR,
    compute_sigma_scores,
    rank_and_select_blocks,
)


def run_phase10a_evaluation(
    calibration_samples: int = 100,
    finetune_epochs: int = 10,
    seed: int = 42,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Execute complete Phase 10A Sigma_b computation and controlled evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    require(torch.cuda.is_available(), "CUDA required for Phase 10A")
    device = torch.device("cuda:0")
    print(f"=== Phase 10A: Paper-Faithful Sigma_b Block Selection ===")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    # 1. Load baseline config and split manifest
    require(PHASE4_CHECKPOINT.is_file(), f"Missing Phase 4 checkpoint: {PHASE4_CHECKPOINT}")
    p4_payload = torch.load(PHASE4_CHECKPOINT, map_location="cpu", weights_only=False)
    p4_cfg = RunConfig(**p4_payload["run_config"])
    split_manifest = p4_payload["split_manifest"]

    # 2. Build datasets using split manifest
    print("Building datasets using Phase 4 split manifest...")
    dataset_root = PROJECT_ROOT / "datasets"
    train_dataset, val_dataset, split_dict = build_datasets(
        dataset_root, p4_cfg, download=False, saved=split_manifest
    )
    val_loader = DataLoader(val_dataset, batch_size=p4_cfg.batch_size, shuffle=False)

    # Calibration loader uses training set with fixed seed (no validation labels used)
    calib_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)

    # 3. Build pristine pretrained backbone for attention map extraction
    set_seed(seed)
    backbone = build_backbone()
    strict_load_backbone(backbone)
    backbone = backbone.to(device).eval()

    # 4. Compute Sigma_h and Sigma_b
    print(f"Computing Sigma_h and Sigma_b across {calibration_samples} calibration samples...")
    sigma_h, sigma_b, meta = compute_sigma_scores(
        backbone=backbone,
        calibration_loader=calib_loader,
        device=device,
        max_samples=calibration_samples,
    )
    print(f"Sigma_b computation complete in {meta['compute_time_seconds']}s")

    # 5. Save per-head and per-block scores
    head_csv = output_dir / "sigma_head_scores.csv"
    with head_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["block_index", "head_index", "sigma_h"])
        for (b, h), score in sorted(sigma_h.items()):
            writer.writerow([b, h, round(score, 6)])

    block_csv = output_dir / "sigma_block_scores.csv"
    with block_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["block_index", "sigma_b"])
        for b, score in sorted(sigma_b.items()):
            writer.writerow([b, round(score, 6)])

    # 6. Rank and select blocks
    sigma_4 = rank_and_select_blocks(sigma_b, k=4)
    sigma_8 = rank_and_select_blocks(sigma_b, k=8)
    manual_4 = [2, 5, 8, 11]

    print("\n=== Block Selection Results ===")
    print(f"Manual 4 Blocks:   {manual_4}")
    print(f"Sigma_b 4 Blocks:  {sigma_4} (Lowest Sigma_b values)")
    print(f"Sigma_b 8 Blocks:  {sigma_8} (Lowest Sigma_b values)")

    manifest = {
        "manual_4_blocks": manual_4,
        "sigma_b_4_blocks": sigma_4,
        "sigma_b_8_blocks": sigma_8,
        "sigma_b_ranking_ascending": sorted(sigma_b.keys(), key=lambda b: sigma_b[b]),
        "calibration_metadata": meta,
    }
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 7. Controlled Fine-Tuning Comparison for Sigma_b 4 Blocks
    # Evaluate manual_4 baseline from existing checkpoint
    mod_ckpt_p = PROJECT_ROOT / "outputs" / "finetune_moderate" / "multiepoch_10" / "best_checkpoint.pt"
    require(mod_ckpt_p.is_file(), f"Missing moderate checkpoint: {mod_ckpt_p}")
    mod_payload = load_finetuned_checkpoint(mod_ckpt_p, "moderate", manual_4, split_manifest)
    mod_payload["_path"] = str(mod_ckpt_p)

    manual_backbone, manual_head, _ = prepare_mode(
        "moderate_dwconv", manual_4, p4_cfg, p4_payload["head_state_dict"], device, mod_payload
    )
    manual_metrics = evaluate(manual_backbone, manual_head, val_loader, device)

    # Fine-tune Sigma_b 4 Blocks using exact same protocol as Phase 6B
    print(f"\n--- Fine-tuning Sigma_b Selected 4 Blocks {sigma_4} [{finetune_epochs} epochs] ---")
    sig4_backbone, sig4_head, _ = prepare_mode(
        "sigma_b_4_blocks", sigma_4, p4_cfg, p4_payload["head_state_dict"], device, None
    )

    trainable = configure_trainable_parameters(sig4_backbone, sig4_head, sigma_4)
    trainable_names = [name for name, _ in trainable]
    optimizer = torch.optim.AdamW(
        (param for _, param in trainable),
        lr=p4_cfg.learning_rate,
        weight_decay=p4_cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=p4_cfg.amp and device.type == "cuda")

    train_loader = DataLoader(train_dataset, batch_size=p4_cfg.batch_size, shuffle=True)
    best_sig4_miou = float("-inf")
    best_sig4_metrics: dict[str, Any] = {}

    ft_config = FineTuneConfig(
        mode="sigma_b_4_blocks",
        blocks=sigma_4,
        image_size=p4_cfg.image_size,
        epochs=finetune_epochs,
        batch_size=p4_cfg.batch_size,
        learning_rate=p4_cfg.learning_rate,
        weight_decay=p4_cfg.weight_decay,
        seed=seed,
        amp=p4_cfg.amp,
        device=str(device),
        backbone_checkpoint=str(PROJECT_ROOT / "reference" / "weights" / "dinov2_vits14_pretrain.pth"),
        head_checkpoint=str(PHASE4_CHECKPOINT),
        split_manifest=str(output_dir / "split_manifest.json"),
        dataset_root=str(dataset_root),
    )

    for epoch in range(1, finetune_epochs + 1):
        sig4_backbone.train()
        sig4_head.train()
        total_loss = 0.0
        total_images = 0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                logits = sig4_head(features(sig4_backbone, images), masks.shape[-2:])
                loss = torch.nn.functional.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in epoch {epoch}: {loss.item()}")
            scaler.scale(loss).backward()
            verify_finite_gradients(trainable, epoch)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.shape[0]
            total_images += images.shape[0]

        val_eval = evaluate(sig4_backbone, sig4_head, val_loader, device)
        avg_loss = total_loss / max(total_images, 1)
        miou = val_eval["mean_iou"]
        print(f"Epoch {epoch:2d}/{finetune_epochs}: Loss={avg_loss:.4f}, Val mIoU={miou:.6f}")

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "validation": val_eval,
        }
        save_checkpoint(
            output_dir / "last_checkpoint.pt",
            sig4_backbone,
            sig4_head,
            ft_config,
            split_manifest,
            epoch,
            epoch_metrics,
            trainable_names,
        )
        if miou is not None and miou > best_sig4_miou:
            best_sig4_miou = miou
            best_sig4_metrics = val_eval
            save_checkpoint(
                output_dir / "best_checkpoint.pt",
                sig4_backbone,
                sig4_head,
                ft_config,
                split_manifest,
                epoch,
                epoch_metrics,
                trainable_names,
            )

    # Load best checkpoint for final evaluation and benchmarking
    best_ckpt = torch.load(output_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    sig4_backbone.load_state_dict(best_ckpt["backbone_state_dict"], strict=True)
    sig4_head.load_state_dict(best_ckpt["head_state_dict"], strict=True)
    best_sig4_metrics = evaluate(sig4_backbone, sig4_head, val_loader, device)

    # 8. Measure device-resident latency (100 repeats, identical to Phase 6B)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    print("\nMeasuring steady-state inference latency (100 repeats)...")
    manual_lat = measure_latency(manual_backbone, manual_head, dummy_input, warmups=10, repeats=100)
    sig4_lat = measure_latency(sig4_backbone, sig4_head, dummy_input, warmups=10, repeats=100)

    # 9. Aggregate results
    comparison_results = {
        "manual_selection_4_blocks": {
            "blocks": manual_4,
            "validation_miou": round(manual_metrics["mean_iou"], 6),
            "background_iou": round(manual_metrics["per_class_iou"]["background"], 6),
            "pet_iou": round(manual_metrics["per_class_iou"]["pet"], 6),
            "median_latency_ms": round(manual_lat["median_ms"], 4),
            "p95_latency_ms": round(manual_lat["p95_ms"], 4),
            "p99_latency_ms": round(manual_lat["p99_ms"], 4),
        },
        "sigma_b_selection_4_blocks": {
            "blocks": sigma_4,
            "validation_miou": round(best_sig4_metrics["mean_iou"], 6),
            "background_iou": round(best_sig4_metrics["per_class_iou"]["background"], 6),
            "pet_iou": round(best_sig4_metrics["per_class_iou"]["pet"], 6),
            "median_latency_ms": round(sig4_lat["median_ms"], 4),
            "p95_latency_ms": round(sig4_lat["p95_ms"], 4),
            "p99_latency_ms": round(sig4_lat["p99_ms"], 4),
        },
        "sigma_b_selection_8_blocks": {
            "blocks": sigma_8,
            "note": "Ranked block indices for aggressive budget",
        },
    }

    (output_dir / "results.json").write_text(json.dumps(comparison_results, indent=2), encoding="utf-8")
    print(f"\nPhase 10A results saved to {output_dir / 'results.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10A Sigma_b Evaluation")
    parser.add_argument("--samples", type=int, default=100, help="Calibration sample count")
    parser.add_argument("--epochs", type=int, default=10, help="Fine-tuning epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()
    run_phase10a_evaluation(
        calibration_samples=args.samples,
        finetune_epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
