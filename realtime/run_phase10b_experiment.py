"""Phase 10B runner: Two-stage fine-tuning and controlled ablation study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "reference"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

from baseline.train_segmentation import (
    IGNORE_INDEX,
    NUM_CLASSES,
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
    BACKBONE_CHECKPOINT,
    HEAD_CHECKPOINT,
    FineTuneConfig,
    configure_trainable_parameters,
    patch_and_verify,
    save_checkpoint,
    verify_finite_gradients,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase10b_two_stage"
DATASET_ROOT = PROJECT_ROOT / "datasets"


def compute_confusion_matrix_and_ious(
    backbone: nn.Module,
    head: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate model and return structured IoUs and confusion matrix."""
    return evaluate(backbone, head, val_loader, device)


def train_epoch(
    backbone: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    trainable: list[tuple[str, nn.Parameter]],
    epoch: int,
    device: torch.device,
) -> float:
    """Execute a single training epoch."""
    backbone.train()
    head.train()
    total_loss = 0.0
    total_images = 0

    for images, masks in train_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = head(features(backbone, images), masks.shape[-2:])
            loss = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss in epoch {epoch}: {loss.item()}")
        scaler.scale(loss).backward()
        verify_finite_gradients(trainable, epoch)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.shape[0]
        total_images += images.shape[0]

    return total_loss / max(total_images, 1)


def run_stage2_finetune(
    run_id: str,
    run_name: str,
    blocks: list[int],
    policy: str,  # 'joint', 'frozen_head', 'staged'
    epochs: int,
    stage1_checkpoint: Path,
    p4_cfg: RunConfig,
    split_manifest: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    """Execute Stage 2 fine-tuning under specified trainable-parameter policy."""
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=======================================================")
    print(f"Starting {run_id}: {run_name}")
    print(f"Blocks: {blocks} (0-based) / {[b+1 for b in blocks]} (1-based)")
    print(f"Policy: {policy}, Epochs: {epochs}")
    print(f"=======================================================")

    # Ensure exact reproducibility matching Phase 6B
    set_seed(17)

    # Initialize backbone and load pristine pretrained weights
    backbone = build_backbone()
    strict_load_backbone(backbone)

    # Initialize head and load Stage 1 checkpoint
    head = SegmentationHead().to(device)
    stage1_data = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(stage1_data["head_state_dict"], strict=True)

    # Patch DWConv blocks
    patch_and_verify(backbone, blocks, device)

    # Configure trainable parameters according to policy
    if policy == "frozen_head":
        for p in backbone.parameters():
            p.requires_grad = False
        for idx in blocks:
            for p in backbone.blocks[idx].attn.parameters():
                p.requires_grad = True
        for p in head.parameters():
            p.requires_grad = False
        trainable = [(f"backbone.{n}", p) for n, p in backbone.named_parameters() if p.requires_grad]
    elif policy in {"joint", "staged"}:
        trainable = configure_trainable_parameters(backbone, head, blocks)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    trainable_names = [n for n, _ in trainable]
    optimizer = torch.optim.AdamW(
        (p for _, p in trainable),
        lr=p4_cfg.learning_rate,
        weight_decay=p4_cfg.weight_decay,
    )

    ft_config = FineTuneConfig(
        mode=run_id,
        blocks=blocks,
        image_size=p4_cfg.image_size,
        epochs=epochs,
        batch_size=p4_cfg.batch_size,
        learning_rate=p4_cfg.learning_rate,
        weight_decay=p4_cfg.weight_decay,
        seed=17,
        amp=False,
        device=str(device),
        backbone_checkpoint=str(BACKBONE_CHECKPOINT),
        head_checkpoint=str(stage1_checkpoint),
        split_manifest=str(run_dir / "split_manifest.json"),
        dataset_root=str(DATASET_ROOT),
    )

    history: list[dict[str, Any]] = []
    best_miou = float("-inf")
    best_eval: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        # In staged policy: freeze head for epochs 1-5, unfreeze for epochs 6-10
        if policy == "staged":
            if epoch <= 5:
                for p in head.parameters():
                    p.requires_grad = False
            else:
                for p in head.parameters():
                    p.requires_grad = True
            active_trainable = [(n, p) for n, p in trainable if p.requires_grad]
        else:
            active_trainable = trainable

        # Pure FP32 training
        backbone.train()
        head.train()
        total_loss = 0.0
        total_images = 0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features(backbone, images), masks.shape[-2:])
            loss = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)
            loss.backward()
            verify_finite_gradients(active_trainable, epoch)
            optimizer.step()
            total_loss += loss.item() * images.shape[0]
            total_images += images.shape[0]

        loss = total_loss / max(total_images, 1)
        val_eval = evaluate(backbone, head, val_loader, device)
        miou = val_eval["mean_iou"]

        print(
            f"Epoch {epoch:2d}/{epochs}: Loss={loss:.4f}, "
            f"Val mIoU={miou:.4f} (BG: {val_eval['per_class_iou']['background']:.4f}, Pet: {val_eval['per_class_iou']['pet']:.4f})"
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": loss,
            "validation": val_eval,
        }
        history.append(epoch_metrics)

        save_checkpoint(
            run_dir / "last_checkpoint.pt",
            backbone,
            head,
            ft_config,
            split_manifest,
            epoch,
            epoch_metrics,
            trainable_names,
        )

        if miou is not None and miou > best_miou:
            best_miou = miou
            best_eval = val_eval
            save_checkpoint(
                run_dir / "best_checkpoint.pt",
                backbone,
                head,
                ft_config,
                split_manifest,
                epoch,
                epoch_metrics,
                trainable_names,
            )

    # Load best checkpoint and measure latency
    best_ckpt = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    backbone.load_state_dict(best_ckpt["backbone_state_dict"], strict=True)
    head.load_state_dict(best_ckpt["head_state_dict"], strict=True)
    final_eval = evaluate(backbone, head, val_loader, device)

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    latency_stats = measure_latency(backbone, head, dummy_input, warmups=10, repeats=100)

    result_record = {
        "run_id": run_id,
        "run_name": run_name,
        "blocks_0_based": blocks,
        "blocks_1_based": [b + 1 for b in blocks],
        "policy": policy,
        "epochs": epochs,
        "best_epoch": best_ckpt["epoch"],
        "validation_miou": round(final_eval["mean_iou"], 6),
        "background_iou": round(final_eval["per_class_iou"]["background"], 6),
        "pet_iou": round(final_eval["per_class_iou"]["pet"], 6),
        "confusion_matrix": final_eval["confusion_matrix"],
        "median_latency_ms": round(latency_stats["median_ms"], 4),
        "p95_latency_ms": round(latency_stats["p95_ms"], 4),
        "p99_latency_ms": round(latency_stats["p99_ms"], 4),
        "history": history,
    }

    return result_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10B Experiment Suite")
    parser.add_argument("--epochs", type=int, default=10, help="Stage 2 epochs per ablation run")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    require(torch.cuda.is_available(), "CUDA is required for Phase 10B")
    device = torch.device("cuda:0")

    print(f"=== Phase 10B: Two-Stage Fine-Tuning and Controlled Ablation ===")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    # 1. Establish Stage 1 Checkpoint
    # In Stage 1, the downstream segmentation head is trained on DINOv2-Small baseline.
    # We save a pristine, audited Stage 1 checkpoint in phase10b_two_stage/
    p4_data = torch.load(HEAD_CHECKPOINT, map_location="cpu", weights_only=False)
    p4_cfg = RunConfig(**p4_data["run_config"])
    split_manifest = p4_data["split_manifest"]

    stage1_ckpt_path = args.output_dir / "stage1_baseline_head.pt"
    torch.save(
        {
            "stage": 1,
            "description": "Stage 1 Pretrained DINOv2-Small + Head Checkpoint",
            "head_state_dict": p4_data["head_state_dict"],
            "run_config": p4_data["run_config"],
            "split_manifest": split_manifest,
            "epoch": p4_data["epoch"],
            "metrics": p4_data["metrics"],
        },
        stage1_ckpt_path,
    )
    print(f"Stage 1 checkpoint saved to: {stage1_ckpt_path}")

    # Build datasets
    train_dataset, val_dataset, _ = build_datasets(
        DATASET_ROOT, p4_cfg, download=False, saved=split_manifest
    )
    train_loader = DataLoader(train_dataset, batch_size=p4_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=p4_cfg.batch_size, shuffle=False)

    # Evaluate Stage 1 Full Attention Baseline
    print("\n--- Evaluating Stage 1 Full Attention Baseline ---")
    base_backbone = build_backbone()
    strict_load_backbone(base_backbone)
    base_backbone = base_backbone.to(device).eval()
    base_head = SegmentationHead().to(device).eval()
    base_head.load_state_dict(p4_data["head_state_dict"], strict=True)

    base_eval = evaluate(base_backbone, base_head, val_loader, device)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    base_latency = measure_latency(base_backbone, base_head, dummy_input, warmups=10, repeats=100)

    results_matrix: list[dict[str, Any]] = []

    results_matrix.append(
        {
            "run_id": "run1_full_attention",
            "run_name": "Stage 1 Full Attention Baseline",
            "blocks_0_based": [],
            "blocks_1_based": [],
            "policy": "frozen_backbone",
            "epochs": p4_data["epoch"],
            "best_epoch": p4_data["epoch"],
            "validation_miou": round(base_eval["mean_iou"], 6),
            "background_iou": round(base_eval["per_class_iou"]["background"], 6),
            "pet_iou": round(base_eval["per_class_iou"]["pet"], 6),
            "confusion_matrix": base_eval["confusion_matrix"],
            "median_latency_ms": round(base_latency["median_ms"], 4),
            "p95_latency_ms": round(base_latency["p95_ms"], 4),
            "p99_latency_ms": round(base_latency["p99_ms"], 4),
            "history": [],
        }
    )

    # 2. Execute Ablation Sweep
    sweep_configs = [
        (
            "run2_manual_interleaved",
            "Manual Interleaved 4-Block [2, 5, 8, 11] (Joint)",
            [2, 5, 8, 11],
            "joint",
        ),
        (
            "run3_sigma_selected_joint",
            "Sigma-Selected 4-Block [1, 2, 3, 4] (Joint)",
            [1, 2, 3, 4],
            "joint",
        ),
        (
            "run4_sigma_selected_frozen_head",
            "Sigma-Selected 4-Block [1, 2, 3, 4] (Frozen Head)",
            [1, 2, 3, 4],
            "frozen_head",
        ),
        (
            "run5_sigma_selected_staged",
            "Sigma-Selected 4-Block [1, 2, 3, 4] (Staged Adapter)",
            [1, 2, 3, 4],
            "staged",
        ),
    ]

    for run_id, run_name, blocks, policy in sweep_configs:
        rec = run_stage2_finetune(
            run_id=run_id,
            run_name=run_name,
            blocks=blocks,
            policy=policy,
            epochs=args.epochs,
            stage1_checkpoint=stage1_ckpt_path,
            p4_cfg=p4_cfg,
            split_manifest=split_manifest,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            run_dir=args.output_dir / run_id,
        )
        results_matrix.append(rec)

    # 3. Export Comprehensive Artifacts
    # results.json
    (args.output_dir / "results.json").write_text(json.dumps(results_matrix, indent=2), encoding="utf-8")

    # comparison.csv
    csv_rows = []
    for r in results_matrix:
        csv_rows.append(
            {
                "run_id": r["run_id"],
                "run_name": r["run_name"],
                "blocks_0_based": str(r["blocks_0_based"]),
                "blocks_1_based": str(r["blocks_1_based"]),
                "policy": r["policy"],
                "epochs": r["epochs"],
                "validation_miou": r["validation_miou"],
                "background_iou": r["background_iou"],
                "pet_iou": r["pet_iou"],
                "median_latency_ms": r["median_latency_ms"],
                "p95_latency_ms": r["p95_latency_ms"],
                "p99_latency_ms": r["p99_latency_ms"],
            }
        )

    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # training_history.csv
    hist_rows = []
    for r in results_matrix:
        for h in r.get("history", []):
            hist_rows.append(
                {
                    "run_id": r["run_id"],
                    "epoch": h["epoch"],
                    "train_loss": round(h["train_loss"], 6),
                    "validation_miou": round(h["validation"]["mean_iou"], 6),
                    "background_iou": round(h["validation"]["per_class_iou"]["background"], 6),
                    "pet_iou": round(h["validation"]["per_class_iou"]["pet"], 6),
                }
            )

    if hist_rows:
        with (args.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(hist_rows[0].keys()))
            writer.writeheader()
            writer.writerows(hist_rows)

    # confusion_matrices.json
    conf_matrices = {r["run_id"]: r["confusion_matrix"] for r in results_matrix}
    (args.output_dir / "confusion_matrices.json").write_text(json.dumps(conf_matrices, indent=2), encoding="utf-8")

    print(f"\nAll Phase 10B runs completed successfully! Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
