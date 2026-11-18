"""Stage 5: Train and fine-tune DWConv variants under full Oxford-IIIT Pet dataset."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ["XFORMERS_DISABLED"] = "true"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "reference"))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.oxford_pet_loader import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    load_dataset_from_manifest,
)
from baseline.train_segmentation import (
    SegmentationHead,
    build_backbone,
    features,
    save_visualization,
    strict_load_backbone,
)
from finetuning.finetune_dwconv import patch_and_verify
from realtime.train_full_pet_baseline import evaluate_full


@dataclass(frozen=True)
class DWConvTrainConfig:
    variant_name: str
    blocks: list[int]
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 17
    amp: bool = True
    num_workers: int = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_dwconv_finetuning(
    cfg: DWConvTrainConfig,
    stage1_head_checkpoint: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU required for training")

    set_seed(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load full datasets from split manifest
    train_ds, val_ds, manifest = load_dataset_from_manifest(
        manifest_path,
        PROJECT_ROOT,
        image_size=cfg.image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # 2. Build backbone and strictly patch targeted DWConv blocks
    backbone = build_backbone()
    strict_load_backbone(backbone)
    patch_and_verify(backbone, cfg.blocks, device)

    # 3. Build segmentation head and load Stage 1 weights
    head = SegmentationHead().to(device)
    stage1_payload = torch.load(stage1_head_checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(stage1_payload["head_state_dict"], strict=True)
    print(f"Loaded Stage 1 head weights from {stage1_head_checkpoint}")

    # Save Stage 1 & converted initial checkpoint
    stage1_out = output_dir / "stage1_head_checkpoint.pt"
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "source": str(stage1_head_checkpoint),
            "description": "Stage 1 trained head before DWConv tuning",
        },
        stage1_out,
    )

    converted_out = output_dir / "converted_untuned_checkpoint.pt"
    torch.save(
        {
            "backbone_state_dict": backbone.state_dict(),
            "head_state_dict": head.state_dict(),
            "blocks": cfg.blocks,
            "description": "Pretrained weights converted with DWConv prior to Stage 2 tuning",
        },
        converted_out,
    )

    # Joint fine-tuning: all patched backbone parameters and head parameters are trainable
    backbone.train()
    head.train()
    trainable = [p for p in list(backbone.parameters()) + list(head.parameters()) if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    print(f"Starting Stage 2 joint fine-tuning for {cfg.variant_name} with blocks {cfg.blocks}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    torch.cuda.reset_peak_memory_stats(device)
    start_total_time = time.perf_counter()

    best_mIoU = float("-inf")
    best_val_loss = float("inf")
    best_epoch = -1
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    history: list[dict[str, Any]] = []

    csv_path = output_dir / "training_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_mean_iou",
            "val_pixel_acc",
            "val_bg_iou",
            "val_pet_iou",
            "epoch_duration_sec",
        ])

        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.perf_counter()
            backbone.train()
            head.train()
            train_loss_sum = 0.0
            samples_seen = 0

            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=cfg.amp):
                    feat = features(backbone, images)
                    logits = head(feat, targets.shape[-2:])
                    loss = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += loss.item() * images.shape[0]
                samples_seen += images.shape[0]

            epoch_train_loss = train_loss_sum / samples_seen

            # Validation
            val_results = evaluate_full(backbone, head, val_loader, device)
            epoch_duration = time.perf_counter() - epoch_start
            mIoU = val_results["mean_iou"] or 0.0
            val_loss = val_results["val_loss"]
            bg_iou = val_results["per_class_iou"]["background"] or 0.0
            pet_iou = val_results["per_class_iou"]["pet"] or 0.0
            pixel_acc = val_results["pixel_accuracy"]

            writer.writerow([
                epoch,
                round(epoch_train_loss, 6),
                round(val_loss, 6),
                round(mIoU, 6),
                round(pixel_acc, 6),
                round(bg_iou, 6),
                round(pet_iou, 6),
                round(epoch_duration, 2),
            ])
            f_csv.flush()

            epoch_record = {
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "val_loss": val_loss,
                "val_mean_iou": mIoU,
                "val_pixel_acc": pixel_acc,
                "val_per_class_iou": val_results["per_class_iou"],
                "confusion_matrix": val_results["confusion_matrix"],
                "duration_seconds": epoch_duration,
            }
            history.append(epoch_record)

            print(
                f"[{cfg.variant_name}] Epoch {epoch}/{cfg.epochs} [{epoch_duration:.1f}s]: "
                f"train_loss={epoch_train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_mIoU={mIoU:.4f} (bg={bg_iou:.4f}, pet={pet_iou:.4f}, acc={pixel_acc:.4f})"
            )

            is_better = False
            if mIoU > best_mIoU + 1e-6:
                is_better = True
            elif abs(mIoU - best_mIoU) <= 1e-6 and val_loss < best_val_loss:
                is_better = True

            if is_better:
                best_mIoU = mIoU
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(
                    {
                        "backbone_state_dict": backbone.state_dict(),
                        "head_state_dict": head.state_dict(),
                        "blocks": cfg.blocks,
                        "variant": cfg.variant_name,
                        "epoch": epoch,
                        "best_validation": val_results,
                        "class_names": CLASS_NAMES,
                    },
                    best_checkpoint_path,
                )
                print(f" -> Best checkpoint updated at epoch {epoch}: mIoU={best_mIoU:.4f}")

    total_training_time = time.perf_counter() - start_total_time
    peak_gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Save last checkpoint
    torch.save(
        {
            "backbone_state_dict": backbone.state_dict(),
            "head_state_dict": head.state_dict(),
            "blocks": cfg.blocks,
            "variant": cfg.variant_name,
            "epoch": cfg.epochs,
            "validation": history[-1],
            "class_names": CLASS_NAMES,
        },
        last_checkpoint_path,
    )

    # Re-evaluate best checkpoint
    best_payload = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    backbone.load_state_dict(best_payload["backbone_state_dict"], strict=True)
    head.load_state_dict(best_payload["head_state_dict"], strict=True)
    final_eval = evaluate_full(backbone, head, val_loader, device)

    save_visualization(backbone, head, val_ds, device, output_dir)

    metrics_payload = {
        "variant_name": cfg.variant_name,
        "blocks": cfg.blocks,
        "run_config": asdict(cfg),
        "total_runtime_seconds": round(total_training_time, 2),
        "peak_gpu_memory_mib": round(peak_gpu_mem, 2),
        "best_epoch": best_epoch,
        "best_validation": final_eval,
        "checkpoint_selection_rule": "Highest validation mean_iou; ties broken by lowest validation loss, then earlier epoch",
        "history": history,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"\n{cfg.variant_name} completed in {total_training_time:.2f}s! Peak GPU Mem: {peak_gpu_mem:.1f} MiB")
    print(f"Best Epoch: {best_epoch}, Best mIoU: {final_eval['mean_iou']:.4f}, Background IoU: {final_eval['per_class_iou']['background']:.4f}, Pet IoU: {final_eval['per_class_iou']['pet']:.4f}")
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("manual", "sigma"), required=True)
    args = parser.parse_args()

    manifest_p = PROJECT_ROOT / "outputs" / "phase12a_full_pet" / "split_manifest.json"
    stage1_head = PROJECT_ROOT / "outputs" / "phase12a_full_pet" / "baseline" / "best_checkpoint.pt"

    if args.variant == "manual":
        cfg = DWConvTrainConfig(variant_name="manual_interleaved", blocks=[2, 5, 8, 11], epochs=5)
        out_dir = PROJECT_ROOT / "outputs" / "phase12b_dwconv" / "manual"
    else:
        cfg = DWConvTrainConfig(variant_name="sigma_selected", blocks=[1, 2, 3, 4], epochs=5)
        out_dir = PROJECT_ROOT / "outputs" / "phase12b_dwconv" / "sigma_selected"

    run_dwconv_finetuning(cfg, stage1_head, out_dir, manifest_p)


if __name__ == "__main__":
    main()
