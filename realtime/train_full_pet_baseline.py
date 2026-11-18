"""Stage 5: Train Full-Attention Baseline on Full Oxford-IIIT Pet Dataset (Phase 12A)."""
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
    FullOxfordPetDataset,
    load_dataset_from_manifest,
)
from baseline.train_segmentation import (
    SegmentationHead,
    build_backbone,
    colorize,
    features,
    save_visualization,
    strict_load_backbone,
)


@dataclass(frozen=True)
class FullPetConfig:
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 17
    freeze_backbone: bool = True
    amp: bool = True
    num_workers: int = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate_full(
    backbone: Any,
    head: SegmentationHead,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    backbone.eval()
    head.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    images_seen = 0
    val_loss_sum = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        feat = features(backbone, images)
        logits = head(feat, targets.shape[-2:])
        loss = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)
        val_loss_sum += loss.item() * images.shape[0]

        prediction = logits.argmax(1)
        valid = targets != IGNORE_INDEX
        encoded = targets[valid] * NUM_CLASSES + prediction[valid]
        confusion += torch.bincount(encoded.cpu(), minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
        images_seen += images.shape[0]

    per_class: dict[str, float | None] = {}
    available: list[float] = []
    total_tp = 0
    total_valid_pixels = confusion.sum().item()

    for i, name in enumerate(CLASS_NAMES):
        tp = confusion[i, i].item()
        total_tp += tp
        union = tp + confusion[:, i].sum().item() - tp + confusion[i, :].sum().item() - tp
        score = None if union == 0 else tp / union
        per_class[name] = score
        if score is not None:
            available.append(score)

    pixel_acc = total_tp / total_valid_pixels if total_valid_pixels > 0 else 0.0

    return {
        "evaluated_images": images_seen,
        "val_loss": val_loss_sum / images_seen if images_seen > 0 else 0.0,
        "mean_iou": None if not available else sum(available) / len(available),
        "pixel_accuracy": pixel_acc,
        "per_class_iou": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def run_training(cfg: FullPetConfig, output_dir: Path, manifest_path: Path) -> dict[str, Any]:
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

    # 2. Build backbone and segmentation head
    backbone = build_backbone()
    strict_load_backbone(backbone)
    backbone = backbone.to(device).eval()
    if cfg.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    head = SegmentationHead().to(device)
    trainable = [p for p in head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    # Save experiment config
    exp_config = {
        "task": "Oxford-IIIT Pet binary segmentation (Full Dataset)",
        "run_config": asdict(cfg),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "class_names": CLASS_NAMES,
        "ignore_index": IGNORE_INDEX,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "checkpoint_selection_rule": "Highest validation mean_iou; ties broken by lowest validation loss, then earlier epoch",
    }
    (output_dir.parent / "experiment_config.json").write_text(json.dumps(exp_config, indent=2), encoding="utf-8")

    print(f"Starting baseline training on {device} ({torch.cuda.get_device_name(0)})")
    print(f"Dataset: train={len(train_ds)}, val={len(val_ds)}, batch_size={cfg.batch_size}, epochs={cfg.epochs}")

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
            head.train()
            train_loss_sum = 0.0
            samples_seen = 0

            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=cfg.amp):
                    with torch.no_grad():
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
                f"Epoch {epoch}/{cfg.epochs} [{epoch_duration:.1f}s]: "
                f"train_loss={epoch_train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_mIoU={mIoU:.4f} (bg={bg_iou:.4f}, pet={pet_iou:.4f}, acc={pixel_acc:.4f})"
            )

            # Checkpoint selection: higher mIoU, or lower val_loss on tie
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
                        "head_state_dict": head.state_dict(),
                        "run_config": asdict(cfg),
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
            "head_state_dict": head.state_dict(),
            "run_config": asdict(cfg),
            "epoch": cfg.epochs,
            "validation": history[-1],
            "class_names": CLASS_NAMES,
        },
        last_checkpoint_path,
    )

    # Evaluate best checkpoint independently
    best_payload = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    head.load_state_dict(best_payload["head_state_dict"], strict=True)
    final_eval = evaluate_full(backbone, head, val_loader, device)

    # Save sample visualization
    save_visualization(backbone, head, val_ds, device, output_dir)

    metrics_payload = {
        "task": "Oxford-IIIT Pet binary segmentation (Full Dataset)",
        "run_config": asdict(cfg),
        "dataset_split": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        },
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "total_runtime_seconds": round(total_training_time, 2),
        "peak_gpu_memory_mib": round(peak_gpu_mem, 2),
        "best_epoch": best_epoch,
        "best_validation": final_eval,
        "checkpoint_selection_rule": exp_config["checkpoint_selection_rule"],
        "history": history,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"\nTraining completed in {total_training_time:.2f}s! Peak GPU Mem: {peak_gpu_mem:.1f} MiB")
    print(f"Best Epoch: {best_epoch}, Best mIoU: {final_eval['mean_iou']:.4f}, Background IoU: {final_eval['per_class_iou']['background']:.4f}, Pet IoU: {final_eval['per_class_iou']['pet']:.4f}")
    return metrics_payload


def main() -> None:
    manifest_p = PROJECT_ROOT / "outputs" / "phase12a_full_pet" / "split_manifest.json"
    out_dir = PROJECT_ROOT / "outputs" / "phase12a_full_pet" / "baseline"
    cfg = FullPetConfig(batch_size=16, epochs=5, num_workers=2)
    run_training(cfg, out_dir, manifest_p)


if __name__ == "__main__":
    main()
