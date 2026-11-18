"""Full BDD100K Fine-Tuning Pipeline for DeiT and EViT Model Configurations.

Trains all 4 vision models on the complete BDD100K driving scene dataset:
- deit_small_baseline
- evit_small_k09_fuse
- evit_small_k07_fuse
- evit_small_k05_fuse

Uses 69,634 training images and evaluates each epoch on 9,947 validation images.
The 19,902 test images are strictly held out for Phase 5 final evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.phase23b.bdd_dataset import (
    BDD_CLASSES,
    NUM_BDD_CLASSES,
    get_bdd_datasets,
    get_bdd_transforms,
    load_bdd_split_samples,
)
from src.phase23b.bdd_finetuning import compute_metrics
from src.phase23b.models import MODEL_CONFIGS, build_model, load_official_checkpoint

CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration", "full_dataset_checkpoints")
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "bdd100k", "annotations")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "bdd100k", "images")


class FastBDDDataset(Dataset):
    """High-throughput dataset leveraging libjpeg DCT downsampling and thread-safe batch collation."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        split_dir: str,
        transform: Optional[transforms.Compose] = None,
    ):
        self.samples = samples
        self.split_dir = split_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.samples[idx]


class FastBatchCollator:
    """Threaded collator that parallelizes PIL decoding and transforms across CPU cores."""

    def __init__(self, split_dir: str, transform: transforms.Compose, max_workers: int = 8):
        self.split_dir = split_dir
        self.transform = transform
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def _load_single(self, sample: Tuple[str, int]) -> Tuple[torch.Tensor, int]:
        image_name, class_id = sample
        img_path = os.path.join(self.split_dir, f"{image_name}.jpg")
        try:
            with Image.open(img_path) as img:
                img.draft("RGB", (512, 512))
                image = img.convert("RGB")
        except Exception:
            # Fallback for unexpected missing or corrupted file
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        tensor = self.transform(image)
        return tensor, class_id

    def __call__(self, batch: List[Tuple[str, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        results = list(self.pool.map(self._load_single, batch))
        imgs = torch.stack([r[0] for r in results])
        targets = torch.tensor([r[1] for r in results], dtype=torch.long)
        return imgs, targets

    def shutdown(self):
        self.pool.shutdown(wait=False)


def evaluate_model(
    model: nn.Module,
    samples: List[Tuple[str, int]],
    split_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers_load: int = 8,
) -> Tuple[float, float, float]:
    """Evaluate model on a dataset split returning (loss, top1_acc, macro_f1)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    transform = get_bdd_transforms(is_train=False, img_size=224)
    collator = FastBatchCollator(split_dir=split_dir, transform=transform, max_workers=num_workers_load)
    dataset = FastBDDDataset(samples=samples, split_dir=split_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().tolist())

    collator.shutdown()
    mean_loss = total_loss / len(all_targets) if all_targets else 0.0
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds), num_classes=NUM_BDD_CLASSES)
    return mean_loss, metrics["top1_accuracy"], metrics["macro_f1"]


def train_single_model(
    config_name: str,
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    train_dir: str,
    val_dir: str,
    device: str = "cuda",
    epochs: int = 3,
    batch_size: int = 64,
    lr_head: float = 3e-4,
    lr_backbone: float = 2e-5,
    seed: int = 42,
    checkpoint_dir: str = CHECKPOINTS_DIR,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Train a single DeiT/EViT model on full BDD100K driving scenes."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, f"{config_name}_best.pth")
    latest_ckpt_path = os.path.join(checkpoint_dir, f"{config_name}_latest.pth")

    # If already trained to completion and not forcing retrain, return existing result
    if os.path.isfile(best_ckpt_path) and not force_retrain:
        print(f"  Found existing best checkpoint at {best_ckpt_path}. Checking completion status...", flush=True)
        ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("epochs_trained", 0) >= epochs:
            print(f"  Model {config_name} already trained for {ckpt['epochs_trained']} epochs. Skipping.", flush=True)
            return ckpt.get("metadata", {
                "config_name": config_name,
                "best_val_accuracy": ckpt.get("best_val_acc"),
                "checkpoint_path": best_ckpt_path,
                "epochs_trained": ckpt.get("epochs_trained"),
            })

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n================================================================================", flush=True)
    print(f"  TRAINING: {config_name} | {len(train_samples):,} train | {len(val_samples):,} val", flush=True)
    print(f"  Epochs: {epochs} | Batch Size: {batch_size} | Head LR: {lr_head} | Backbone LR: {lr_backbone}", flush=True)
    print(f"================================================================================", flush=True)

    # Build model and load official pretrained weights
    model = build_model(config_name, num_classes=NUM_BDD_CLASSES, pretrained=False)
    load_official_checkpoint(model, config_name, device=device)
    model.to(device)

    # Optimizer setup: separate head and backbone
    head_params = list(model.head.parameters())
    head_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr_head, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()

    # Resume from latest checkpoint if available
    start_epoch = 1
    best_val_acc = -1.0
    history: List[Dict[str, Any]] = []

    if os.path.isfile(latest_ckpt_path) and not force_retrain:
        print(f"  Resuming from latest checkpoint: {latest_ckpt_path}...", flush=True)
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", -1.0)
        history = ckpt.get("history", [])
        print(f"  Resumed at epoch {start_epoch} (best val acc so far: {best_val_acc:.2f}%)", flush=True)

    train_transform = get_bdd_transforms(is_train=True, img_size=224)
    train_dataset = FastBDDDataset(samples=train_samples, split_dir=train_dir, transform=train_transform)

    train_t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.time()
        model.train()

        collator = FastBatchCollator(split_dir=train_dir, transform=train_transform, max_workers=8)
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=False,
        )

        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(loader)

        for step, (images, targets) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * targets.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += int((preds == targets).sum().item())
            total += targets.size(0)

            if step % 200 == 0 or step == num_batches:
                step_loss = running_loss / total
                step_acc = (correct / total) * 100.0
                elapsed = time.time() - epoch_t0
                throughput = total / elapsed if elapsed > 0 else 0.0
                print(
                    f"  Epoch {epoch:02d}/{epochs:02d} | Step {step:04d}/{num_batches:04d} "
                    f"| Loss: {step_loss:.4f} | Acc: {step_acc:.2f}% | Speed: {throughput:.1f} samples/s",
                    flush=True,
                )

        collator.shutdown()
        scheduler.step()

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = (correct / total) * 100.0 if total > 0 else 0.0
        epoch_train_time = time.time() - epoch_t0

        # Validate on full validation split
        print(f"  Evaluating epoch {epoch:02d} on {len(val_samples):,} validation images...", flush=True)
        val_t0 = time.time()
        val_loss, val_acc, val_f1 = evaluate_model(
            model=model,
            samples=val_samples,
            split_dir=val_dir,
            device=device,
            batch_size=batch_size,
            num_workers_load=8,
        )
        val_time = time.time() - val_t0

        print(
            f"  >>> Epoch {epoch:02d} Complete: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% "
            f"| Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val F1={val_f1:.2f}% "
            f"| Train Time={epoch_train_time:.1f}s, Val Time={val_time:.1f}s",
            flush=True,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "val_macro_f1": round(val_f1, 2),
            "train_time_s": round(epoch_train_time, 1),
            "val_time_s": round(val_time, 1),
        }
        history.append(epoch_record)

        # Save latest checkpoint
        torch.save({
            "config_name": config_name,
            "epoch": epoch,
            "best_val_acc": max(best_val_acc, val_acc),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
        }, latest_ckpt_path)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  *** New best validation accuracy: {best_val_acc:.2f}%! Saving {best_ckpt_path} ***", flush=True)
            torch.save({
                "config_name": config_name,
                "dataset": "BDD100K",
                "task": "driving_scene_classification",
                "classes": BDD_CLASSES,
                "num_classes": NUM_BDD_CLASSES,
                "best_epoch": epoch,
                "epochs_trained": epoch,
                "best_val_acc": round(best_val_acc, 2),
                "state_dict": copy.deepcopy(model.state_dict()),
                "history": history,
                "metadata": {
                    "config_name": config_name,
                    "best_val_accuracy": round(best_val_acc, 2),
                    "best_val_macro_f1": round(val_f1, 2),
                    "best_epoch": epoch,
                    "epochs_trained": epoch,
                    "checkpoint_path": best_ckpt_path,
                },
            }, best_ckpt_path)

    total_train_time = round(time.time() - train_t0, 1)
    print(f"  Training for {config_name} finished in {total_train_time}s. Best Val Acc: {best_val_acc:.2f}%", flush=True)

    result_summary = {
        "config_name": config_name,
        "best_val_accuracy": round(best_val_acc, 2),
        "best_epoch": next((h["epoch"] for h in history if h["val_acc"] == best_val_acc), epochs),
        "epochs_trained": epochs,
        "total_training_time_s": total_train_time,
        "checkpoint_path": best_ckpt_path,
        "history": history,
    }
    return result_summary


def main():
    parser = argparse.ArgumentParser(description="Train DeiT/EViT models on full BDD100K dataset.")
    parser.add_argument("--model", type=str, default="all", choices=list(MODEL_CONFIGS.keys()) + ["all"],
                        help="Specific model to train, or 'all'.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--smoke-test", action="store_true", help="Run a fast smoke test with 500 samples.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if checkpoint exists.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("================================================================================")
    print(" BDD100K Full-Dataset Training Pipeline (6 Driving Scene Categories)")
    print(f" Device: {device} | PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" GPU Device: {torch.cuda.get_device_name(0)} (Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)")
    print(f" Smoke Test: {args.smoke_test} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("================================================================================")

    # Load annotations
    train_manifest = os.path.join(ANNOTATIONS_DIR, "train_full.json")
    val_manifest = os.path.join(ANNOTATIONS_DIR, "val_full.json")
    train_samples = load_bdd_split_samples(train_manifest)
    val_samples = load_bdd_split_samples(val_manifest)

    train_dir = os.path.join(IMAGES_DIR, "100k", "train")
    val_dir = os.path.join(IMAGES_DIR, "100k", "val")

    if args.smoke_test:
        print("\n>>> SMOKE TEST MODE: Subsetting to 500 train samples and 200 val samples...")
        train_samples = train_samples[:500]
        val_samples = val_samples[:200]
        ckpt_dir = os.path.join(CHECKPOINTS_DIR, "smoke_test")
    else:
        ckpt_dir = CHECKPOINTS_DIR

    print(f"Loaded {len(train_samples):,} training samples from {train_manifest}")
    print(f"Loaded {len(val_samples):,} validation samples from {val_manifest}")

    models_to_train = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    results: Dict[str, Any] = {}
    overall_t0 = time.time()

    for model_name in models_to_train:
        res = train_single_model(
            config_name=model_name,
            train_samples=train_samples,
            val_samples=val_samples,
            train_dir=train_dir,
            val_dir=val_dir,
            device=device,
            epochs=args.epochs if not args.smoke_test else 1,
            batch_size=args.batch_size,
            checkpoint_dir=ckpt_dir,
            force_retrain=args.force_retrain,
        )
        results[model_name] = res

    overall_time = round(time.time() - overall_t0, 1)
    print("\n================================================================================")
    print(f" ALL MODELS TRAINED SUCCESSFULLY IN {overall_time}s!")
    print("================================================================================")
    for name, r in results.items():
        print(f"  {name:22s} -> Best Val Acc: {r.get('best_val_accuracy', 0.0):.2f}% (Epoch {r.get('best_epoch', 0)}) | Saved: {r.get('checkpoint_path')}")

    # Write training metadata JSON
    meta_path = os.path.join(ckpt_dir, "training_metadata.json")
    metadata_record = {
        "dataset": "BDD100K",
        "task": "6-class driving scene classification",
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "classes": BDD_CLASSES,
        "epochs": args.epochs if not args.smoke_test else 1,
        "batch_size": args.batch_size,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "smoke_test": args.smoke_test,
        "total_time_seconds": overall_time,
        "models": results,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_record, f, indent=2)
    print(f"\nSaved training metadata to {meta_path}")


if __name__ == "__main__":
    main()
