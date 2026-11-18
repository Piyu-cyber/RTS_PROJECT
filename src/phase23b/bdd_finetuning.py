"""Fine-tuning and evaluation on BDD100K 6-class driving scene classification."""

from __future__ import annotations

import os
import copy
import time
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .models import build_model, load_official_checkpoint, MODEL_CONFIGS
from .bdd_dataset import get_bdd_datasets, NUM_BDD_CLASSES, BDD_CLASSES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_BDD_CLASSES) -> Dict[str, float]:
    """Calculate Top-1 Accuracy and Macro-F1 score without external scikit-learn dependency."""
    total = len(y_true)
    top1 = float(np.mean(y_true == y_pred)) * 100.0 if total > 0 else 0.0

    # Macro F1
    f1_scores = []
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores)) * 100.0 if f1_scores else 0.0

    return {
        "top1_accuracy": round(top1, 2),
        "macro_f1": round(macro_f1, 2),
    }


def evaluate_bdd_split(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    num_classes: int = NUM_BDD_CLASSES,
) -> Tuple[float, float, float]:
    """Evaluate model on a dataset split, returning (loss, top1_acc, macro_f1)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    mean_loss = total_loss / len(all_targets) if all_targets else 0.0
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds), num_classes=num_classes)
    return mean_loss, metrics["top1_accuracy"], metrics["macro_f1"]


def finetune_bdd_model(
    config_name: str,
    dataset_root: str = "data/bdd100k",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    epochs: int = 5,
    batch_size: int = 16,
    lr_head: float = 1e-4,
    lr_backbone: float = 1e-5,
    output_dir: str = "outputs/bdd100k_migration/checkpoints",
    seed: int = 42,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Fine-tune a pretrained DeiT/EViT model on BDD100K for 6 driving scene classes."""
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    print(f"\n--- Fine-tuning {config_name} on BDD100K ({NUM_BDD_CLASSES} driving scenes) ---", flush=True)
    train_ds, val_ds, test_ds, meta = get_bdd_datasets(dataset_root=dataset_root, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    ckpt_save_path = os.path.join(output_dir, f"{config_name}_bdd_best.pth")

    # If checkpoint already exists with required epochs and force_retrain is False, evaluate directly
    if os.path.isfile(ckpt_save_path) and not force_retrain:
        ckpt_data = torch.load(ckpt_save_path, map_location="cpu", weights_only=False)
        if ckpt_data.get("best_epoch", 0) >= epochs:
            print(f"  Existing complete checkpoint found at {ckpt_save_path}. Evaluating test set...", flush=True)
            model = build_model(config_name, num_classes=NUM_BDD_CLASSES, pretrained=False)
            model.load_state_dict(ckpt_data["state_dict"])
            model.to(device)

            test_loss, test_top1, test_f1 = evaluate_bdd_split(model, test_loader, device=device)
            val_loss, val_top1, val_f1 = evaluate_bdd_split(model, val_loader, device=device)
            print(f"  Val Top-1: {val_top1:.2f}% | Test Top-1: {test_top1:.2f}% | Test Macro-F1: {test_f1:.2f}%", flush=True)
            return {
                "config_name": config_name,
                "best_val_accuracy": val_top1,
                "test_top1_accuracy": test_top1,
                "test_macro_f1": test_f1,
                "checkpoint_path": ckpt_save_path,
                "epochs_trained": ckpt_data.get("best_epoch", epochs),
            }

    # Build model with 6 classes
    model = build_model(config_name, num_classes=NUM_BDD_CLASSES, pretrained=False)

    # Load official ImageNet backbone weights
    print(f"  Loading official backbone weights for {config_name}...", flush=True)
    load_official_checkpoint(model, config_name, device=device)
    model.to(device)

    # Optimizer: differential learning rates for head and backbone
    head_params = list(model.head.parameters())
    head_param_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr_head, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state_dict = None
    best_epoch = 0

    train_t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += int((preds == targets).sum().item())
            total += targets.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        val_loss, val_acc, val_f1 = evaluate_bdd_split(model, val_loader, device=device)
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% F1: {val_f1:.2f}%", flush=True)

        if val_acc > best_val_acc or best_state_dict is None:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

    # Save best checkpoint
    torch.save({
        "config_name": config_name,
        "dataset": "BDD100K",
        "task": "driving_scene_classification",
        "classes": BDD_CLASSES,
        "num_classes": NUM_BDD_CLASSES,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "state_dict": best_state_dict,
    }, ckpt_save_path)
    print(f"  Saved best BDD100K checkpoint to {ckpt_save_path} (Val Acc: {best_val_acc:.2f}%)", flush=True)

    # Evaluate test split with best model
    model.load_state_dict(best_state_dict)
    test_loss, test_acc, test_f1 = evaluate_bdd_split(model, test_loader, device=device)
    print(f"  Final Test Evaluation -> Top-1 Acc: {test_acc:.2f}% | Macro-F1: {test_f1:.2f}%", flush=True)

    return {
        "config_name": config_name,
        "best_val_accuracy": round(best_val_acc, 2),
        "test_top1_accuracy": round(test_acc, 2),
        "test_macro_f1": round(test_f1, 2),
        "checkpoint_path": ckpt_save_path,
        "epochs_trained": epochs,
        "training_time_s": round(time.time() - train_t0, 1),
    }
