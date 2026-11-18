"""Controlled Matched-Training Experiment Suite for EViT Minority-Class Representation.

Implements the 7-arm benchmark matrix:
- M0: Full DeiT-Small baseline (Standard CE)
- M1: EViT-Small k05 baseline (Standard CE)
- M2: EViT-Small k05 + Class-Balanced CE (Cui et al., CVPR 2019)
- M3: EViT-Small k05 + Standard Knowledge Distillation (Hinton et al., 2015)
- M4: EViT-Small k05 + Decoupled Knowledge Distillation (Zhao et al., CVPR 2022)
- M5: EViT-Small k05 + Feature Representation Consistency (Pre-classifier [CLS])
- M6: EViT-Small k05 + Class-Margin-Preserving Regularization (CMPR Margin Loss)

All arms strictly share:
- 3 matched training epochs
- Identical BDD100K train (69,634), val (9,947), and test (19,902) splits
- Input resolution 224x224, RandAugment, AdamW (3e-4 head / 2e-5 backbone), cosine decay
- Checkpoint selection based strictly on validation Macro-F1
- Isolated outputs under outputs/cmpr_matched_training/
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.phase23b.bdd_dataset import (
    BDD_CLASSES,
    NUM_BDD_CLASSES,
    get_bdd_transforms,
    load_bdd_split_samples,
)
from src.phase23b.bdd_finetuning import compute_metrics
from src.phase23b.models import MODEL_CONFIGS, build_model, load_official_checkpoint

MATCHED_DIR = os.path.join(PROJECT_ROOT, "outputs", "cmpr_matched_training")
CHECKPOINTS_DIR = os.path.join(MATCHED_DIR, "checkpoints")
CONFIGS_DIR = os.path.join(MATCHED_DIR, "training_configurations")
LOGS_DIR = os.path.join(MATCHED_DIR, "training_logs")
CM_DIR = os.path.join(MATCHED_DIR, "confusion_matrices")
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "bdd100k", "annotations")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "bdd100k", "images")

for d in [CHECKPOINTS_DIR, CONFIGS_DIR, LOGS_DIR, CM_DIR]:
    os.makedirs(d, exist_ok=True)


class FastBDDDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], split_dir: str):
        self.samples = samples
        self.split_dir = split_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.samples[idx]


class FastBatchCollator:
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


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_class_balanced_weights(device: torch.device) -> torch.Tensor:
    """Computes bounded, smoothed inverse-frequency weights for BDD100K driving scene classes."""
    # Training counts: City(43158), Highway(17816), Residential(8409), Parking(382), Tunnel(77), Gas(21)
    counts = np.array([43158, 17816, 8409, 382, 77, 21], dtype=np.float32)
    N = float(np.sum(counts))
    C = float(len(counts))
    raw_weights = (N / (C * counts)) ** 0.25
    norm_weights = raw_weights / np.mean(raw_weights)
    return torch.tensor(norm_weights, dtype=torch.float32, device=device)


def compute_dkd_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 2.0,
    temperature: float = 3.0,
) -> torch.Tensor:
    """Decoupled Knowledge Distillation (Zhao et al., CVPR 2022)."""
    logits_s = logits_s.float()
    logits_t = logits_t.float()
    gt_mask = torch.zeros_like(logits_s, dtype=torch.bool).scatter_(1, target.unsqueeze(1), True)
    
    # Target Class Knowledge Distillation (TCKD)
    s_prob = F.softmax(logits_s / temperature, dim=1)
    t_prob = F.softmax(logits_t / temperature, dim=1)
    s_target = s_prob.gather(1, target.unsqueeze(1)).squeeze(1)
    t_target = t_prob.gather(1, target.unsqueeze(1)).squeeze(1)
    
    p_t = torch.stack([t_target, 1.0 - t_target], dim=1)
    p_s = torch.stack([s_target, 1.0 - s_target], dim=1)
    tckd_loss = F.kl_div(p_s.log(), p_t, reduction="batchmean") * (temperature ** 2)
    
    # Non-Target Class Knowledge Distillation (NCKD)
    s_other_logits = logits_s.masked_fill(gt_mask, -1e4)
    t_other_logits = logits_t.masked_fill(gt_mask, -1e4)
    s_other_log_prob = F.log_softmax(s_other_logits / temperature, dim=1)
    t_other_prob = F.softmax(t_other_logits / temperature, dim=1)
    
    nckd_loss = (t_other_prob * (t_other_prob.add(1e-9).log() - s_other_log_prob)).masked_fill(gt_mask, 0.0).sum(dim=1).mean() * (temperature ** 2)
    return alpha * tckd_loss + beta * nckd_loss


def compute_cmpr_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 0.0,
    teacher_gate: bool = True,
) -> torch.Tensor:
    """Class-Margin-Preserving Regularization loss."""
    logits_s = logits_s.float()
    logits_t = logits_t.float()
    mask = torch.zeros_like(logits_s, dtype=torch.bool).scatter_(1, target.unsqueeze(1), True)
    s_true = logits_s.gather(1, target.unsqueeze(1)).squeeze(1)
    t_true = logits_t.gather(1, target.unsqueeze(1)).squeeze(1)
    
    s_other_max = logits_s.masked_fill(mask, -1e4).max(dim=1).values
    t_other_max = logits_t.masked_fill(mask, -1e4).max(dim=1).values
    
    m_s = s_true - s_other_max
    m_t = t_true - t_other_max
    
    margin_diff = m_t - m_s - gamma
    hinge = torch.clamp(margin_diff, min=0.0) ** 2
    
    if teacher_gate:
        t_pred = logits_t.argmax(dim=1)
        gate = (t_pred == target).float()
        loss = (gate * hinge).sum() / (gate.sum() + 1e-6)
    else:
        loss = hinge.mean()
    return loss


def evaluate_split(
    model: nn.Module,
    samples: List[Tuple[str, int]],
    split_dir: str,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Evaluates model returning loss, accuracy, Macro-F1, per-class metrics, and confusion matrix."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    transform = get_bdd_transforms(is_train=False, img_size=224)
    collator = FastBatchCollator(split_dir=split_dir, transform=transform, max_workers=8)
    dataset = FastBDDDataset(samples=samples, split_dir=split_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_logits: List[List[float]] = []

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
            all_logits.extend(outputs.cpu().float().tolist())

    collator.shutdown()
    mean_loss = total_loss / len(all_targets) if all_targets else 0.0
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    # Confusion matrix
    cm = np.zeros((NUM_BDD_CLASSES, NUM_BDD_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    per_class = []
    f1s, precs, recs = [], [], []
    for c in range(NUM_BDD_CLASSES):
        tp = int(cm[c, c])
        fn = int(np.sum(cm[c, :]) - tp)
        fp = int(np.sum(cm[:, c]) - tp)
        supp = tp + fn
        rec = (tp / supp * 100.0) if supp > 0 else 0.0
        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class.append({
            "class_id": c,
            "class_name": BDD_CLASSES[c],
            "support": supp,
            "recall": round(rec, 2),
            "precision": round(prec, 2),
            "f1": round(f1, 2),
        })
        recs.append(rec)
        precs.append(prec)
        f1s.append(f1)
        
    top1 = float(np.mean(y_true == y_pred)) * 100.0 if len(y_true) > 0 else 0.0
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.average(f1s, weights=[p["support"] for p in per_class]))
    
    return {
        "loss": round(mean_loss, 4),
        "top1_accuracy": round(top1, 2),
        "macro_f1": round(macro_f1, 2),
        "weighted_f1": round(weighted_f1, 2),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "targets": all_targets,
        "logits": all_logits,
    }


def train_and_evaluate_arm(
    arm_id: str,
    arch_config: str,
    loss_type: str,
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 64,
    lr_head: float = 3e-4,
    lr_backbone: float = 2e-5,
    teacher_model: Optional[nn.Module] = None,
    teacher_ckpt_hash: Optional[str] = None,
    device: str = "cuda",
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Executes a single matched experiment arm."""
    dev = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_name = f"{arm_id}_seed{seed}" if not smoke_test else f"{arm_id}_seed{seed}_smoke"
    best_ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{run_name}_best.pth")
    final_ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{run_name}_final.pth")
    config_save_path = os.path.join(CONFIGS_DIR, f"{run_name}_config.json")
    log_save_path = os.path.join(LOGS_DIR, f"{run_name}_history.json")

    print(f"\n================================================================================", flush=True)
    print(f"  LAUNCHING EXPERIMENT ARM: {arm_id} | Run: {run_name}", flush=True)
    print(f"  Architecture: {arch_config} | Loss Type: {loss_type} | Seed: {seed} | Epochs: {epochs} | Smoke: {smoke_test}", flush=True)
    print(f"================================================================================", flush=True)

    # Save run configuration
    run_config = {
        "arm_id": arm_id,
        "run_name": run_name,
        "arch_config": arch_config,
        "loss_type": loss_type,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_head": lr_head,
        "lr_backbone": lr_backbone,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "teacher_checkpoint_hash": teacher_ckpt_hash,
        "smoke_test": smoke_test,
    }
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    # Load annotations
    train_samples = load_bdd_split_samples(os.path.join(ANNOTATIONS_DIR, "train_full.json"))
    val_samples = load_bdd_split_samples(os.path.join(ANNOTATIONS_DIR, "val_full.json"))
    test_samples = load_bdd_split_samples(os.path.join(ANNOTATIONS_DIR, "test_full.json"))

    if smoke_test:
        train_samples = train_samples[:300]
        val_samples = val_samples[:100]
        test_samples = test_samples[:100]
        epochs = 1

    train_dir = os.path.join(IMAGES_DIR, "100k", "train")
    val_dir = os.path.join(IMAGES_DIR, "100k", "val")
    test_dir = os.path.join(IMAGES_DIR, "100k", "test")

    # Build model & load official pretrained weights
    model = build_model(arch_config, num_classes=NUM_BDD_CLASSES, pretrained=False)
    load_official_checkpoint(model, arch_config, device=device)
    model.to(dev)

    # Optimizer
    head_params = list(model.head.parameters())
    head_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr_head, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # Loss components
    ce_loss_fn = nn.CrossEntropyLoss()
    cb_weights = get_class_balanced_weights(dev) if loss_type in ["cb_ce", "cb_ce_cmpr"] else None
    cb_ce_loss_fn = nn.CrossEntropyLoss(weight=cb_weights) if loss_type in ["cb_ce", "cb_ce_cmpr"] else None

    train_transform = get_bdd_transforms(is_train=True, img_size=224)
    train_dataset = FastBDDDataset(samples=train_samples, split_dir=train_dir)

    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = 1

    train_start_t = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start_t = time.time()
        model.train()
        collator = FastBatchCollator(split_dir=train_dir, transform=train_transform, max_workers=8)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, drop_last=False)

        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(loader)

        for step, (images, targets) in enumerate(loader, start=1):
            images = images.to(dev, non_blocking=True)
            targets = targets.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                if loss_type == "feature" and teacher_model is not None:
                    # Forward features to capture pre-classifier [CLS]
                    f_s = model.forward_features(images)
                    feat_s = f_s[0] if isinstance(f_s, tuple) else f_s
                    outputs_s = model.head(feat_s)
                    with torch.no_grad():
                        f_t = teacher_model.forward_features(images)
                        feat_t = f_t[0] if isinstance(f_t, tuple) else f_t
                    loss_ce = ce_loss_fn(outputs_s, targets)
                    loss_feat = (1.0 - F.cosine_similarity(feat_s, feat_t, dim=-1)).mean()
                    loss = loss_ce + 1.0 * loss_feat
                else:
                    outputs_s = model(images)
                    if loss_type == "ce":
                        loss = ce_loss_fn(outputs_s, targets)
                    elif loss_type == "cb_ce":
                        loss = cb_ce_loss_fn(outputs_s, targets)
                    elif loss_type == "kd" and teacher_model is not None:
                        with torch.no_grad():
                            outputs_t = teacher_model(images)
                        loss_ce = ce_loss_fn(outputs_s, targets)
                        loss_kd = F.kl_div(
                            F.log_softmax(outputs_s / 3.0, dim=1),
                            F.softmax(outputs_t / 3.0, dim=1),
                            reduction="batchmean",
                        ) * (3.0 ** 2)
                        loss = 0.5 * loss_ce + 0.5 * loss_kd
                    elif loss_type == "dkd" and teacher_model is not None:
                        with torch.no_grad():
                            outputs_t = teacher_model(images)
                        loss_ce = ce_loss_fn(outputs_s, targets)
                        loss_dkd = compute_dkd_loss(outputs_s, outputs_t, targets, alpha=1.0, beta=2.0, temperature=3.0)
                        loss = loss_ce + loss_dkd
                    elif loss_type == "cmpr" and teacher_model is not None:
                        with torch.no_grad():
                            outputs_t = teacher_model(images)
                        loss_ce = ce_loss_fn(outputs_s, targets)
                        loss_cmpr = compute_cmpr_loss(outputs_s, outputs_t, targets, gamma=0.0, teacher_gate=True)
                        loss = loss_ce + 0.5 * loss_cmpr
                    elif loss_type == "cb_ce_cmpr" and teacher_model is not None:
                        with torch.no_grad():
                            outputs_t = teacher_model(images)
                        loss_base = cb_ce_loss_fn(outputs_s, targets)
                        loss_cmpr = compute_cmpr_loss(outputs_s, outputs_t, targets, gamma=0.0, teacher_gate=True)
                        loss = loss_base + 0.5 * loss_cmpr
                    else:
                        loss = ce_loss_fn(outputs_s, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * targets.size(0)
            preds = torch.argmax(outputs_s, dim=1)
            correct += int((preds == targets).sum().item())
            total += targets.size(0)

            if step % 200 == 0 or step == num_batches:
                cur_loss = running_loss / total
                cur_acc = (correct / total) * 100.0
                elapsed = time.time() - epoch_start_t
                speed = total / elapsed if elapsed > 0 else 0.0
                print(
                    f"  [{run_name}] Epoch {epoch:02d}/{epochs:02d} | Step {step:04d}/{num_batches:04d} "
                    f"| Loss: {cur_loss:.4f} | Acc: {cur_acc:.2f}% | Speed: {speed:.1f} samp/s",
                    flush=True,
                )

        collator.shutdown()
        scheduler.step()

        epoch_train_loss = running_loss / total if total > 0 else 0.0
        epoch_train_acc = (correct / total) * 100.0 if total > 0 else 0.0
        epoch_train_time = round(time.time() - epoch_start_t, 1)

        # Validation evaluation
        print(f"  --> Evaluating Epoch {epoch:02d} on {len(val_samples):,} validation images...", flush=True)
        val_res = evaluate_split(model, val_samples, val_dir, dev, batch_size=batch_size)
        print(
            f"  === Epoch {epoch:02d} Results: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.2f}% "
            f"| Val Loss={val_res['loss']}, Val Acc={val_res['top1_accuracy']}%, Val Macro-F1={val_res['macro_f1']}%",
            flush=True,
        )

        record = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 2),
            "val_loss": val_res["loss"],
            "val_acc": val_res["top1_accuracy"],
            "val_macro_f1": val_res["macro_f1"],
            "train_time_s": epoch_train_time,
        }
        history.append(record)

        # Save latest checkpoint
        torch.save({
            "run_name": run_name,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "history": history,
        }, final_ckpt_path)

        # Checkpoint selection strictly based on validation Macro-F1
        if val_res["macro_f1"] > best_val_f1:
            best_val_f1 = val_res["macro_f1"]
            best_epoch = epoch
            print(f"  *** New Best Validation Macro-F1: {best_val_f1:.2f}% (Epoch {epoch})! Saving Best Checkpoint ***", flush=True)
            torch.save({
                "arm_id": arm_id,
                "run_name": run_name,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_f1,
                "best_val_acc": val_res["top1_accuracy"],
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "history": history,
            }, best_ckpt_path)

    total_train_time = round(time.time() - train_start_t, 1)
    with open(log_save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Final Evaluation on strictly held-out TEST set using best checkpoint
    print(f"\n  >>> Evaluating Selected Best Checkpoint (Epoch {best_epoch}) on Official Test Set (19,902 images)...", flush=True)
    best_ckpt = torch.load(best_ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_res = evaluate_split(model, test_samples, test_dir, dev, batch_size=batch_size)

    # Save confusion matrix
    cm_path = os.path.join(CM_DIR, f"{run_name}_confusion_matrix.json")
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_name": run_name,
            "classes": BDD_CLASSES,
            "confusion_matrix": test_res["confusion_matrix"],
            "per_class": test_res["per_class"],
            "top1_accuracy": test_res["top1_accuracy"],
            "macro_f1": test_res["macro_f1"],
            "weighted_f1": test_res["weighted_f1"],
            "test_predictions": test_res["predictions"],
            "test_targets": test_res["targets"],
            "test_logits": test_res["logits"],
        }, f, indent=2)

    print(
        f"  [TEST COMPLETE] {run_name}: Top-1={test_res['top1_accuracy']}%, "
        f"Macro-F1={test_res['macro_f1']}%, Weighted-F1={test_res['weighted_f1']}%",
        flush=True,
    )
    for pc in test_res["per_class"]:
        print(f"    Class {pc['class_id']} ({pc['class_name']:12s}): Recall={pc['recall']:5.2f}%, F1={pc['f1']:5.2f}%", flush=True)

    summary_result = {
        "arm_id": arm_id,
        "run_name": run_name,
        "seed": seed,
        "arch_config": arch_config,
        "loss_type": loss_type,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_top1_accuracy": test_res["top1_accuracy"],
        "test_macro_f1": test_res["macro_f1"],
        "test_weighted_f1": test_res["weighted_f1"],
        "test_per_class": test_res["per_class"],
        "training_time_s": total_train_time,
        "best_checkpoint_path": best_ckpt_path,
        "best_checkpoint_hash": compute_sha256(best_ckpt_path),
        "test_predictions": test_res["predictions"],
        "test_targets": test_res["targets"],
        "test_logits": test_res["logits"],
    }
    return summary_result


def main():
    parser = argparse.ArgumentParser(description="Run controlled matched-training experiments on BDD100K.")
    parser.add_argument("--arm", type=str, default="all", choices=["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "all"],
                        help="Specific arm to run, or 'all'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (default: cuda).")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test with subset of data.")
    args = parser.parse_args()

    print("================================================================================")
    print(" BDD100K CONTROLLED MATCHED-TRAINING BENCHMARK SUITE")
    print(f" Device: {args.device} | PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" Epochs: {args.epochs} | Batch Size: {args.batch_size} | Seed: {args.seed}")
    print("================================================================================")

    # Arm definitions
    ARMS = {
        "M0": {"arch": "deit_small_baseline", "loss": "ce", "needs_teacher": False},
        "M1": {"arch": "evit_small_k05_fuse", "loss": "ce", "needs_teacher": False},
        "M2": {"arch": "evit_small_k05_fuse", "loss": "cb_ce", "needs_teacher": False},
        "M3": {"arch": "evit_small_k05_fuse", "loss": "kd", "needs_teacher": True},
        "M4": {"arch": "evit_small_k05_fuse", "loss": "dkd", "needs_teacher": True},
        "M5": {"arch": "evit_small_k05_fuse", "loss": "feature", "needs_teacher": True},
        "M6": {"arch": "evit_small_k05_fuse", "loss": "cmpr", "needs_teacher": True},
        "M7": {"arch": "evit_small_k05_fuse", "loss": "cb_ce_cmpr", "needs_teacher": True},
    }

    # Step 1: Ensure Teacher (M0) is trained and available
    teacher_model = None
    teacher_hash = None
    m0_best_ckpt = os.path.join(CHECKPOINTS_DIR, f"M0_seed{args.seed}_best.pth")

    if os.path.isfile(m0_best_ckpt):
        print(f"[*] Found existing trained teacher checkpoint: {m0_best_ckpt}", flush=True)
        teacher_hash = compute_sha256(m0_best_ckpt)
        teacher_model = build_model("deit_small_baseline", num_classes=NUM_BDD_CLASSES, pretrained=False)
        ckpt = torch.load(m0_best_ckpt, map_location=args.device, weights_only=False)
        teacher_model.load_state_dict(ckpt["model_state_dict"])
        teacher_model.to(args.device)
        teacher_model.eval()
    elif args.arm in ["M3", "M4", "M5", "M6", "all"]:
        print("[*] Teacher M0 checkpoint not found. Training M0 first to establish matched teacher...", flush=True)
        m0_res = train_and_evaluate_arm(
            arm_id="M0",
            arch_config="deit_small_baseline",
            loss_type="ce",
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            smoke_test=args.smoke_test,
        )
        teacher_hash = m0_res["best_checkpoint_hash"]
        teacher_model = build_model("deit_small_baseline", num_classes=NUM_BDD_CLASSES, pretrained=False)
        ckpt = torch.load(m0_res["best_checkpoint_path"], map_location=args.device, weights_only=False)
        teacher_model.load_state_dict(ckpt["model_state_dict"])
        teacher_model.to(args.device)
        teacher_model.eval()

    arms_to_run = list(ARMS.keys()) if args.arm == "all" else [args.arm]
    if args.arm == "all" and os.path.isfile(m0_best_ckpt):
        # M0 already handled
        arms_to_run = [a for a in arms_to_run if a != "M0"]

    results = {}
    for arm in arms_to_run:
        arm_info = ARMS[arm]
        t_model = teacher_model if arm_info["needs_teacher"] else None
        t_hash = teacher_hash if arm_info["needs_teacher"] else None
        res = train_and_evaluate_arm(
            arm_id=arm,
            arch_config=arm_info["arch"],
            loss_type=arm_info["loss"],
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            teacher_model=t_model,
            teacher_ckpt_hash=t_hash,
            device=args.device,
            smoke_test=args.smoke_test,
        )
        results[arm] = res

    print("\n================================================================================")
    print(" ALL REQUESTED EXPERIMENT ARMS COMPLETED SUCCESSFULLY!")
    print("================================================================================")


if __name__ == "__main__":
    main()
