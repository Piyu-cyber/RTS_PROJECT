"""Cross-Keep-Rate Generalization Experiment Engine for EViT (k09, k07, k05).

Executes controlled 3-epoch matched training across:
- K09-CE, K09-CB, K09-CMPR, K09-CB-CMPR
- K07-CE, K07-CB, K07-CMPR, K07-CB-CMPR
- K05-CE, K05-CB, K05-CMPR, K05-CB-CMPR

Strictly enforces:
- Exact verified class-balanced weights: [0.3393, 0.4268, 0.5167, 1.1118, 2.1507, 1.4547]
- 3 full epochs (3,267 steps) per model
- Checkpoint selection via validation Macro-F1
- Isolated outputs under outputs/cross_keep_rate_study/
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
from src.phase23b.models import MODEL_CONFIGS, build_model, load_official_checkpoint

STUDY_DIR = os.path.join(PROJECT_ROOT, "outputs", "cross_keep_rate_study")
CHECKPOINTS_DIR = os.path.join(STUDY_DIR, "checkpoints")
CONFIGS_DIR = os.path.join(STUDY_DIR, "training_configurations")
LOGS_DIR = os.path.join(STUDY_DIR, "training_logs")
CM_DIR = os.path.join(STUDY_DIR, "confusion_matrices")
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "bdd100k", "annotations")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "bdd100k", "images")
TEACHER_DIR = os.path.join(PROJECT_ROOT, "outputs", "cmpr_matched_training", "checkpoints")

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


def get_verified_class_balanced_weights(device: torch.device) -> torch.Tensor:
    """Exact verified normalized class-balanced weights from Phase 0 audit."""
    # Counts: City(43581), Highway(17414), Residential(8105), Parking(378), Gas(27), Tunnel(129)
    # raw = (N / (C * n_c))^0.25, normalized by mean(raw) = 2.1174
    weights = np.array([0.3393, 0.4268, 0.5167, 1.1118, 2.1507, 1.4547], dtype=np.float32)
    return torch.tensor(weights, dtype=torch.float32, device=device)


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


def train_and_evaluate_model(
    model_id: str,
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
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Trains a model configuration and evaluates on official test set."""
    dev = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_study_dir = output_dir or STUDY_DIR
    target_checkpoints_dir = os.path.join(target_study_dir, "checkpoints")
    target_configs_dir = os.path.join(target_study_dir, "training_configurations")
    target_logs_dir = os.path.join(target_study_dir, "training_logs")
    target_cm_dir = os.path.join(target_study_dir, "confusion_matrices")
    for d in [target_checkpoints_dir, target_configs_dir, target_logs_dir, target_cm_dir]:
        os.makedirs(d, exist_ok=True)

    run_name = f"{model_id}_seed{seed}"
    best_ckpt_path = os.path.join(target_checkpoints_dir, f"{run_name}_best.pth")
    final_ckpt_path = os.path.join(target_checkpoints_dir, f"{run_name}_final.pth")
    config_save_path = os.path.join(target_configs_dir, f"{run_name}_config.json")
    log_save_path = os.path.join(target_logs_dir, f"{run_name}_history.json")

    print(f"\n================================================================================", flush=True)
    print(f"  LAUNCHING CROSS-KEEP-RATE MODEL: {model_id} | Run: {run_name}", flush=True)
    print(f"  Architecture: {arch_config} | Loss: {loss_type} | Seed: {seed} | Epochs: {epochs}", flush=True)
    print(f"================================================================================", flush=True)

    run_config = {
        "model_id": model_id,
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
        "verified_class_weights": [0.3393, 0.4268, 0.5167, 1.1118, 2.1507, 1.4547],
        "teacher_checkpoint_hash": teacher_ckpt_hash,
    }
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    train_samples = load_bdd_split_samples(os.path.join(ANNOTATIONS_DIR, "train_full.json"))
    val_samples = load_bdd_split_samples(os.path.join(ANNOTATIONS_DIR, "val_full.json"))
    test_samples = load_bdd_split_samples(os.path.join(ANNOTATIONS_DIR, "test_full.json"))

    train_dir = os.path.join(IMAGES_DIR, "100k", "train")
    val_dir = os.path.join(IMAGES_DIR, "100k", "val")
    test_dir = os.path.join(IMAGES_DIR, "100k", "test")

    model = build_model(arch_config, num_classes=NUM_BDD_CLASSES, pretrained=False)
    load_official_checkpoint(model, arch_config, device=device)
    model.to(dev)

    head_params = list(model.head.parameters())
    head_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr_head, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    ce_loss_fn = nn.CrossEntropyLoss()
    cb_weights = get_verified_class_balanced_weights(dev) if "cb" in loss_type else None
    cb_ce_loss_fn = nn.CrossEntropyLoss(weight=cb_weights) if "cb" in loss_type else None

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
                outputs_s = model(images)
                if loss_type == "ce":
                    loss = ce_loss_fn(outputs_s, targets)
                elif loss_type == "cb_ce":
                    loss = cb_ce_loss_fn(outputs_s, targets)
                elif loss_type == "cmpr" and teacher_model is not None:
                    with torch.no_grad():
                        outputs_t = teacher_model(images)
                    loss_base = ce_loss_fn(outputs_s, targets)
                    loss_cmpr = compute_cmpr_loss(outputs_s, outputs_t, targets, gamma=0.0, teacher_gate=True)
                    loss = loss_base + 0.5 * loss_cmpr
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

        torch.save({
            "run_name": run_name,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "history": history,
        }, final_ckpt_path)

        if val_res["macro_f1"] > best_val_f1:
            best_val_f1 = val_res["macro_f1"]
            best_epoch = epoch
            print(f"  *** New Best Validation Macro-F1: {best_val_f1:.2f}% (Epoch {epoch})! Saving Best Checkpoint ***", flush=True)
            torch.save({
                "model_id": model_id,
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

    # Evaluate Best Checkpoint on Test Set
    print(f"\n  >>> Evaluating Selected Best Checkpoint (Epoch {best_epoch}) on Official Test Set (19,902 images)...", flush=True)
    best_ckpt = torch.load(best_ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_res = evaluate_split(model, test_samples, test_dir, dev, batch_size=batch_size)

    cm_path = os.path.join(target_cm_dir, f"{run_name}_confusion_matrix.json")
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": model_id,
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

    return {
        "model_id": model_id,
        "run_name": run_name,
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
    }


def main():
    parser = argparse.ArgumentParser(description="Run cross-keep-rate experiments.")
    parser.add_argument("--model-id", type=str, required=True,
                        choices=[
                            "K09-CE", "K09-CB", "K09-CMPR", "K09-CB-CMPR",
                            "K07-CE", "K07-CB", "K07-CMPR", "K07-CB-CMPR",
                            "K05-CE", "K05-CB", "K05-CMPR", "K05-CB-CMPR",
                            "DeiT",
                        ],
                        help="Specific model to run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (default: cuda).")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save outputs (checkpoints, logs, configs, cm).")
    args = parser.parse_args()

    MODEL_REGISTRY = {
        "DeiT": ("deit_small_baseline", "ce", False),

        "K09-CE": ("evit_small_k09_fuse", "ce", False),
        "K09-CB": ("evit_small_k09_fuse", "cb_ce", False),
        "K09-CMPR": ("evit_small_k09_fuse", "cmpr", True),
        "K09-CB-CMPR": ("evit_small_k09_fuse", "cb_ce_cmpr", True),

        "K07-CE": ("evit_small_k07_fuse", "ce", False),
        "K07-CB": ("evit_small_k07_fuse", "cb_ce", False),
        "K07-CMPR": ("evit_small_k07_fuse", "cmpr", True),
        "K07-CB-CMPR": ("evit_small_k07_fuse", "cb_ce_cmpr", True),

        "K05-CE": ("evit_small_k05_fuse", "ce", False),
        "K05-CB": ("evit_small_k05_fuse", "cb_ce", False),
        "K05-CMPR": ("evit_small_k05_fuse", "cmpr", True),
        "K05-CB-CMPR": ("evit_small_k05_fuse", "cb_ce_cmpr", True),
    }

    arch_config, loss_type, needs_teacher = MODEL_REGISTRY[args.model_id]

    teacher_model = None
    teacher_hash = None
    if needs_teacher:
        cand_paths = [
            os.path.join(CHECKPOINTS_DIR, f"DeiT_seed{args.seed}_best.pth"),
            os.path.join(TEACHER_DIR, f"M0_seed{args.seed}_best.pth"),
            os.path.join(TEACHER_DIR, "M0_seed42_best.pth"),
        ]
        teacher_path = next((p for p in cand_paths if os.path.isfile(p)), None)
        if teacher_path is None:
            raise FileNotFoundError(f"No suitable teacher checkpoint found for seed {args.seed} in {cand_paths}")
        print(f"  [TEACHER RESOLVED] Using teacher checkpoint: {teacher_path}", flush=True)
        teacher_hash = compute_sha256(teacher_path)
        teacher_model = build_model("deit_small_baseline", num_classes=NUM_BDD_CLASSES, pretrained=False)
        ckpt = torch.load(teacher_path, map_location=args.device, weights_only=False)
        teacher_model.load_state_dict(ckpt["model_state_dict"])
        teacher_model.to(args.device)
        teacher_model.eval()

    res = train_and_evaluate_model(
        model_id=args.model_id,
        arch_config=arch_config,
        loss_type=loss_type,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        teacher_model=teacher_model,
        teacher_ckpt_hash=teacher_hash,
        device=args.device,
        output_dir=args.output_dir,
    )
    print("\n[+] Cross-keep-rate run finished successfully:", res["model_id"])


if __name__ == "__main__":
    main()
