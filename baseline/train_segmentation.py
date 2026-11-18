"""Reproducible Oxford-IIIT Pet binary foreground/background segmentation baseline."""
from __future__ import annotations

import argparse
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
REFERENCE_ROOT = PROJECT_ROOT / "reference"
sys.path.insert(0, str(REFERENCE_ROOT))

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF

from lib.models.dino_vision_transformer import DinoVisionTransformer

WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
WEIGHTS_PATH = REFERENCE_ROOT / "weights" / "dinov2_vits14_pretrain.pth"
IGNORE_INDEX, NUM_CLASSES, PATCH_SIZE, EMBED_DIM, DEPTH, NUM_HEADS = 255, 2, 14, 384, 12, 6
CLASS_NAMES = ["background", "pet"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class RunConfig:
    image_size: int; batch_size: int; epochs: int; learning_rate: float; weight_decay: float
    train_samples: int; val_samples: int; seed: int; freeze_backbone: bool; amp: bool


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class PetDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """The same resize applies to each image and mask, with nearest mask resize."""
    def __init__(self, source: OxfordIIITPet, image_size: int) -> None:
        self.source, self.image_size = source, image_size
    def __len__(self) -> int:
        return len(self.source)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.source[index]
        image = TF.resize(image, [self.image_size] * 2, interpolation=Image.Resampling.BILINEAR)
        mask = TF.resize(mask, [self.image_size] * 2, interpolation=Image.Resampling.NEAREST)
        image = TF.normalize(TF.to_tensor(image), MEAN, STD)
        raw = np.asarray(mask, dtype=np.uint8)
        target = np.full(raw.shape, IGNORE_INDEX, dtype=np.int64)
        target[raw == 3], target[raw == 1] = 0, 1  # border value 2 remains ignored
        return image, torch.from_numpy(target)


class SegmentationHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(EMBED_DIM, 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, NUM_CLASSES, 1))
    def forward(self, features: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(self.layers(features), size=size, mode="bilinear", align_corners=False)


def build_backbone() -> DinoVisionTransformer:
    return DinoVisionTransformer(img_size=518, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, depth=DEPTH,
        num_heads=NUM_HEADS, mlp_ratio=4.0, drop_path_rate=0.0, num_register_tokens=0,
        interpolate_antialias=True, interpolate_offset=0, block_chunks=0, init_values=1e-5, use_cls=True)


def strict_load_backbone(backbone: DinoVisionTransformer) -> None:
    require(WEIGHTS_PATH.is_file(), f"Missing official DINOv2-Small checkpoint: {WEIGHTS_PATH}")
    state: Any = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    for key in ("model", "state_dict"):
        if isinstance(state, dict) and isinstance(state.get(key), dict): state = state[key]
    require(isinstance(state, dict), "DINOv2 checkpoint is not a state dictionary")
    expected = backbone.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    mismatched = sorted(k for k in set(expected) & set(state) if expected[k].shape != state[k].shape)
    print(f"Checkpoint provenance: Meta official DINOv2 ViT-S/14 at {WEIGHTS_URL}")
    print(f"Strict load compatibility: missing={len(missing)}, unexpected={len(unexpected)}, shape_mismatched={len(mismatched)}")
    require(not missing and not unexpected and not mismatched, "Checkpoint is incompatible with this DINOv2-Small backbone")
    backbone.load_state_dict(state, strict=True)


def build_datasets(root: Path, cfg: RunConfig, download: bool, saved: dict[str, Any] | None = None) -> tuple[Dataset, Dataset, dict[str, Any]]:
    train_source = OxfordIIITPet(root, split="trainval", target_types="segmentation", download=download)
    val_source = OxfordIIITPet(root, split="test", target_types="segmentation", download=download)
    train, val = PetDataset(train_source, cfg.image_size), PetDataset(val_source, cfg.image_size)
    if saved is None:
        require(cfg.train_samples <= len(train) and cfg.val_samples <= len(val), "Requested subset exceeds dataset size")
        generator = torch.Generator().manual_seed(cfg.seed)
        train_indices = torch.randperm(len(train), generator=generator)[:cfg.train_samples].tolist()
        val_indices = torch.randperm(len(val), generator=generator)[:cfg.val_samples].tolist()
    else:
        train_indices, val_indices = list(saved["train_indices"]), list(saved["val_indices"])
    manifest = {"train_source": "OxfordIIITPet/trainval", "validation_source": "OxfordIIITPet/test",
        "train_indices": train_indices, "val_indices": val_indices,
        "train_ids": [train_source._images[i].stem for i in train_indices],
        "val_ids": [val_source._images[i].stem for i in val_indices]}
    return Subset(train, train_indices), Subset(val, val_indices), manifest


def features(backbone: DinoVisionTransformer, images: torch.Tensor) -> torch.Tensor:
    return backbone.get_intermediate_layers(images, [DEPTH - 1], reshape=True, return_class_token=False)[0]


def loader(data: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


@torch.inference_mode()
def evaluate(backbone: DinoVisionTransformer, head: SegmentationHead, batches: DataLoader, device: torch.device) -> dict[str, Any]:
    backbone.eval(); head.eval(); confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64); images_seen = 0
    for images, targets in batches:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        prediction = head(features(backbone, images), targets.shape[-2:]).argmax(1)
        require(torch.all((prediction >= 0) & (prediction < NUM_CLASSES)).item(), "Prediction has an invalid class label")
        valid = targets != IGNORE_INDEX
        encoded = targets[valid] * NUM_CLASSES + prediction[valid]
        confusion += torch.bincount(encoded.cpu(), minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
        images_seen += images.shape[0]
    per_class: dict[str, float | None] = {}; available: list[float] = []
    for i, name in enumerate(CLASS_NAMES):
        tp = confusion[i, i].item(); union = tp + confusion[:, i].sum().item() - tp + confusion[i, :].sum().item() - tp
        score = None if union == 0 else tp / union
        per_class[name] = score
        if score is not None: available.append(score)
    return {"evaluated_images": images_seen, "mean_iou": None if not available else sum(available) / len(available),
        "per_class_iou": per_class, "confusion_matrix": confusion.tolist()}


def colorize(mask: np.ndarray) -> np.ndarray:
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    result[mask == 0], result[mask == 1], result[mask == IGNORE_INDEX] = (35, 75, 160), (230, 145, 30), (190, 190, 190)
    return result


def save_visualization(backbone: DinoVisionTransformer, head: SegmentationHead, data: Dataset, device: torch.device, output: Path) -> None:
    image, truth = data[0]
    with torch.inference_mode(): pred = head(features(backbone, image.unsqueeze(0).to(device)), truth.shape).argmax(1)[0].byte().cpu().numpy()
    require(set(np.unique(pred)).issubset({0, 1}), "Saved prediction has invalid labels")
    restored = image.numpy().transpose(1, 2, 0) * np.asarray(STD) + np.asarray(MEAN)
    view = np.concatenate([(np.clip(restored, 0, 1) * 255).astype(np.uint8), colorize(truth.numpy()), colorize(pred)], axis=1)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred, "L").save(output / "sample_prediction.png")
    Image.fromarray(view, "RGB").save(output / "sample_visualization.png")
    print(f"Prediction labels: {sorted(np.unique(pred).tolist())}; visualization: {output / 'sample_visualization.png'}")


def read_checkpoint(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict) and "head_state_dict" in payload and "run_config" in payload, "Checkpoint is not a baseline checkpoint")
    return payload


def save_checkpoint(path: Path, head: SegmentationHead, cfg: RunConfig, split: dict[str, Any], epoch: int, metrics: dict[str, Any]) -> None:
    torch.save({"head_state_dict": head.state_dict(), "run_config": asdict(cfg), "split_manifest": split, "epoch": epoch, "metrics": metrics, "class_names": CLASS_NAMES}, path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def prepare_model(cfg: RunConfig, device: torch.device) -> tuple[DinoVisionTransformer, SegmentationHead]:
    backbone = build_backbone(); strict_load_backbone(backbone); backbone = backbone.to(device).eval()
    if cfg.freeze_backbone:
        for parameter in backbone.parameters(): parameter.requires_grad = False
    return backbone, SegmentationHead().to(device)


def run_eval(args: argparse.Namespace, device: torch.device) -> None:
    payload = read_checkpoint(Path(args.checkpoint)); cfg = RunConfig(**payload["run_config"]); set_seed(cfg.seed)
    train, val, split = build_datasets(Path(args.dataset_root), cfg, False, payload.get("split_manifest"))
    backbone, head = prepare_model(cfg, device); head.load_state_dict(payload["head_state_dict"], strict=True)
    metrics = evaluate(backbone, head, loader(val, cfg.batch_size, False), device)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    metrics.update({"run_config": asdict(cfg), "split_manifest": split, "checkpoint": args.checkpoint, "device": str(device)})
    write_json(output / "evaluation_metrics.json", metrics); save_visualization(backbone, head, val, device, output)
    saved_path = Path(args.checkpoint).parent / "metrics.json"
    if saved_path.is_file():
        saved = json.loads(saved_path.read_text(encoding="utf-8")); baseline = saved.get("mean_iou", saved.get("best_validation", {}).get("mean_iou"))
        print(f"Saved mIoU: {baseline}; reevaluated mIoU: {metrics['mean_iou']}; difference: {None if baseline is None else metrics['mean_iou'] - baseline}")
    print(f"Validation mIoU: {metrics['mean_iou']}; per-class IoU: {metrics['per_class_iou']}")
    print("PASS: checkpoint-only validation completed.")


def run_train(args: argparse.Namespace, device: torch.device) -> None:
    cfg = RunConfig(args.image_size, args.batch_size, args.epochs, args.learning_rate, args.weight_decay, args.train_samples, args.val_samples, args.seed, not args.finetune_backbone, args.amp)
    require(cfg.image_size % PATCH_SIZE == 0, f"--image-size must be divisible by {PATCH_SIZE}"); set_seed(cfg.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    train, val, split = build_datasets(Path(args.dataset_root), cfg, args.download)
    write_json(output / "split_manifest.json", split)
    write_json(output / "experiment_config.json", {"task": "Oxford-IIIT Pet binary foreground/background segmentation", "run_config": asdict(cfg), "class_names": CLASS_NAMES, "ignore_index": IGNORE_INDEX, "device": str(device)})
    backbone, head = prepare_model(cfg, device)
    trainable = [p for p in list(backbone.parameters()) + list(head.parameters()) if p.requires_grad]
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Dataset: Oxford-IIIT Pet binary foreground/background, train={len(train)}, validation={len(val)}")
    print(f"Parameters: backbone={sum(p.numel() for p in backbone.parameters()):,}, head={sum(p.numel() for p in head.parameters()):,}, trainable={sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay); scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    train_batches, val_batches = loader(train, cfg.batch_size, True), loader(val, cfg.batch_size, False)
    torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter(); history: list[dict[str, Any]] = []; best = float("-inf"); best_path = output / "best_checkpoint.pt"
    for epoch in range(1, cfg.epochs + 1):
        head.train(); backbone.eval() if cfg.freeze_backbone else backbone.train(); loss_sum = 0.0
        for images, targets in train_batches:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                if cfg.freeze_backbone:
                    with torch.no_grad(): feat = features(backbone, images)
                else: feat = features(backbone, images)
                loss = F.cross_entropy(head(feat, targets.shape[-2:]), targets, ignore_index=IGNORE_INDEX)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); loss_sum += loss.item() * images.shape[0]
        validation = evaluate(backbone, head, val_batches, device); record = {"epoch": epoch, "train_loss": loss_sum / len(train), "validation": validation}; history.append(record)
        print(f"Epoch {epoch}/{cfg.epochs}: train_loss={record['train_loss']:.6f}, validation_mIoU={validation['mean_iou']}")
        if validation["mean_iou"] is not None and validation["mean_iou"] > best:
            best = validation["mean_iou"]; save_checkpoint(best_path, head, cfg, split, epoch, record); print(f"Best checkpoint updated: epoch={epoch}, mIoU={best}")
    elapsed = time.perf_counter() - started; last_path = output / "last_checkpoint.pt"; save_checkpoint(last_path, head, cfg, split, cfg.epochs, history[-1])
    best_payload = read_checkpoint(best_path); head.load_state_dict(best_payload["head_state_dict"], strict=True); final = evaluate(backbone, head, val_batches, device)
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    result = {"task": "Oxford-IIIT Pet binary foreground/background segmentation", "run_config": asdict(cfg), "split_manifest": split, "device": str(device), "history": history, "best_epoch": best_payload["epoch"], "best_validation": final, "runtime_seconds": elapsed, "peak_gpu_memory_mib": peak, "best_checkpoint": str(best_path), "last_checkpoint": str(last_path)}
    write_json(output / "metrics.json", result); save_visualization(backbone, head, val, device, output)
    print(f"Best validation mIoU: {final['mean_iou']}; per-class IoU: {final['per_class_iou']}")
    print(f"Runtime: {elapsed:.2f}s; peak GPU memory: {peak:.1f} MiB")
    print("PASS: baseline training completed. This small subset is not benchmark performance.")


def main(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"); require(device.type == "cuda", "CUDA GPU is required")
    torch.set_float32_matmul_precision("high")
    run_eval(args, device) if args.mode == "eval" else run_train(args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "datasets")); parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "pet_experiment"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "outputs" / "pet_smoke" / "smoke_checkpoint.pt")); parser.add_argument("--download", action="store_true")
    parser.add_argument("--image-size", type=int, default=224); parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=64); parser.add_argument("--val-samples", type=int, default=32); parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--seed", type=int, default=17); parser.add_argument("--finetune-backbone", action="store_true"); parser.add_argument("--amp", action="store_true")
    main(parser.parse_args())
