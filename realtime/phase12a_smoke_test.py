"""Stage 4 Pre-training Smoke Test for Phase 12A."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["XFORMERS_DISABLED"] = "true"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "reference"))

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.oxford_pet_loader import IGNORE_INDEX, load_dataset_from_manifest
from baseline.train_segmentation import build_backbone, strict_load_backbone, SegmentationHead, features


def run_smoke_test() -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device required for smoke test")

    manifest_path = PROJECT_ROOT / "outputs" / "phase12a_full_pet" / "split_manifest.json"
    train_ds, val_ds, manifest = load_dataset_from_manifest(manifest_path, PROJECT_ROOT, image_size=224)

    batch_size = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    images, targets = next(iter(train_loader))

    # 1. Shapes check
    img_shape = list(images.shape)
    target_shape = list(targets.shape)
    assert img_shape == [batch_size, 3, 224, 224], f"Unexpected image shape: {img_shape}"
    assert target_shape == [batch_size, 224, 224], f"Unexpected target shape: {target_shape}"

    # 2. Target values check
    unique_targets = torch.unique(targets).tolist()
    assert set(unique_targets).issubset({0, 1, IGNORE_INDEX}), f"Unexpected target values: {unique_targets}"

    # 3. Model setup
    backbone = build_backbone()
    strict_load_backbone(backbone)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    head = SegmentationHead().to(device).train()

    # 4. Forward pass
    images = images.to(device)
    targets = targets.to(device)

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        feats = features(backbone, images)
    logits = head(feats, targets.shape[-2:])

    # 5. Loss computation
    loss = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)
    assert torch.isfinite(loss).item(), "Loss is not finite!"

    # 6. Backward pass and optimizer step
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad()
    loss.backward()

    # 7. Check gradients
    finite_grads = True
    grad_norms = []
    for p in head.parameters():
        if p.grad is not None:
            if not torch.all(torch.isfinite(p.grad)).item():
                finite_grads = False
            grad_norms.append(p.grad.norm().item())

    assert finite_grads, "Non-finite gradients detected!"
    optimizer.step()
    t1 = time.perf_counter()

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    results = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "batch_size": batch_size,
        "image_shape": img_shape,
        "target_shape": target_shape,
        "unique_target_values": unique_targets,
        "loss_value": float(loss.item()),
        "gradients_finite": finite_grads,
        "mean_grad_norm": float(sum(grad_norms) / len(grad_norms)) if grad_norms else 0.0,
        "peak_gpu_memory_mb": round(peak_mem_mb, 2),
        "iteration_time_seconds": round(t1 - t0, 4),
    }

    out_file = PROJECT_ROOT / "outputs" / "phase12a_full_pet" / "smoke_test.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Smoke test completed successfully: {results}")
    return results


if __name__ == "__main__":
    run_smoke_test()
