"""Benchmark full-attention and DWConv-patched segmentation inference modes."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

os.environ["XFORMERS_DISABLED"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "reference"))

import numpy as np
import torch

from baseline.train_segmentation import (
    DEPTH,
    EMBED_DIM,
    PATCH_SIZE,
    RunConfig,
    SegmentationHead,
    build_backbone,
    build_datasets,
    evaluate,
    features,
    loader,
    read_checkpoint,
    require,
    set_seed,
    strict_load_backbone,
)
from lib.config import Config
from lib.convatt.convatt_dw import CnvAttnDW_blockwise
from lib.patching.patch import patch_dinov2_convatt


MODES = {
    "full_attention": [],
    "moderate_dwconv": [2, 5, 8, 11],
    "aggressive_dwconv": [1, 2, 3, 4, 5, 6, 7, 8],
}


def make_patch_config(blocks: list[int], image_size: int) -> Config:
    return Config(
        float32_matmul_precision="highest", fp16=False, img_size=image_size,
        backbone={
            "backbone": "dinov2", "backbone_pretrained_path": None, "output_layers": [DEPTH - 1],
            "patch_size": PATCH_SIZE, "embed_dim": EMBED_DIM, "depth": DEPTH, "num_heads": 6,
            "mlp_ratio": 4.0, "drop_path_rate": 0.0, "include_cls_token": True,
            "pretrained_img_size": 518, "learnable_pos_embed": False, "num_register_tokens": 0,
        },
        convatt={
            "conv_style": "depthwise", "conv_granularity": "blockwise", "conv_selection": "manual",
            "head_ensembling": False, "blocks": blocks, "weight_decay": False, "layer_decay": False,
            "conv_init": True,
        },
    )


def capture_transfer_sources(backbone: torch.nn.Module, blocks: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    captured: dict[int, dict[str, torch.Tensor]] = {}
    for index in blocks:
        state = backbone.blocks[index].attn.state_dict()
        _, _, value_weight = torch.chunk(state["qkv.weight"], 3)
        _, _, value_bias = torch.chunk(state["qkv.bias"], 3)
        captured[index] = {
            "v_proj.weight": value_weight.detach().clone(), "v_proj.bias": value_bias.detach().clone(),
            "proj.weight": state["proj.weight"].detach().clone(), "proj.bias": state["proj.bias"].detach().clone(),
        }
    return captured


def verify_patch(backbone: torch.nn.Module, blocks: list[int], expected: dict[int, dict[str, torch.Tensor]], device: torch.device) -> dict[str, Any]:
    replacement_devices: dict[str, list[str]] = {}
    for index, block in enumerate(backbone.blocks):
        if index in blocks:
            require(isinstance(block.attn, CnvAttnDW_blockwise), f"Block {index} is not CnvAttnDW_blockwise")
            actual = {"v_proj.weight": block.attn.v_proj.weight, "v_proj.bias": block.attn.v_proj.bias,
                "proj.weight": block.attn.proj.weight, "proj.bias": block.attn.proj.bias}
            for name, source in expected[index].items():
                require(torch.equal(actual[name], source.to(actual[name].device)), f"Weight transfer failed for block {index}: {name}")
            replacement_devices[str(index)] = sorted({str(tensor.device) for _, tensor in list(block.attn.named_parameters()) + list(block.attn.named_buffers())})
        else:
            require(type(block.attn).__name__ == "MemEffAttention", f"Unexpected attention class in untouched block {index}")
    off_device = [name for name, tensor in list(backbone.named_parameters()) + list(backbone.named_buffers()) if tensor.device != device]
    require(not off_device, f"Model tensors not on {device}: {off_device[:10]}")
    return {"pretrained_weight_transfer": "v_proj and proj exact-copy PASS", "new_parameters": ["qk_conv", "cls_all", "all_cls"], "replacement_tensor_devices": replacement_devices}


def load_finetuned_checkpoint(path: Path, mode: str, blocks: list[int], expected_split: dict[str, Any]) -> dict[str, Any]:
    require(path.is_file(), f"Fine-tuned checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"backbone_state_dict", "head_state_dict", "mode", "blocks", "split_manifest", "metrics"}
    require(isinstance(payload, dict) and required.issubset(payload), f"Not a Phase 6 fine-tuning checkpoint: {path}")
    require(payload["mode"] == mode, f"Checkpoint mode mismatch for {path}: expected {mode}, got {payload['mode']}")
    require(payload["blocks"] == blocks, f"Checkpoint block policy mismatch for {path}: expected {blocks}, got {payload['blocks']}")
    require(payload["split_manifest"] == expected_split, f"Checkpoint split manifest differs from the fixed Phase 4 split: {path}")
    return payload


def prepare_mode(
    mode: str,
    blocks: list[int],
    cfg: RunConfig,
    head_state: dict[str, torch.Tensor],
    device: torch.device,
    finetuned_checkpoint: dict[str, Any] | None = None,
) -> tuple[torch.nn.Module, SegmentationHead, dict[str, Any]]:
    set_seed(cfg.seed)
    backbone = build_backbone()
    strict_load_backbone(backbone)
    details: dict[str, Any] = {"name": mode, "block_indices": blocks, "patched_block_count": len(blocks)}
    if blocks:
        expected = capture_transfer_sources(backbone, blocks)
        patch_dinov2_convatt(backbone, make_patch_config(blocks, cfg.image_size))
    backbone = backbone.to(device).eval()
    head = SegmentationHead().to(device).eval()
    head.load_state_dict(head_state, strict=True)
    if blocks:
        details["patch_validation"] = verify_patch(backbone, blocks, expected, device)
    else:
        off_device = [name for name, tensor in list(backbone.named_parameters()) + list(backbone.named_buffers()) if tensor.device != device]
        require(not off_device, f"Full-attention backbone tensors not on {device}: {off_device[:10]}")
        details["patch_validation"] = {"replacement_tensor_devices": {}, "pretrained_weight_transfer": "not applicable"}
    if finetuned_checkpoint is not None:
        backbone.load_state_dict(finetuned_checkpoint["backbone_state_dict"], strict=True)
        head.load_state_dict(finetuned_checkpoint["head_state_dict"], strict=True)
        off_device = [name for name, tensor in list(backbone.named_parameters()) + list(backbone.named_buffers()) if tensor.device != device]
        require(not off_device, f"Fine-tuned {mode} tensors not on {device}: {off_device[:10]}")
        details["fine_tuned_checkpoint"] = str(Path(finetuned_checkpoint["_path"]).resolve())
        details["fine_tuned_epoch"] = finetuned_checkpoint["epoch"]
        details["fine_tuned_state_load"] = "backbone and segmentation head strict-load PASS"
    return backbone, head, details


@torch.inference_mode()
def validate_logits(backbone: torch.nn.Module, head: SegmentationHead, image: torch.Tensor) -> dict[str, Any]:
    logits = head(features(backbone, image), (224, 224))
    prediction = logits.argmax(dim=1)
    require(logits.shape == (1, 2, 224, 224), f"Unexpected logits shape: {tuple(logits.shape)}")
    require(torch.isfinite(logits).all().item(), "Logits contain NaN or infinity")
    require(torch.all((prediction >= 0) & (prediction < 2)).item(), "Predictions contain invalid class labels")
    return {"logits_shape": list(logits.shape), "finite_logits": True, "prediction_labels": sorted(prediction.unique().cpu().tolist())}


@torch.inference_mode()
def measure_latency(backbone: torch.nn.Module, head: SegmentationHead, image: torch.Tensor, warmups: int, repeats: int) -> dict[str, float | int]:
    for _ in range(warmups):
        _ = head(features(backbone, image), (224, 224))
    torch.cuda.synchronize(image.device)
    values: list[float] = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        _ = head(features(backbone, image), (224, 224))
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return {"warmups": warmups, "repetitions": repeats, "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(statistics.mean(values)), "std_ms": float(statistics.pstdev(values)), "max_observed_ms": float(max(values))}


def main(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "CUDA GPU is required for this benchmark")
    torch.set_float32_matmul_precision("high")
    payload = read_checkpoint(Path(args.checkpoint))
    cfg = RunConfig(**payload["run_config"])
    require(cfg.image_size == 224, "This benchmark is defined for the validated 224x224 baseline")
    _, validation_data, split = build_datasets(Path(args.dataset_root), cfg, False, payload.get("split_manifest"))
    moderate_checkpoint = load_finetuned_checkpoint(Path(args.moderate_checkpoint), "moderate", MODES["moderate_dwconv"], split)
    moderate_checkpoint["_path"] = args.moderate_checkpoint
    aggressive_checkpoint = load_finetuned_checkpoint(Path(args.aggressive_checkpoint), "aggressive", MODES["aggressive_dwconv"], split)
    aggressive_checkpoint["_path"] = args.aggressive_checkpoint
    validation_loader = loader(validation_data, cfg.batch_size, False)
    image, _ = validation_data[0]
    image = image.unsqueeze(0).to(device)
    results: dict[str, Any] = {
        "task": "Oxford-IIIT Pet binary foreground/background segmentation inference benchmark",
        "checkpoint": {
            "full_attention_segmentation_head": str(Path(args.checkpoint).resolve()),
            "moderate_finetuned": str(Path(args.moderate_checkpoint).resolve()),
            "aggressive_finetuned": str(Path(args.aggressive_checkpoint).resolve()),
            "backbone_initialization": "Meta official dinov2_vits14_pretrain.pth",
        },
        "device": {"torch_device": str(device), "name": torch.cuda.get_device_name(device)},
        "input_shape": [1, 3, 224, 224], "validation_images": len(validation_data), "split_manifest": split,
        "latency_methodology": "CUDA events around device-resident backbone feature extraction plus segmentation head; excludes image preprocessing and host-to-device transfer. Maximum observed latency is not WCET.",
        "modes": {},
    }
    for mode, blocks in MODES.items():
        print(f"Running mode: {mode}; blocks={blocks}")
        finetuned = {
            "moderate_dwconv": moderate_checkpoint,
            "aggressive_dwconv": aggressive_checkpoint,
        }.get(mode)
        backbone, head, details = prepare_mode(mode, blocks, cfg, payload["head_state_dict"], device, finetuned)
        details["logit_validation"] = validate_logits(backbone, head, image)
        details["latency_ms"] = measure_latency(backbone, head, image, args.warmups, args.repetitions)
        details["accuracy"] = evaluate(backbone, head, validation_loader, device)
        results["modes"][mode] = details
        print(f"{mode}: mIoU={details['accuracy']['mean_iou']}, median_ms={details['latency_ms']['median_ms']:.3f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"PASS: benchmark completed. Results: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "outputs" / "pet_smoke" / "smoke_checkpoint.pt"))
    parser.add_argument("--moderate-checkpoint", default=str(PROJECT_ROOT / "outputs" / "finetune_moderate" / "multiepoch_10" / "best_checkpoint.pt"))
    parser.add_argument("--aggressive-checkpoint", default=str(PROJECT_ROOT / "outputs" / "finetune_aggressive" / "multiepoch_10" / "best_checkpoint.pt"))
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "inference_benchmark" / "results.json"))
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    main(parser.parse_args())
