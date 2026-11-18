"""Phase 23B Model builder wrapping official EViT and DeiT implementations."""

import os
import sys
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn

# Ensure reference/evit is on sys.path
EVIT_REF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reference", "evit"))
if EVIT_REF_DIR not in sys.path:
    sys.path.insert(0, EVIT_REF_DIR)

from evit import deit_small_patch16_224, deit_small_patch16_shrink_base

WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reference", "evit_weights"))

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "deit_small_baseline": {
        "model_fn": "deit_small_patch16_224",
        "keep_rate": 1.0,
        "fuse_token": False,
        "drop_loc": (),
        "checkpoint_filename": "deit_small_patch16_224.pth",
        "published_macs": 4.6,
        "published_top1_in1k": 79.8,
        "published_top5_in1k": 95.0,
        "published_throughput": None,  # Not explicitly tabulated in same row in paper Table 1
        "description": "Standard pretrained DeiT-Small, full token sequence",
    },
    "evit_small_k09_fuse": {
        "model_fn": "deit_small_patch16_shrink_base",
        "keep_rate": 0.9,
        "fuse_token": True,
        "drop_loc": (3, 6, 9),
        "checkpoint_filename": "evit_small_patch16_0.9_fuse.pth",
        "published_macs": 4.0,
        "published_top1_in1k": 79.8,
        "published_top5_in1k": 95.0,
        "published_throughput": 3197.0,
        "description": "EViT-DeiT-Small with keep rate 0.9 and inattentive token fusion",
    },
    "evit_small_k07_fuse": {
        "model_fn": "deit_small_patch16_shrink_base",
        "keep_rate": 0.7,
        "fuse_token": True,
        "drop_loc": (3, 6, 9),
        "checkpoint_filename": "evit_small_patch16_0.7_fuse.pth",
        "published_macs": 3.0,
        "published_top1_in1k": 79.5,
        "published_top5_in1k": 94.8,
        "published_throughput": 4385.0,
        "description": "EViT-DeiT-Small with keep rate 0.7 and inattentive token fusion",
    },
    "evit_small_k05_fuse": {
        "model_fn": "deit_small_patch16_shrink_base",
        "keep_rate": 0.5,
        "fuse_token": True,
        "drop_loc": (3, 6, 9),
        "checkpoint_filename": "evit_small_patch16_0.5_fuse.pth",
        "published_macs": 2.3,
        "published_top1_in1k": 78.5,
        "published_top5_in1k": 94.2,
        "published_throughput": 5408.0,
        "description": "EViT-DeiT-Small with keep rate 0.5 and inattentive token fusion",
    },
}


def build_model(
    config_name: str,
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs,
) -> nn.Module:
    """Build a DeiT or EViT model using the official architecture."""
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config_name: {config_name}. Available: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[config_name]
    if cfg["model_fn"] == "deit_small_patch16_224":
        model = deit_small_patch16_224(pretrained=pretrained, num_classes=num_classes, **kwargs)
    elif cfg["model_fn"] == "deit_small_patch16_shrink_base":
        model = deit_small_patch16_shrink_base(
            pretrained=pretrained,
            num_classes=num_classes,
            base_keep_rate=cfg["keep_rate"],
            drop_loc=cfg["drop_loc"],
            fuse_token=cfg["fuse_token"],
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported model_fn: {cfg['model_fn']}")

    return model


def load_official_checkpoint(
    model: nn.Module,
    config_name: str,
    weights_dir: str = WEIGHTS_DIR,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load the official pretrained checkpoint for a model and verify strict key matching."""
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config_name: {config_name}")

    ckpt_file = os.path.join(weights_dir, MODEL_CONFIGS[config_name]["checkpoint_filename"])
    if not os.path.isfile(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # If the model has a different number of classes (e.g. 6 for BDD100K driving scene classification),
    # exclude the head weights from the checkpoint.
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        if model.head.weight.shape != state_dict["head.weight"].shape:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
            load_res = model.load_state_dict(state_dict, strict=False)
            return model, {
                "checkpoint_path": ckpt_file,
                "strict": False,
                "missing_keys": list(load_res.missing_keys),
                "unexpected_keys": list(load_res.unexpected_keys),
                "head_reinitialized": True,
            }

    load_res = model.load_state_dict(state_dict, strict=True)
    return model, {
        "checkpoint_path": ckpt_file,
        "strict": True,
        "missing_keys": list(load_res.missing_keys),
        "unexpected_keys": list(load_res.unexpected_keys),
        "head_reinitialized": False,
    }
