"""Model factory for DeiT-Small baseline and EViT variants."""

import os
import sys

# Ensure reference/evit is in sys.path
EVIT_REF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reference", "evit"))
if EVIT_REF_DIR not in sys.path:
    sys.path.insert(0, EVIT_REF_DIR)

import torch
import torch.nn as nn
from evit import deit_small_patch16_224, deit_small_patch16_shrink_base


CONFIGS = {
    "deit_small_baseline": {
        "model_fn": "deit_small_patch16_224",
        "keep_rate": 1.0,
        "fuse_token": False,
        "drop_loc": (),
        "checkpoint_filename": "deit_small_patch16_224.pth",
        "description": "Standard pretrained DeiT-Small, full token sequence (196 tokens)",
    },
    "evit_small_k09_fuse": {
        "model_fn": "deit_small_patch16_shrink_base",
        "keep_rate": 0.9,
        "fuse_token": True,
        "drop_loc": (3, 6, 9),
        "checkpoint_filename": "evit_small_patch16_0.9_fuse.pth",
        "description": "EViT-DeiT-Small with keep rate 0.9 and inattentive token fusion",
    },
    "evit_small_k07_fuse": {
        "model_fn": "deit_small_patch16_shrink_base",
        "keep_rate": 0.7,
        "fuse_token": True,
        "drop_loc": (3, 6, 9),
        "checkpoint_filename": "evit_small_patch16_0.7_fuse.pth",
        "description": "EViT-DeiT-Small with keep rate 0.7 and inattentive token fusion",
    },
    "evit_small_k05_fuse": {
        "model_fn": "deit_small_patch16_shrink_base",
        "keep_rate": 0.5,
        "fuse_token": True,
        "drop_loc": (3, 6, 9),
        "checkpoint_filename": "evit_small_patch16_0.5_fuse.pth",
        "description": "EViT-DeiT-Small with keep rate 0.5 and inattentive token fusion",
    },
}


def build_evit_model(config_name: str, pretrained: bool = False, **kwargs) -> nn.Module:
    """Build an EViT or DeiT model based on predefined configuration."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config_name '{config_name}'. Available: {list(CONFIGS.keys())}")

    cfg = CONFIGS[config_name]
    if cfg["model_fn"] == "deit_small_patch16_224":
        model = deit_small_patch16_224(pretrained=pretrained, **kwargs)
    elif cfg["model_fn"] == "deit_small_patch16_shrink_base":
        model = deit_small_patch16_shrink_base(
            pretrained=pretrained,
            base_keep_rate=cfg["keep_rate"],
            drop_loc=cfg["drop_loc"],
            fuse_token=cfg["fuse_token"],
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported model function: {cfg['model_fn']}")

    return model
