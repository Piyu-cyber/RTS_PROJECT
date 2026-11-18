"""Checkpoint loader and validator for DeiT-Small and EViT checkpoints."""

import os
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from .models import CONFIGS, build_evit_model

DEFAULT_WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reference", "evit_weights"))


def get_checkpoint_path(config_name: str, weights_dir: str = DEFAULT_WEIGHTS_DIR) -> str:
    """Return the absolute path to the checkpoint file for a given configuration."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config_name '{config_name}'. Available: {list(CONFIGS.keys())}")
    filename = CONFIGS[config_name]["checkpoint_filename"]
    return os.path.join(weights_dir, filename)


def verify_checkpoint_exists(config_name: str, weights_dir: str = DEFAULT_WEIGHTS_DIR) -> bool:
    """Check if checkpoint file exists and is non-empty."""
    path = get_checkpoint_path(config_name, weights_dir)
    return os.path.isfile(path) and os.path.getsize(path) > 1024 * 1024


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load checkpoint state_dict into model and verify key alignment."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unrecognized checkpoint format in {checkpoint_path}")

    # Load state dict
    load_result = model.load_state_dict(state_dict, strict=True)

    info = {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "checkpoint_epoch": checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None,
        "is_perfect_match": len(load_result.missing_keys) == 0 and len(load_result.unexpected_keys) == 0,
    }

    return model, info


def get_loaded_model(
    config_name: str,
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Instantiate and load weights for a given configuration."""
    ckpt_path = get_checkpoint_path(config_name, weights_dir)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint for '{config_name}' not found at: {ckpt_path}")

    model = build_evit_model(config_name, pretrained=False)
    model, info = load_checkpoint_into_model(model, ckpt_path, device=device)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    info["param_count"] = param_count
    info["config_name"] = config_name
    info["checkpoint_path"] = ckpt_path
    return model, info
