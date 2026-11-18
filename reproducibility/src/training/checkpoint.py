import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from ..utils.hashing import compute_sha256


def save_training_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    val_metric: float,
    config: Dict[str, Any],
    file_path: str,
) -> str:
    """Save model checkpoint with training state and compute its SHA256 hash.
    
    Returns:
        Hexadecimal SHA256 checksum string.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "val_metric": val_metric,
        "config": config,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state["scheduler"] = scheduler.state_dict()

    torch.save(state, file_path)
    return compute_sha256(file_path)


def load_training_checkpoint(
    model: nn.Module,
    file_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model and optional optimizer/scheduler state from checkpoint."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

    ckpt = torch.load(file_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None and isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and isinstance(ckpt, dict) and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    return {
        "epoch": ckpt.get("epoch", 0) if isinstance(ckpt, dict) else 0,
        "step": ckpt.get("step", 0) if isinstance(ckpt, dict) else 0,
        "val_metric": ckpt.get("val_metric", 0.0) if isinstance(ckpt, dict) else 0.0,
        "sha256": compute_sha256(file_path),
    }
