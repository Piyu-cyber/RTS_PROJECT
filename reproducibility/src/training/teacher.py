import os
import torch
import torch.nn as nn
from typing import Optional
from ..models.deit import build_deit_small


def load_frozen_teacher(
    checkpoint_path: str,
    device: torch.device = None,
    num_classes: int = 6,
) -> nn.Module:
    """Load pretrained DeiT-Small teacher model in strict frozen evaluation mode.
    
    Args:
        checkpoint_path: Path to teacher checkpoint (.pth).
        device: Device to map model.
        num_classes: Number of classification classes.
        
    Returns:
        Frozen nn.Module ready for logit generation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = build_deit_small(num_classes=num_classes)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


@torch.no_grad()
def get_teacher_logits(teacher_model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Extract detached logits from frozen teacher."""
    teacher_model.eval()
    return teacher_model(images).detach()
