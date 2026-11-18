"""EViT model builder with attention-guided token pruning and fusion."""

import os
import sys
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

REF_EVIT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "reference", "evit"))
if REF_EVIT_DIR not in sys.path:
    sys.path.insert(0, REF_EVIT_DIR)

try:
    from evit import deit_small_patch16_shrink_base
except ImportError:
    from reference.evit.evit import deit_small_patch16_shrink_base

from .deit import ModelWithMeta

EVIT_CONFIG_REGISTRY = {
    "k09": {"keep_rate": 0.9, "drop_loc": (3, 6, 9), "fuse_token": True, "gmacs": 4.03, "final_tokens": 148},
    "k07": {"keep_rate": 0.7, "drop_loc": (3, 6, 9), "fuse_token": True, "gmacs": 3.03, "final_tokens": 72},
    "k05": {"keep_rate": 0.5, "drop_loc": (3, 6, 9), "fuse_token": True, "gmacs": 2.31, "final_tokens": 28},
}


def build_evit_small(
    keep_rate_variant: Any = "k07",
    num_classes: int = 6,
    pretrained_path: Optional[str] = None,
    device: str = "cpu",
    **kwargs,
) -> ModelWithMeta:
    """Build an EViT-DeiT-Small model with attention-guided token pruning and fusion."""
    # Handle keep_rate as float
    if isinstance(keep_rate_variant, (float, int)):
        if keep_rate_variant >= 0.85:
            variant_key = "k09"
        elif keep_rate_variant >= 0.65:
            variant_key = "k07"
        else:
            variant_key = "k05"
    elif "keep_rate" in kwargs:
        k_val = kwargs.pop("keep_rate")
        if isinstance(k_val, (float, int)):
            if k_val >= 0.85:
                variant_key = "k09"
            elif k_val >= 0.65:
                variant_key = "k07"
            else:
                variant_key = "k05"
        else:
            variant_key = str(k_val)
    else:
        variant_key = str(keep_rate_variant)

    # Clean up redundant kwargs
    kwargs.pop("fuse_token", None)
    kwargs.pop("pretrained", None)

    if variant_key not in EVIT_CONFIG_REGISTRY:
        raise ValueError(f"Unknown EViT variant '{variant_key}'. Choose from {list(EVIT_CONFIG_REGISTRY.keys())}")

    cfg = EVIT_CONFIG_REGISTRY[variant_key]
    model = deit_small_patch16_shrink_base(
        pretrained=False,
        num_classes=num_classes,
        base_keep_rate=cfg["keep_rate"],
        drop_loc=cfg["drop_loc"],
        fuse_token=cfg["fuse_token"],
        **kwargs,
    )
    load_meta = {"loaded": False, "variant": variant_key, "cfg": cfg}

    if pretrained_path and os.path.isfile(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        if hasattr(model, "head") and isinstance(model.head, nn.Linear):
            if "head.weight" in state_dict and model.head.weight.shape != state_dict["head.weight"].shape:
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
                load_res = model.load_state_dict(state_dict, strict=False)
                load_meta.update({
                    "loaded": True,
                    "strict": False,
                    "head_reinitialized": True,
                    "missing_keys": list(load_res.missing_keys),
                    "unexpected_keys": list(load_res.unexpected_keys),
                })
            else:
                load_res = model.load_state_dict(state_dict, strict=False)
                load_meta.update({
                    "loaded": True,
                    "strict": True,
                    "head_reinitialized": False,
                    "missing_keys": list(load_res.missing_keys),
                    "unexpected_keys": list(load_res.unexpected_keys),
                })

    return ModelWithMeta(model, load_meta)
