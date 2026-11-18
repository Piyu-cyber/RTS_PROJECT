"""DeiT-Small model builder and ImageNet weight loader."""

import os
import sys
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

REF_EVIT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "reference", "evit"))
if REF_EVIT_DIR not in sys.path:
    sys.path.insert(0, REF_EVIT_DIR)

try:
    from evit import deit_small_patch16_224
except ImportError:
    from reference.evit.evit import deit_small_patch16_224


class ModelWithMeta(tuple):
    """Container that behaves both as a model (delegating calls and attributes) and a 2-tuple (model, meta)."""

    def __new__(cls, model, meta):
        return super().__new__(cls, (model, meta))

    def __getattr__(self, name):
        return getattr(self[0], name)

    def __call__(self, *args, **kwargs):
        return self[0](*args, **kwargs)


def build_deit_small(
    num_classes: int = 6,
    pretrained_path: Optional[str] = None,
    pretrained: bool = False,
    device: str = "cpu",
    **kwargs,
) -> ModelWithMeta:
    """Build a DeiT-Small (patch16, 224x224) vision transformer baseline."""
    model = deit_small_patch16_224(pretrained=False, num_classes=num_classes)
    load_meta = {"loaded": False, "pretrained_path": pretrained_path}

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
