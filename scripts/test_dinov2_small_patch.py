"""CPU-only architecture smoke test for DWConv_VFM DINOv2-Small patching.

This test deliberately uses random weights and a tiny valid image.  It checks
the repository's model and patching APIs without checkpoints, datasets, or GPU.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ["XFORMERS_DISABLED"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "reference"
if not REFERENCE_ROOT.is_dir():
    raise RuntimeError(f"Reference repository not found: {REFERENCE_ROOT}")
sys.path.insert(0, str(REFERENCE_ROOT))

import torch

from lib.config import Config
from lib.convatt.convatt_dw import CnvAttnDW_blockwise
from lib.models.dino_vision_transformer import DinoVisionTransformer
from lib.models.layers import MemEffAttention
from lib.patching.patch import patch_dinov2_convatt


PATCH_BLOCK = 3
IMAGE_SIZE = 28
PATCH_SIZE = 14
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_config() -> Config:
    """Create the smallest Config accepted by the reference schema."""
    return Config(
        float32_matmul_precision="highest",
        fp16=False,
        img_size=IMAGE_SIZE,
        backbone={
            "backbone": "dinov2",
            "backbone_pretrained_path": None,
            "output_layers": [DEPTH - 1],
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "depth": DEPTH,
            "num_heads": NUM_HEADS,
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.0,
            "include_cls_token": True,
            "pretrained_img_size": IMAGE_SIZE,
            "learnable_pos_embed": False,
            "num_register_tokens": 0,
        },
        convatt={
            "conv_style": "depthwise",
            "conv_granularity": "blockwise",
            "conv_selection": "manual",
            "head_ensembling": False,
            "blocks": [PATCH_BLOCK],
            "weight_decay": False,
            "layer_decay": False,
            "conv_init": False,
        },
    )


def build_model(config: Config) -> DinoVisionTransformer:
    backbone = config.backbone
    return DinoVisionTransformer(
        img_size=backbone.pretrained_img_size,
        patch_size=backbone.patch_size,
        embed_dim=backbone.embed_dim,
        depth=backbone.depth,
        num_heads=backbone.num_heads,
        mlp_ratio=backbone.mlp_ratio,
        drop_path_rate=backbone.drop_path_rate,
        num_register_tokens=backbone.num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0,
        block_chunks=0,
        init_values=1e-5,
        use_cls=backbone.include_cls_token,
    )


def main() -> None:
    torch.manual_seed(7)
    config = make_config()
    model = build_model(config).cpu().eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print("DINOv2-Small DWConv patch smoke test")
    print(
        "Configuration: "
        f"image_size={IMAGE_SIZE}, patch_size={PATCH_SIZE}, embed_dim={EMBED_DIM}, "
        f"depth={DEPTH}, num_heads={NUM_HEADS}, mlp_ratio=4.0"
    )
    print(f"Parameter count before patch: {parameter_count:,}")
    print(f"Target block: {PATCH_BLOCK}")

    require(len(model.blocks) == DEPTH, f"Expected {DEPTH} blocks, found {len(model.blocks)}")
    for index, block in enumerate(model.blocks):
        require(
            isinstance(block.attn, MemEffAttention),
            f"Block {index} expected MemEffAttention before patch, found {type(block.attn).__name__}",
        )
        require(
            block.attn.num_heads == NUM_HEADS,
            f"Block {index} expected {NUM_HEADS} heads, found {block.attn.num_heads}",
        )

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        original_output = model(dummy_input)
    require(original_output.shape == (1, EMBED_DIM), f"Unexpected original output shape: {tuple(original_output.shape)}")
    require(torch.isfinite(original_output).all().item(), "Original output contains NaN or infinity")
    print(f"Original output shape: {tuple(original_output.shape)}")

    before_classes = [type(block.attn).__name__ for block in model.blocks]
    print(f"Target attention before patch: {before_classes[PATCH_BLOCK]}")
    patch_dinov2_convatt(model, config)
    after_classes = [type(block.attn).__name__ for block in model.blocks]
    print(f"Target attention after patch: {after_classes[PATCH_BLOCK]}")

    require(
        isinstance(model.blocks[PATCH_BLOCK].attn, CnvAttnDW_blockwise),
        "Target block was not replaced with CnvAttnDW_blockwise",
    )
    for index, attention_class in enumerate(after_classes):
        if index != PATCH_BLOCK:
            require(
                attention_class == before_classes[index],
                f"Block {index} changed unexpectedly: {before_classes[index]} -> {attention_class}",
            )

    with torch.no_grad():
        patched_output = model(dummy_input)
    require(patched_output.shape == original_output.shape, "Patched output shape is not compatible with original output")
    require(torch.isfinite(patched_output).all().item(), "Patched output contains NaN or infinity")
    print(f"Patched output shape: {tuple(patched_output.shape)}")
    print("PASS: DINOv2-Small construction and one-block depthwise patching succeeded.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from error
