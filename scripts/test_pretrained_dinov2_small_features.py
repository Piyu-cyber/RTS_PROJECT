"""Strict pretrained DINOv2-Small feature-extraction smoke test.

Downloads only the official Meta DINOv2 ViT-S/14 checkpoint when --download is
given. It never substitutes random weights for a missing pretrained checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any

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
from lib.patching.patch import patch_dinov2_convatt


OFFICIAL_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
CHECKPOINT_PATH = REFERENCE_ROOT / "weights" / "dinov2_vits14_pretrain.pth"
INPUT_SIZE = 224
PRETRAINED_SIZE = 518
PATCH_SIZE = 14
EMBED_DIM = 384
DEPTH = 12
NUM_HEADS = 6
PATCH_BLOCK = 3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_config() -> Config:
    return Config(
        float32_matmul_precision="highest",
        fp16=False,
        img_size=INPUT_SIZE,
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
            "pretrained_img_size": PRETRAINED_SIZE,
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


def download_checkpoint() -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official checkpoint from: {OFFICIAL_URL}")
    try:
        torch.hub.download_url_to_file(OFFICIAL_URL, str(CHECKPOINT_PATH), progress=True)
    except Exception as error:
        raise RuntimeError(f"Official checkpoint download failed: {error}") from error
    require(CHECKPOINT_PATH.is_file() and CHECKPOINT_PATH.stat().st_size > 0, "Downloaded checkpoint is missing or empty")


def checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dictionary, found {type(checkpoint).__name__}")
    for key in ("model", "state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            print(f"Checkpoint container key: {key}")
            checkpoint = nested
            break
    if not all(isinstance(key, str) for key in checkpoint):
        raise TypeError("Checkpoint state dictionary contains a non-string key")
    if not all(torch.is_tensor(value) for value in checkpoint.values()):
        raise TypeError("Checkpoint state dictionary contains a non-tensor value")
    return checkpoint


def print_keys(label: str, keys: list[str]) -> None:
    print(f"{label}: {len(keys)}")
    for key in keys[:20]:
        print(f"  {key}")
    if len(keys) > 20:
        print(f"  ... and {len(keys) - 20} more")


def strict_load(model: DinoVisionTransformer) -> None:
    require(CHECKPOINT_PATH.is_file(), f"Checkpoint not found: {CHECKPOINT_PATH}. Re-run with --download.")
    print(f"Checkpoint provenance: Meta official DINOv2 ViT-S/14 at {OFFICIAL_URL}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Checkpoint size: {CHECKPOINT_PATH.stat().st_size:,} bytes")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    state_dict = checkpoint_state_dict(checkpoint)

    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    mismatched = sorted(
        key for key in set(expected) & set(state_dict) if expected[key].shape != state_dict[key].shape
    )
    print_keys("Missing keys", missing)
    print_keys("Unexpected keys", unexpected)
    print_keys("Shape-mismatched keys", mismatched)
    require(not missing, "Checkpoint is missing model parameters")
    require(not unexpected, "Checkpoint contains unexpected parameters")
    require(not mismatched, "Checkpoint parameter shapes do not match the DINOv2-Small model")

    result = model.load_state_dict(state_dict, strict=True)
    require(not result.missing_keys and not result.unexpected_keys, "Strict load returned incompatible keys")
    print("Strict checkpoint load: PASS")


def extract_spatial_features(model: DinoVisionTransformer, image: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        features = model.get_intermediate_layers(image, [DEPTH - 1], reshape=True, return_class_token=False)
    require(len(features) == 1, f"Expected one feature tensor, got {len(features)}")
    spatial = features[0]
    require(spatial.shape == (1, EMBED_DIM, 16, 16), f"Unexpected spatial feature shape: {tuple(spatial.shape)}")
    require(torch.isfinite(spatial).all().item(), "Spatial features contain NaN or infinity")
    return spatial


def capture_replaced_attention_weights(model: DinoVisionTransformer) -> dict[str, torch.Tensor]:
    """Keep the pretrained tensors that CnvAttnDW_blockwise must preserve."""
    state = model.blocks[PATCH_BLOCK].attn.state_dict()
    _, _, value_weight = torch.chunk(state["qkv.weight"], 3)
    captured = {
        "v_proj.weight": value_weight.detach().clone(),
        "proj.weight": state["proj.weight"].detach().clone(),
    }
    if "qkv.bias" in state:
        _, _, value_bias = torch.chunk(state["qkv.bias"], 3)
        captured["v_proj.bias"] = value_bias.detach().clone()
    if "proj.bias" in state:
        captured["proj.bias"] = state["proj.bias"].detach().clone()
    return captured


def verify_replacement_weight_transfer(replacement: CnvAttnDW_blockwise, expected: dict[str, torch.Tensor]) -> None:
    actual = {
        "v_proj.weight": replacement.v_proj.weight,
        "proj.weight": replacement.proj.weight,
    }
    if replacement.v_proj.bias is not None:
        actual["v_proj.bias"] = replacement.v_proj.bias
    if replacement.proj.bias is not None:
        actual["proj.bias"] = replacement.proj.bias
    require(set(actual) == set(expected), "Replacement has an unexpected pretrained-weight layout")
    for name, expected_tensor in expected.items():
        expected_on_actual_device = expected_tensor.to(actual[name].device)
        require(
            torch.equal(actual[name], expected_on_actual_device),
            f"Pretrained weight transfer failed for {name}",
        )
    print("Replacement pretrained-weight transfer: PASS (V projection and output projection)")
    print("New replacement parameters: qk_conv, cls_all, all_cls")


def verify_device_placement(model: DinoVisionTransformer, replacement: CnvAttnDW_blockwise, device: torch.device) -> None:
    replacement_tensors = list(replacement.named_parameters()) + list(replacement.named_buffers())
    cpu_tensors = [name for name, tensor in replacement_tensors if tensor.device.type == "cpu"]
    wrong_device = [name for name, tensor in replacement_tensors if tensor.device != device]
    print(f"Replacement tensor devices: {sorted({str(tensor.device) for _, tensor in replacement_tensors})}")
    print(f"Replacement CPU tensors participating in forward: {cpu_tensors or 'none'}")
    require(not cpu_tensors, "Replacement module still has CPU tensors")
    require(not wrong_device, f"Replacement tensors are not on {device}: {wrong_device}")

    model_cpu_tensors = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers()) if tensor.device.type == "cpu"]
    require(not model_cpu_tensors, f"Model still has CPU tensors: {model_cpu_tensors[:20]}")
    print("Full model device placement: PASS")


def main(download: bool) -> None:
    torch.manual_seed(17)
    if download and not CHECKPOINT_PATH.exists():
        download_checkpoint()
    require(CHECKPOINT_PATH.is_file(), f"Checkpoint not found: {CHECKPOINT_PATH}. Run with --download.")

    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "CUDA GPU is required for this phase but is not available")
    print("DINOv2-Small pretrained spatial-feature test")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Architecture: ViT-S/14, embed_dim={EMBED_DIM}, depth={DEPTH}, num_heads={NUM_HEADS}")
    print(f"Input: (1, 3, {INPUT_SIZE}, {INPUT_SIZE})")

    config = make_config()
    model = build_model(config)
    strict_load(model)
    model = model.to(device).eval()
    image = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)

    original_features = extract_spatial_features(model, image)
    print(f"Unmodified spatial features: {tuple(original_features.shape)}")

    before_class = type(model.blocks[PATCH_BLOCK].attn).__name__
    expected_replacement_weights = capture_replaced_attention_weights(model)
    patch_dinov2_convatt(model, config)
    require(isinstance(model.blocks[PATCH_BLOCK].attn, CnvAttnDW_blockwise), "Target block did not become CnvAttnDW_blockwise")
    for index, block in enumerate(model.blocks):
        if index != PATCH_BLOCK:
            require(type(block.attn).__name__ == before_class, f"Block {index} changed unexpectedly")
    print(f"Patched block: {PATCH_BLOCK}; {before_class} -> {type(model.blocks[PATCH_BLOCK].attn).__name__}")
    replacement = model.blocks[PATCH_BLOCK].attn
    verify_replacement_weight_transfer(replacement, expected_replacement_weights)

    # The patcher creates a new CPU module, so recursively move the updated model.
    model = model.to(device).eval()
    replacement = model.blocks[PATCH_BLOCK].attn
    verify_device_placement(model, replacement, device)

    patched_features = extract_spatial_features(model, image)
    require(patched_features.shape == original_features.shape, "Patched spatial feature shape changed")
    print(f"Patched spatial features: {tuple(patched_features.shape)}")
    print("PASS: official pretrained DINOv2-Small loaded strictly and produced finite spatial features before and after DWConv patching.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download the official checkpoint if it is absent")
    arguments = parser.parse_args()
    try:
        main(arguments.download)
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from error
