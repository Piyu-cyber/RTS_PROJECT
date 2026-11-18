"""Tests for DeiT and EViT model forward pass, output shapes, and sequence lengths."""

import pytest
import torch
from reproducibility.src.models.deit import build_deit_small
from reproducibility.src.models.evit import build_evit_small


def test_deit_forward_shape():
    """Verify DeiT-Small forward pass output shape for 6 classes."""
    model = build_deit_small(num_classes=6, pretrained=False).eval()
    x = torch.zeros(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 6)


def test_evit_forward_shapes():
    """Verify EViT forward pass output shape across keep-rates k in {0.9, 0.7, 0.5}."""
    for k in [0.9, 0.7, 0.5]:
        model = build_evit_small(keep_rate=k, num_classes=6, fuse_token=True).eval()
        x = torch.zeros(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 6)


def test_evit_token_lengths_after_pruning():
    """Verify exact token sequence lengths after blocks 3, 6, 9."""
    expected_final_lengths = {
        0.9: 148,
        0.7: 72,
        0.5: 28,
    }

    for k, expected_len in expected_final_lengths.items():
        model = build_evit_small(keep_rate=k, num_classes=6, fuse_token=True).eval()
        x = torch.zeros(1, 3, 224, 224)

        recorded_shapes = []

        def make_hook(block_idx):
            def hook(module, inp, output):
                tensor = output[0] if isinstance(output, tuple) else output
                recorded_shapes.append((block_idx, tensor.shape[1]))
            return hook

        for b_idx in [3, 6, 9]:
            model.blocks[b_idx].register_forward_hook(make_hook(b_idx))

        with torch.no_grad():
            _ = model(x)

        final_tokens = recorded_shapes[-1][1]
        assert final_tokens == expected_len, f"Expected {expected_len} tokens for k={k}, got {final_tokens}"
