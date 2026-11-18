"""Implementation of Section 3.3 Block Selection Criterion (Sigma_h and Sigma_b).

Reference:
  Carmelo Scribano et al., "Accelerating Vision Foundation Models with Drop-in Depthwise Convolution," arXiv:2605.22132.

Methodology:
  1. For each calibration image X, extract attention map E_h(X) for head h in block b.
  2. Maintain running mean and M2 using Welford's algorithm to compute pointwise standard deviations std(E_h(X)).
  3. Compute Sigma_h as the sum of pointwise standard deviations across the N x N attention matrix.
  4. Average Sigma_h across heads within block b to obtain Sigma_b.
  5. Rank blocks in ascending order of Sigma_b (lower Sigma_b = lower variability = better candidates for DWConv replacement).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

os.environ["XFORMERS_DISABLED"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "reference"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

from baseline.train_segmentation import (
    DEPTH,
    EMBED_DIM,
    PATCH_SIZE,
    RunConfig,
    build_backbone,
    build_datasets,
    features,
    require,
    set_seed,
    strict_load_backbone,
)
from realtime.runtime_inference import PHASE4_CHECKPOINT

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase10a_sigma_selection"


class WelfordAccumulator:
    """Pointwise online mean and variance accumulator using Welford's algorithm."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Add a single observation x of shape self.shape."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def std(self, ddof: int = 1) -> np.ndarray:
        """Compute pointwise standard deviation."""
        if self.count <= ddof:
            return np.zeros(self.shape, dtype=np.float64)
        variance = self.M2 / (self.count - ddof)
        return np.sqrt(np.maximum(variance, 0.0))


class AttentionMapExtractor:
    """Context manager to hook and extract un-softmaxed/softmaxed attention maps from DINOv2 blocks."""

    def __init__(self, backbone: nn.Module) -> None:
        self.backbone = backbone
        self.captured_maps: dict[int, Tensor] = {}
        self.hooks = []

    def _make_hook(self, block_idx: int):
        def hook(module, input, output):
            # In MemEffAttention.forward:
            # x is input[0] of shape (B, N, C)
            x = input[0]
            B, N, C = x.shape
            num_heads = module.num_heads
            qkv = module.qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0] * module.scale, qkv[1], qkv[2]
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            # attn shape: (B, num_heads, N, N)
            self.captured_maps[block_idx] = attn.detach()
        return hook

    def __enter__(self):
        self.captured_maps.clear()
        for idx, block in enumerate(self.backbone.blocks):
            h = block.attn.register_forward_hook(self._make_hook(idx))
            self.hooks.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def compute_sigma_scores(
    backbone: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 100,
) -> tuple[dict[tuple[int, int], float], dict[int, float], dict[str, Any]]:
    """Compute Sigma_h and Sigma_b across calibration samples.

    Returns:
      (sigma_h_dict, sigma_b_dict, metadata)
    """
    backbone.eval()
    num_blocks = len(backbone.blocks)
    num_heads = backbone.blocks[0].attn.num_heads

    # Initialize accumulators for each (block, head)
    # Target sequence length N = 1 + (224/14)^2 = 257
    N = 1 + (224 // PATCH_SIZE) ** 2
    accumulators: dict[tuple[int, int], WelfordAccumulator] = {
        (b, h): WelfordAccumulator((N, N)) for b in range(num_blocks) for h in range(num_heads)
    }

    t0 = time.perf_counter()
    samples_processed = 0

    with torch.inference_mode(), AttentionMapExtractor(backbone) as extractor:
        for batch_idx, (images, _) in enumerate(calibration_loader):
            if samples_processed >= max_samples:
                break
            images = images.to(device)
            B = images.shape[0]

            # Forward pass through backbone to trigger hooks
            _ = backbone.prepare_tokens_with_masks(images)
            for block in backbone.blocks:
                _ = block(_)

            # Extract captured attention maps for this batch
            for b_idx in range(num_blocks):
                attn_tensor = extractor.captured_maps[b_idx].cpu().numpy()  # (B, num_heads, N, N)
                for sample_i in range(min(B, max_samples - samples_processed)):
                    for h_idx in range(num_heads):
                        head_map = attn_tensor[sample_i, h_idx]  # (N, N)
                        accumulators[(b_idx, h_idx)].update(head_map)

            samples_processed += min(B, max_samples - samples_processed)

    t1 = time.perf_counter()
    compute_time_s = t1 - t0

    # Aggregate Sigma_h: sum of pointwise standard deviations across the N x N map
    sigma_h: dict[tuple[int, int], float] = {}
    for (b, h), acc in accumulators.items():
        std_map = acc.std(ddof=1)
        sigma_h[(b, h)] = float(np.sum(std_map))

    # Aggregate Sigma_b: average Sigma_h across heads in block b
    sigma_b: dict[int, float] = {}
    for b in range(num_blocks):
        head_scores = [sigma_h[(b, h)] for h in range(num_heads)]
        sigma_b[b] = float(np.mean(head_scores))

    metadata = {
        "samples_processed": samples_processed,
        "num_blocks": num_blocks,
        "num_heads_per_block": num_heads,
        "map_resolution": [N, N],
        "compute_time_seconds": round(compute_time_s, 2),
    }

    return sigma_h, sigma_b, metadata


def rank_and_select_blocks(sigma_b: dict[int, float], k: int) -> list[int]:
    """Rank blocks by Sigma_b in ascending order and select the top k blocks."""
    sorted_blocks = sorted(sigma_b.keys(), key=lambda b: sigma_b[b])
    return sorted(sorted_blocks[:k])
