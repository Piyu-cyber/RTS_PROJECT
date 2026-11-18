"""Tests for token pruning and attention-weighted token fusion."""

import pytest
import torch
from reproducibility.src.models.token_pruning import prune_tokens_topk
from reproducibility.src.models.token_fusion import fuse_tokens_weighted


def test_prune_tokens_topk():
    """Verify top-k token selection and complement indexing."""
    B, N, D = 2, 10, 8
    tokens = torch.randn(B, N, D)
    # Give token 3 highest score, token 0 lowest
    scores = torch.arange(N, dtype=torch.float32).unsqueeze(0).repeat(B, 1)

    k = 4
    kept, discarded, kept_idx, disc_idx = prune_tokens_topk(tokens, scores, k=k)

    assert kept.shape == (B, k, D)
    assert discarded.shape == (B, N - k, D)
    assert kept_idx.shape == (B, k)
    assert disc_idx.shape == (B, N - k)

    # Top-4 scores should be tokens 6, 7, 8, 9
    for b in range(B):
        assert set(kept_idx[b].tolist()) == {6, 7, 8, 9}
        assert set(disc_idx[b].tolist()) == {0, 1, 2, 3, 4, 5}


def test_fuse_tokens_weighted():
    """Verify attention-weighted fusion produces exactly 1 summary token per batch item."""
    B, M, D = 2, 6, 8
    discarded_tokens = torch.ones(B, M, D) * 2.0
    discarded_scores = torch.ones(B, M) * 0.5

    fused = fuse_tokens_weighted(discarded_tokens, discarded_scores)

    assert fused.shape == (B, 1, D)
    # Since all tokens are 2.0, weighted average must be 2.0
    assert torch.allclose(fused, torch.ones(B, 1, D) * 2.0, atol=1e-5)
