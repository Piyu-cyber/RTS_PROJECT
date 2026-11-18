"""Token pruning functions for EViT architecture."""

import math
import torch
from typing import Tuple


def complement_idx(idx: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """Compute the complement indices of selected tokens within total_tokens."""
    device = idx.device
    B, k = idx.shape
    all_indices = torch.arange(total_tokens, device=device).unsqueeze(0).expand(B, total_tokens)

    mask = torch.ones((B, total_tokens), dtype=torch.bool, device=device)
    mask.scatter_(dim=1, index=idx, src=torch.zeros_like(idx, dtype=torch.bool))

    complement = all_indices[mask].view(B, total_tokens - k)
    return complement


def select_topk_tokens(
    cls_attn: torch.Tensor,
    keep_rate: float,
    num_tokens_override: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Select the top-k most attentive tokens based on CLS attention weights."""
    B, N = cls_attn.shape
    if num_tokens_override is not None:
        left_tokens = num_tokens_override
    else:
        left_tokens = math.ceil(keep_rate * N)

    left_tokens = max(1, min(left_tokens, N))

    _, topk_idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)
    compl_idx = complement_idx(topk_idx, N)

    return topk_idx, compl_idx, left_tokens


def prune_tokens_topk(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prune tokens by keeping top-k scores and partitioning remainder.

    Args:
        tokens: Tensor of shape [B, N, D].
        scores: Tensor of shape [B, N].
        k: Number of tokens to keep.

    Returns:
        Tuple of (kept_tokens, discarded_tokens, kept_indices, discarded_indices).
    """
    B, N, D = tokens.shape
    _, topk_idx = torch.topk(scores, k, dim=1, largest=True, sorted=True)
    compl_idx = complement_idx(topk_idx, N)

    kept = tokens.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
    discarded = tokens.gather(1, compl_idx.unsqueeze(-1).expand(-1, -1, D))

    return kept, discarded, topk_idx, compl_idx
