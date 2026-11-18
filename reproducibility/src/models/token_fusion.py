"""Token fusion functions for EViT architecture."""

import torch


def fuse_discarded_tokens(
    tokens: torch.Tensor,
    attention_weights: torch.Tensor,
    complement_indices: torch.Tensor,
) -> torch.Tensor:
    """Fuse inattentive (discarded) patch tokens into a single summary token via attention weighting.
    
    Args:
        tokens: Non-CLS visual tokens of shape [B, N_patches, C].
        attention_weights: CLS attention weights of shape [B, N_patches].
        complement_indices: Indices of discarded tokens of shape [B, N_discarded].
        
    Returns:
        Summary token tensor of shape [B, 1, C].
    """
    B, N, C = tokens.shape
    _, num_discarded = complement_indices.shape
    
    if num_discarded == 0:
        return torch.zeros((B, 0, C), dtype=tokens.dtype, device=tokens.device)

    # Gather discarded token vectors
    idx_expanded = complement_indices.unsqueeze(-1).expand(-1, -1, C)
    discarded_tokens = torch.gather(tokens, dim=1, index=idx_expanded)  # [B, num_discarded, C]

    # Gather corresponding attention weights
    discarded_attn = torch.gather(attention_weights, dim=1, index=complement_indices)  # [B, num_discarded]

    # Attention-weighted fusion
    fused_summary = torch.sum(discarded_tokens * discarded_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
    return fused_summary


def fuse_tokens_weighted(discarded_tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Weighted average of discarded tokens: [B, M, D] with scores [B, M] -> [B, 1, D]."""
    weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-9)
    return torch.sum(discarded_tokens * weights.unsqueeze(-1), dim=1, keepdim=True)
