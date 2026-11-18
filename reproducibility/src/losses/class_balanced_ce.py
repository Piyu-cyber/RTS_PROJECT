from typing import List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPECTED_BDD_WEIGHTS = [0.3393, 0.4268, 0.5167, 1.1118, 2.1507, 1.4547]
VERIFIED_BDD_COUNTS = [43581, 17414, 8105, 378, 27, 129]


def compute_class_weights(
    class_counts: Union[List[int], np.ndarray, torch.Tensor],
    power: float = 0.25,
    beta: float = None,
    normalize_by_mean: bool = True,
) -> torch.Tensor:
    """Programmatically compute smoothed inverse-frequency class weights.
    
    Formula:
        w_c^{raw} = (N / (C * n_c)) ** power
        w_c = w_c^{raw} / mean(w^{raw})
        
    Args:
        class_counts: Sample counts for each class.
        power: Smoothing exponent (0.25 used for BDD100K).
        normalize_by_mean: Whether to normalize by mean so average weight is 1.0.
        
    Returns:
        1D float32 torch.Tensor of normalized weights.
    """
    if beta is not None:
        power = beta
    counts = np.array(class_counts, dtype=np.float64)
    total_samples = counts.sum()
    num_classes = len(counts)

    # Prevent division by zero
    counts = np.maximum(counts, 1.0)
    raw_weights = (total_samples / (num_classes * counts)) ** power

    if normalize_by_mean:
        weights = raw_weights / raw_weights.mean()
    else:
        weights = raw_weights

    return torch.tensor(weights, dtype=torch.float32)


# Alias
compute_class_balanced_weights = compute_class_weights


class ClassBalancedCrossEntropyLoss(nn.Module):
    """Class-Balanced Cross-Entropy loss module."""

    def __init__(
        self,
        class_counts: Union[List[int], torch.Tensor] = None,
        weights: torch.Tensor = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if weights is not None:
            self.register_buffer("weights", weights.float())
        elif class_counts is not None:
            w = compute_class_weights(class_counts)
            self.register_buffer("weights", w.float())
        else:
            w = compute_class_weights(VERIFIED_BDD_COUNTS)
            self.register_buffer("weights", w.float())

        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted class logits [B, C].
            targets: Ground-truth class indices [B].
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weights,
            label_smoothing=self.label_smoothing,
        )
