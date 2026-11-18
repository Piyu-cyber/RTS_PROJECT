"""Correctness-Gated Margin Preservation Regularization (CMPR) loss module."""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_logit_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute margin m = z_y - max_{j != y} z_j."""
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, target.unsqueeze(1), True)
    true_l = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    other_max = logits.masked_fill(mask, -1e9).max(dim=1).values
    return true_l - other_max


def compute_cmpr_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 0.0,
    teacher_gate: bool = True,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, int]:
    """Compute Class-Margin-Preserving Regularization (CMPR) loss."""
    logits_s = logits_s.float()
    logits_t = logits_t.float().detach()

    device = logits_s.device

    m_s = compute_logit_margin(logits_s, target)
    m_t = compute_logit_margin(logits_t, target)

    margin_diff = m_t - m_s - gamma
    hinge = torch.clamp(margin_diff, min=0.0) ** 2

    if teacher_gate:
        t_pred = logits_t.argmax(dim=1)
        gate = (t_pred == target).float()
        num_teacher_correct = int(gate.sum().item())

        if num_teacher_correct == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss = (gate * hinge).sum() / (gate.sum() + epsilon)
    else:
        num_teacher_correct = int((logits_t.argmax(dim=1) == target).sum().item())
        loss = hinge.mean()

    return loss, num_teacher_correct


class CorrectnessGatedCMPRLoss(nn.Module):
    """Correctness-Gated CMPR Loss Module."""

    def __init__(
        self,
        margin_gamma: float = 0.0,
        loss_type: str = "squared_hinge",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.margin_gamma = margin_gamma
        self.loss_type = loss_type
        self.eps = eps

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss, n_correct = compute_cmpr_loss(
            logits_s=student_logits,
            logits_t=teacher_logits,
            target=targets,
            gamma=self.margin_gamma,
            teacher_gate=True,
            epsilon=self.eps,
        )
        metrics = {
            "valid_samples": float(n_correct),
            "cmpr_loss": float(loss.item()),
        }
        return loss, metrics


class CMPRLoss(nn.Module):
    """Module wrapper for Class-Margin-Preserving Regularization."""

    def __init__(self, lambda_cmpr: float = 0.5, gamma: float = 0.0, teacher_gate: bool = True):
        super().__init__()
        self.lambda_cmpr = lambda_cmpr
        self.gamma = gamma
        self.teacher_gate = teacher_gate

    def forward(
        self,
        logits_s: torch.Tensor,
        logits_t: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        loss, n_correct = compute_cmpr_loss(
            logits_s,
            logits_t,
            target,
            gamma=self.gamma,
            teacher_gate=self.teacher_gate,
        )
        return self.lambda_cmpr * loss, n_correct


class CompositeTrainingObjective(nn.Module):
    """Combines Cross-Entropy (or Class-Balanced CE) with CMPR teacher regularization."""

    def __init__(
        self,
        base_criterion: nn.Module,
        use_cmpr: bool = False,
        lambda_cmpr: float = 0.5,
        gamma: float = 0.0,
        teacher_gate: bool = True,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.use_cmpr = use_cmpr
        self.cmpr_module = CMPRLoss(lambda_cmpr=lambda_cmpr, gamma=gamma, teacher_gate=teacher_gate) if use_cmpr else None

    def forward(
        self,
        logits_s: torch.Tensor,
        target: torch.Tensor,
        logits_t: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        base_loss = self.base_criterion(logits_s, target)

        if self.use_cmpr and logits_t is not None:
            cmpr_loss, n_correct = self.cmpr_module(logits_s, logits_t, target)
            total_loss = base_loss + cmpr_loss
            return total_loss, base_loss, cmpr_loss, n_correct
        else:
            return base_loss, base_loss, torch.tensor(0.0, device=logits_s.device), 0
