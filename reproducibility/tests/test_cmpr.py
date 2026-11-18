"""Tests for Correctness-Gated Margin Preservation Regularization (CMPR) loss."""

import pytest
import torch
from reproducibility.src.losses.cmpr import (
    compute_logit_margin,
    compute_cmpr_loss,
    CorrectnessGatedCMPRLoss,
)


def test_logit_margin_computation():
    """Verify margin m = z_y - max_{j != y} z_j."""
    logits = torch.tensor([[10.0, 4.0, 2.0], [1.0, 8.0, 5.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    margins = compute_logit_margin(logits, targets)

    # Item 0: target 0, z_0 = 10, max non-target = max(4, 2) = 4 -> margin = 6.0
    # Item 1: target 1, z_1 = 8, max non-target = max(1, 5) = 5 -> margin = 3.0
    expected = torch.tensor([6.0, 3.0], dtype=torch.float32)
    assert torch.allclose(margins, expected, atol=1e-5)


def test_cmpr_loss_teacher_gating():
    """Verify teacher correctness gate g_i = 1[argmax(z_t) == y_i]."""
    # Item 0: teacher correct (pred 0, gt 0)
    # Item 1: teacher incorrect (pred 2, gt 1)
    teacher_logits = torch.tensor([[10.0, 2.0, 1.0], [2.0, 1.0, 8.0]], dtype=torch.float32)
    student_logits = torch.tensor([[5.0, 2.0, 1.0], [2.0, 5.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    loss_fn = CorrectnessGatedCMPRLoss(margin_gamma=0.0, loss_type="squared_hinge")
    loss, metrics = loss_fn(student_logits, teacher_logits, targets)

    # Gate should be [True, False], so 1 valid sample
    assert metrics["valid_samples"] == 1.0
    # Only Item 0 should contribute: m_t = 10 - 2 = 8.0, m_s = 5 - 2 = 3.0, diff = 5.0, squared = 25.0
    assert abs(loss.item() - 25.0) < 1e-4


def test_cmpr_zero_correct_batch_safety():
    """Verify numerical stability when teacher is wrong on all samples in batch."""
    teacher_logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]], dtype=torch.float32)
    student_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=torch.float32)
    # Ground truth is reversed so teacher gets 0/2 correct
    targets = torch.tensor([0, 1], dtype=torch.long)

    loss_fn = CorrectnessGatedCMPRLoss(eps=1e-6)
    loss, metrics = loss_fn(student_logits, teacher_logits, targets)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() == 0.0
    assert metrics["valid_samples"] == 0.0


def test_cmpr_gradient_flow():
    """Verify student logits receive non-zero gradient while teacher has none."""
    teacher_logits = torch.tensor([[10.0, 2.0], [8.0, 1.0]], dtype=torch.float32)
    student_logits = torch.tensor([[4.0, 2.0], [3.0, 1.0]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0, 0], dtype=torch.long)

    loss_fn = CorrectnessGatedCMPRLoss()
    loss, _ = loss_fn(student_logits, teacher_logits, targets)
    loss.backward()

    assert student_logits.grad is not None
    assert torch.all(torch.isfinite(student_logits.grad))
    assert not torch.all(student_logits.grad == 0.0)
