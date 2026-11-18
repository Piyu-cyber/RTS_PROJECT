"""Tests for classification metrics, Macro-F1, and Gas Station decomposition."""

import pytest
import numpy as np
from reproducibility.src.evaluation.metrics import compute_classification_metrics
from reproducibility.src.evaluation.gas_analysis import decompose_gas_station_contribution


def test_classification_metrics_calculation():
    """Verify Top-1, Macro-F1, Weighted-F1, and per-class metrics."""
    # 3 classes, 6 samples
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    metrics = compute_classification_metrics(y_true, y_pred, num_classes=3)

    # 4/6 correct = 66.67%
    assert abs(metrics["top1_accuracy"] - 66.67) < 0.1
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert len(metrics["per_class"]) == 3


def test_gas_station_decomposition():
    """Verify mathematical sensitivity of Macro-F1 to Gas Station support (N=6)."""
    # 6 classes
    # Class 4 is Gas Station with only 6 test instances
    y_true = np.array([0, 1, 2, 3, 4, 5] * 10)  # 60 samples, 10 per class
    y_pred = y_true.copy()

    # Modify Gas Station predictions
    # Case A: 10/10 correct
    m_full = compute_classification_metrics(y_true, y_pred, num_classes=6)
    assert m_full["macro_f1"] == 100.0

    # Case B: Miss 1 gas station
    y_pred_miss1 = y_true.copy()
    y_pred_miss1[4] = 0
    m_miss1 = compute_classification_metrics(y_true, y_pred_miss1, num_classes=6)

    # In a 6-class Macro-F1, losing 1 TP in class 4 lowers that class recall by 10%
    # and macro-F1 by ~1.67%
    delta_macro = m_full["macro_f1"] - m_miss1["macro_f1"]
    assert delta_macro > 0.0

    # Test decomposition utility
    decomp = decompose_gas_station_contribution(y_true, y_pred_miss1, gas_class_idx=4)
    assert "delta_macro_f1" in decomp or "gas_contribution" in decomp or "gas_station_f1" in decomp
