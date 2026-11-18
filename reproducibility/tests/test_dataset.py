"""Tests for dataset splits, disjointness, and class balance."""

import os
import pytest
import torch

from reproducibility.src.data.validate_dataset import validate_splits
from reproducibility.src.losses.class_balanced_ce import compute_class_weights

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "reproducibility", "data", "manifests")


def test_split_counts_and_disjointness():
    """Verify exact sample counts and zero overlap across splits."""
    train_manifest = os.path.join(MANIFEST_DIR, "train.json")
    val_manifest = os.path.join(MANIFEST_DIR, "val.json")
    test_manifest = os.path.join(MANIFEST_DIR, "test.json")

    res = validate_splits(train_manifest, val_manifest, test_manifest)

    assert res["status"] == "PASS"
    assert res["disjointness"] is True
    assert len(res["overlaps"]) == 0

    assert res["splits"]["train"]["total_samples"] == 69634
    assert res["splits"]["val"]["total_samples"] == 9947
    assert res["splits"]["test"]["total_samples"] == 19902
    assert 69634 + 9947 + 19902 == 99483


def test_class_balanced_weights():
    """Verify programmatic class weights match the canonical tensor."""
    train_counts = [43581, 17414, 8105, 378, 27, 129]
    expected = [0.3393, 0.4268, 0.5167, 1.1118, 2.1507, 1.4547]

    weights = compute_class_weights(train_counts, power=0.25, normalize_by_mean=True)
    calculated = [round(float(w), 4) for w in weights.tolist()]

    assert len(calculated) == 6
    for c, e in zip(calculated, expected):
        assert abs(c - e) < 1e-3
