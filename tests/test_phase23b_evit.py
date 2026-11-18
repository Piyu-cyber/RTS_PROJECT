"""Unit and regression tests for Phase 23B-1 EViT reproduction and BDD100K driving scene classification."""

import os
import csv
import json
import unittest
import numpy as np
import torch
import torch.nn as nn

from src.phase23b.models import MODEL_CONFIGS, build_model, load_official_checkpoint
from src.phase23b.bdd_dataset import get_bdd_datasets, load_bdd_split_samples, NUM_BDD_CLASSES, BDD_CLASSES
from src.phase23b.bdd_finetuning import compute_metrics


class TestPhase23BEViT(unittest.TestCase):
    def test_model_configs_integrity(self):
        """Verify all 4 configurations are defined with expected metadata."""
        expected_configs = [
            "deit_small_baseline",
            "evit_small_k09_fuse",
            "evit_small_k07_fuse",
            "evit_small_k05_fuse",
        ]
        for name in expected_configs:
            self.assertIn(name, MODEL_CONFIGS)
            cfg = MODEL_CONFIGS[name]
            self.assertIn("keep_rate", cfg)
            self.assertIn("fuse_token", cfg)
            self.assertIn("drop_loc", cfg)
            self.assertIn("checkpoint_filename", cfg)
            self.assertIn("published_top1_in1k", cfg)

    def test_build_model_1000_classes(self):
        """Verify model output shape is [B, 1000] for ImageNet classification."""
        for cfg_name in MODEL_CONFIGS.keys():
            model = build_model(cfg_name, num_classes=1000, pretrained=False)
            self.assertIsInstance(model, nn.Module)
            dummy = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                out = model(dummy)
            self.assertEqual(out.shape, (2, 1000), f"Model {cfg_name} failed output shape test")

    def test_build_model_6_classes(self):
        """Verify model output shape is [B, 6] for BDD100K driving scene classification."""
        for cfg_name in MODEL_CONFIGS.keys():
            model = build_model(cfg_name, num_classes=NUM_BDD_CLASSES, pretrained=False)
            self.assertIsInstance(model, nn.Module)
            dummy = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                out = model(dummy)
            self.assertEqual(out.shape, (2, NUM_BDD_CLASSES), f"Model {cfg_name} failed {NUM_BDD_CLASSES}-class shape test")

    def test_token_sequences_across_keep_rates(self):
        """Verify expected token counts across stages for all models."""
        expected_tokens = {
            "deit_small_baseline": [196] * 12,
            "evit_small_k09_fuse": [196, 196, 196, 178, 178, 178, 162, 162, 162, 147, 147, 147],
            "evit_small_k07_fuse": [196, 196, 196, 139, 139, 139, 99, 99, 99, 71, 71, 71],
            "evit_small_k05_fuse": [196, 196, 196, 99, 99, 99, 51, 51, 51, 27, 27, 27],
        }
        dummy = torch.randn(1, 3, 224, 224)
        for cfg_name, expected_seq in expected_tokens.items():
            model = build_model(cfg_name, num_classes=1000, pretrained=False)
            model.eval()
            with torch.no_grad():
                _, left_tokens, _ = model.forward_features(dummy)
            self.assertEqual(left_tokens, expected_seq, f"Mismatch in token sequence for {cfg_name}")

    def test_bdd100k_dataset_splits_and_leakage(self):
        """Verify BDD100K train/val/test sample counts, disjointness, and valid label ranges."""
        train_ds, val_ds, test_ds, meta = get_bdd_datasets(dataset_root="data/bdd100k", seed=42)

        self.assertEqual(meta["dataset"], "BDD100K")
        self.assertEqual(meta["num_classes"], 6)
        self.assertGreater(meta["num_train"], 0)
        self.assertGreater(meta["num_val"], 0)
        self.assertGreater(meta["num_test"], 0)

        # Ensure no data leakage between train, val, and test
        train_names = set(img_name for img_name, _ in train_ds.samples)
        val_names = set(img_name for img_name, _ in val_ds.samples)
        test_names = set(img_name for img_name, _ in test_ds.samples)

        self.assertEqual(len(train_names.intersection(val_names)), 0, "Found overlap between train and val")
        self.assertEqual(len(train_names.intersection(test_names)), 0, "Found overlap between train and test")
        self.assertEqual(len(val_names.intersection(test_names)), 0, "Found overlap between val and test")

        # Check label range
        for _, class_id in train_ds.samples:
            self.assertTrue(0 <= class_id < NUM_BDD_CLASSES, f"Invalid label {class_id} in train split")

    def test_compute_metrics_accuracy_and_f1(self):
        """Test top-1 accuracy and macro-F1 on controlled synthetic predictions."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])

        metrics = compute_metrics(y_true, y_pred, num_classes=3)
        self.assertEqual(metrics["top1_accuracy"], 66.67)
        self.assertEqual(metrics["macro_f1"], 65.56)


if __name__ == "__main__":
    unittest.main()
