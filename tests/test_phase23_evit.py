"""Unit tests for Phase 23A EViT feasibility and baseline reproduction."""

import os
import json
import csv
import unittest
import torch
import torch.nn as nn

from src.phase23_evit.models import CONFIGS, build_evit_model
from src.phase23_evit.checkpoint_manager import (
    get_checkpoint_path,
    verify_checkpoint_exists,
    load_checkpoint_into_model,
    get_loaded_model,
    DEFAULT_WEIGHTS_DIR,
)
from src.phase23_evit.benchmark import inspect_token_counts, verify_real_image_pipeline


class TestPhase23AEViT(unittest.TestCase):
    def test_model_construction(self):
        """Test model construction for all 4 required configurations."""
        for cfg_name in ["deit_small_baseline", "evit_small_k09_fuse", "evit_small_k07_fuse", "evit_small_k05_fuse"]:
            model = build_evit_model(cfg_name, pretrained=False)
            self.assertIsInstance(model, nn.Module)
            param_count = sum(p.numel() for p in model.parameters())
            self.assertGreater(param_count, 20_000_000, f"Expected >20M params, got {param_count}")

    def test_expected_input_output_shapes(self):
        """Test expected input (B, 3, 224, 224) and output (B, 1000) shapes."""
        model = build_evit_model("evit_small_k07_fuse", pretrained=False)
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy_input)
        self.assertEqual(out.shape, (2, 1000), f"Expected shape (2, 1000), got {out.shape}")

        # Test token counts
        with torch.no_grad():
            _, left_tokens, _ = model.forward_features(dummy_input)
        self.assertEqual(len(left_tokens), 12)
        self.assertEqual(left_tokens[0], 196)
        self.assertEqual(left_tokens[-1], 71)  # 0.7 keep rate after 3 shrink layers with fusion

    def test_correct_model_to_checkpoint_mapping(self):
        """Verify correct model-to-checkpoint mapping."""
        expected_mapping = {
            "deit_small_baseline": "deit_small_patch16_224.pth",
            "evit_small_k09_fuse": "evit_small_patch16_0.9_fuse.pth",
            "evit_small_k07_fuse": "evit_small_patch16_0.7_fuse.pth",
            "evit_small_k05_fuse": "evit_small_patch16_0.5_fuse.pth",
        }
        for cfg_name, expected_file in expected_mapping.items():
            self.assertEqual(CONFIGS[cfg_name]["checkpoint_filename"], expected_file)
            path = get_checkpoint_path(cfg_name)
            self.assertEqual(os.path.basename(path), expected_file)

        with self.assertRaises(ValueError):
            _ = get_checkpoint_path("invalid_model_name_xyz")

    def test_checkpoint_loading_and_keys(self):
        """Verify checkpoint loading and that keys match successfully."""
        for cfg_name in ["deit_small_baseline", "evit_small_k09_fuse", "evit_small_k07_fuse", "evit_small_k05_fuse"]:
            self.assertTrue(verify_checkpoint_exists(cfg_name), f"Checkpoint missing for {cfg_name}")
            model, info = get_loaded_model(cfg_name, device="cpu")
            self.assertTrue(info["is_perfect_match"], f"Key mismatch in {cfg_name}: missing={info['missing_keys']}, unexpected={info['unexpected_keys']}")
            self.assertEqual(len(info["missing_keys"]), 0)
            self.assertEqual(len(info["unexpected_keys"]), 0)

    def test_latency_profile_serialization(self):
        """Verify latency profiles and model comparison files serialize and deserialize properly."""
        json_path = os.path.join("outputs", "phase23_evit_migration", "latency_profiles.json")
        csv_path = os.path.join("outputs", "phase23_evit_migration", "model_comparison.csv")

        self.assertTrue(os.path.isfile(json_path), f"JSON file missing at {json_path}")
        self.assertTrue(os.path.isfile(csv_path), f"CSV file missing at {csv_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)

        # Check metadata fields
        meta = profiles["metadata"]
        self.assertIn("device", meta)
        self.assertIn("precision", meta)
        self.assertIn("batch_size_latency", meta)
        self.assertIn("warmup_iters_latency", meta)
        self.assertIn("eval_iters_latency", meta)
        self.assertIn("measurement_units", meta)

        # Check models
        models = profiles["models"]
        for cfg_name in ["deit_small_baseline", "evit_small_k09_fuse", "evit_small_k07_fuse", "evit_small_k05_fuse"]:
            self.assertIn(cfg_name, models)
            m = models[cfg_name]
            self.assertIn("latency_ms", m)
            self.assertIn("batch_latency_ms", m["latency_ms"])
            self.assertIn("mean", m["latency_ms"]["batch_latency_ms"])
            self.assertIn("median", m["latency_ms"]["batch_latency_ms"])
            self.assertIn("p95", m["latency_ms"]["batch_latency_ms"])
            self.assertIn("p99", m["latency_ms"]["batch_latency_ms"])

        # Check CSV
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 4)
        for row in reader:
            self.assertEqual(row["experiment_status"], "SUCCESS_MEASURED")
            self.assertGreater(float(row["latency_mean_ms"]), 0)
            self.assertGreater(float(row["throughput_fps"]), 0)

    def test_handling_missing_checkpoints_or_datasets(self):
        """Test graceful handling of missing checkpoints or datasets."""
        with self.assertRaises(FileNotFoundError):
            _ = get_loaded_model("deit_small_baseline", weights_dir="non_existent_dir_123")

        model = build_evit_model("deit_small_baseline", pretrained=False)
        res = verify_real_image_pipeline(model, image_dir="non_existent_images_folder_456")
        self.assertEqual(res["status"], "dataset_missing")
        self.assertEqual(res["tested_samples"], 0)

    def test_detection_of_incomplete_experiments(self):
        """Test detection logic for incomplete experiment rows."""
        incomplete_row = {
            "model_id": "test_incomplete",
            "latency_mean_ms": "N/A",
            "experiment_status": "FAILED_MISSING_CHECKPOINT",
        }
        complete_row = {
            "model_id": "test_complete",
            "latency_mean_ms": "15.42",
            "experiment_status": "SUCCESS_MEASURED",
        }

        def is_experiment_complete(row: dict) -> bool:
            if row.get("experiment_status") != "SUCCESS_MEASURED":
                return False
            if row.get("latency_mean_ms") in ["N/A", "", None]:
                return False
            return True

        self.assertFalse(is_experiment_complete(incomplete_row))
        self.assertTrue(is_experiment_complete(complete_row))


if __name__ == "__main__":
    unittest.main()
