"""Automated test suite for Phase 9 runtime deadline-aware inference prototype."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from realtime.controller import load_benchmark_modes, select_mode
from realtime.runtime_inference import (
    DEFAULT_BENCHMARK,
    MODERATE_CHECKPOINT,
    AGGRESSIVE_CHECKPOINT,
    PHASE4_CHECKPOINT,
    RuntimeModelRegistry,
    execute_model_forward,
    preprocess_image,
    process_runtime_request,
)
from realtime.simulate_strategies import generate_deadline_stream


class TestCheckpointAndModelIntegrity(unittest.TestCase):
    """Verify that all required checkpoints exist and have the expected structure."""

    def test_checkpoints_exist(self) -> None:
        self.assertTrue(PHASE4_CHECKPOINT.is_file(), f"Missing: {PHASE4_CHECKPOINT}")
        self.assertTrue(MODERATE_CHECKPOINT.is_file(), f"Missing: {MODERATE_CHECKPOINT}")
        self.assertTrue(AGGRESSIVE_CHECKPOINT.is_file(), f"Missing: {AGGRESSIVE_CHECKPOINT}")

    def test_checkpoint_payload_structures(self) -> None:
        c4 = torch.load(PHASE4_CHECKPOINT, map_location="cpu", weights_only=False)
        self.assertIn("head_state_dict", c4)
        self.assertIn("run_config", c4)
        self.assertIn("split_manifest", c4)

        c_mod = torch.load(MODERATE_CHECKPOINT, map_location="cpu", weights_only=False)
        self.assertIn("backbone_state_dict", c_mod)
        self.assertIn("head_state_dict", c_mod)
        self.assertEqual(c_mod.get("mode"), "moderate")
        self.assertEqual(c_mod.get("blocks"), [2, 5, 8, 11])

        c_agg = torch.load(AGGRESSIVE_CHECKPOINT, map_location="cpu", weights_only=False)
        self.assertIn("backbone_state_dict", c_agg)
        self.assertIn("head_state_dict", c_agg)
        self.assertEqual(c_agg.get("mode"), "aggressive")
        self.assertEqual(c_agg.get("blocks"), [1, 2, 3, 4, 5, 6, 7, 8])


class TestPreprocessingAndTiming(unittest.TestCase):
    """Verify image preprocessing, tensor dimensions, and timing units."""

    def test_preprocess_synthetic_image(self) -> None:
        img = Image.new("RGB", (300, 300), color=(128, 64, 32))
        tensor = preprocess_image(img, target_size=224)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all())

    def test_preprocess_tensor_input(self) -> None:
        raw_t = torch.randn(3, 224, 224)
        tensor = preprocess_image(raw_t)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))


class TestRuntimeExecutionOnGPU(unittest.TestCase):
    """CUDA integration tests validating actual model execution and timing."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available; skipping GPU runtime tests.")
        cls.device = torch.device("cuda:0")
        cls.registry = RuntimeModelRegistry(cls.device)
        cls.registry.preload_all()
        cls.registry.warmup(num_iterations=2)
        cls.modes = load_benchmark_modes(DEFAULT_BENCHMARK)

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_models_produce_valid_segmentation_masks(self) -> None:
        dummy = torch.randn(1, 3, 224, 224, device=self.device)
        for mode_name, (bb, hd) in self.registry.models.items():
            preds_cpu, host_ms, cuda_ms = execute_model_forward(bb, hd, dummy)
            self.assertEqual(preds_cpu.shape, (224, 224))
            self.assertTrue(torch.isin(preds_cpu, torch.tensor([0, 1])).all())
            self.assertGreater(host_ms, 0.0)
            self.assertGreater(cuda_ms, 0.0)

    def test_runtime_request_flow_and_timing_isolation(self) -> None:
        img = Image.new("RGB", (224, 224), color=(50, 150, 200))
        save_mask = self.temp_dir / "test_mask.png"

        # Feasible request: 100 ms deadline
        res = process_runtime_request(
            request_id="test_001",
            strategy="adaptive_controller",
            deadline_ms=100.0,
            workload_category="relaxed",
            image_source=img,
            registry=self.registry,
            modes_metadata=self.modes,
            latency_metric="p95",
            save_prediction_path=save_mask,
        )

        self.assertEqual(res.request_id, "test_001")
        self.assertEqual(res.selected_mode, "full_attention")
        self.assertTrue(res.predicted_empirical_feasibility)
        self.assertFalse(res.likely_deadline_miss)
        self.assertTrue(res.actual_deadline_hit)
        self.assertTrue(save_mask.is_file())

        # Check all timing metrics are positive finite values in milliseconds
        self.assertGreater(res.controller_latency_ms, 0.0)
        self.assertGreater(res.preprocessing_latency_ms, 0.0)
        self.assertGreater(res.transfer_latency_ms, 0.0)
        self.assertGreater(res.model_execution_ms, 0.0)
        self.assertGreater(res.total_request_latency_ms, res.model_execution_ms)

        # Infeasible request: 1.0 ms deadline
        res_tight = process_runtime_request(
            request_id="test_002",
            strategy="adaptive_controller",
            deadline_ms=1.0,
            workload_category="tight",
            image_source=img,
            registry=self.registry,
            modes_metadata=self.modes,
            latency_metric="p95",
        )
        self.assertFalse(res_tight.predicted_empirical_feasibility)
        self.assertTrue(res_tight.likely_deadline_miss)
        self.assertFalse(res_tight.actual_deadline_hit)


if __name__ == "__main__":
    unittest.main()
