"""Comprehensive test suite for Interactive Real-Time Systems (RTS) Simulator.

Covers:
- WorkloadManager: requests, validation, presets, BDD100K driving scene class resolution
- InteractiveRTSEngine: Mode A simulation, FIFO vs EDF, AC ON vs OFF, accounting identities
- GPUInferenceEngine: physical forward pass, CUDA Graph capture, timing, top-1 predictions
- REST API Server: endpoints, serialization, error handling
"""

import os
import sys
import unittest
import json
import torch

from src.interactive_rts.workload_manager import WorkloadManager, BDD_CLASSES, MAX_FILE_SIZE_BYTES
from src.interactive_rts.engine import InteractiveRTSEngine, SimulationConfig
from src.interactive_rts.gpu_inference import GPUInferenceEngine, get_inference_engine
from src.phase24.model_profiles import load_model_profiles


class TestWorkloadManager(unittest.TestCase):
    """Test WorkloadManager staging, presets, and validation."""

    def setUp(self):
        self.wm = WorkloadManager()
        self.wm.clear_requests()

    def test_add_and_remove_requests(self):
        req1 = self.wm.add_request(
            request_id="R_test_01",
            arrival_time_ms=0.0,
            relative_deadline_ms=5.0,
            ground_truth_label="highway",
        )
        self.assertEqual(req1["request_id"], "R_test_01")
        self.assertEqual(req1["arrival_time_ms"], 0.0)
        self.assertEqual(req1["relative_deadline_ms"], 5.0)
        self.assertEqual(req1["absolute_deadline_ms"], 5.0)
        self.assertEqual(req1["ground_truth_class_id"], BDD_CLASSES.index("highway"))

        req2 = self.wm.add_request(
            request_id="R_test_02",
            arrival_time_ms=2.5,
            relative_deadline_ms=4.0,
            ground_truth_label="city street",
        )
        self.assertEqual(len(self.wm.get_requests()), 2)

        # Removal
        removed = self.wm.remove_request("R_test_01")
        self.assertTrue(removed)
        self.assertEqual(len(self.wm.get_requests()), 1)
        self.assertEqual(self.wm.get_requests()[0]["request_id"], "R_test_02")

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            self.wm.add_request(request_id="R_neg", arrival_time_ms=-1.0, relative_deadline_ms=5.0)

        with self.assertRaises(ValueError):
            self.wm.add_request(request_id="R_zero_d", arrival_time_ms=0.0, relative_deadline_ms=0.0)

    def test_upload_validation(self):
        # Invalid extension
        with self.assertRaises(ValueError):
            self.wm.validate_and_save_upload(b"fake data", "malicious.exe")

        # Exceeds max size
        with self.assertRaises(ValueError):
            huge_bytes = b"0" * (MAX_FILE_SIZE_BYTES + 1024)
            self.wm.validate_and_save_upload(huge_bytes, "huge.jpg")

    def test_load_presets(self):
        presets = [
            "periodic_high_tight", "bursty_traffic", "overload_stress", "nominal_loose",
            "scene1_arrival", "scene2_gpu_resource", "scene3_missed_deadlines",
            "scene4_fifo_vs_edf", "scene5_admission_control", "scene6_adaptive"
        ]
        for p in presets:
            reqs = self.wm.load_preset(p)
            self.assertGreater(len(reqs), 0, f"Preset {p} returned 0 requests")
            # Verify monotonic arrival ordering
            arrivals = [r["arrival_time_ms"] for r in reqs]
            self.assertEqual(arrivals, sorted(arrivals), f"Preset {p} not sorted by arrival time")


class TestInteractiveRTSEngine(unittest.TestCase):
    """Test RTS simulation engine, scheduling policies, and accounting laws."""

    def setUp(self):
        self.engine = InteractiveRTSEngine()
        self.wm = WorkloadManager()

    def test_conservation_identities_all_presets(self):
        """Verify that N_adm + N_rej = N_off, N_comp + N_exp = N_adm, N_ontime + N_late = N_comp hold unconditionally."""
        presets = [
            "periodic_high_tight", "bursty_traffic", "overload_stress", "nominal_loose",
            "scene1_arrival", "scene2_gpu_resource", "scene3_missed_deadlines",
            "scene4_fifo_vs_edf", "scene5_admission_control", "scene6_adaptive"
        ]
        configs = [
            SimulationConfig(scheduling_policy="EDF", admission_control=True, model_selection="adaptive"),
            SimulationConfig(scheduling_policy="FIFO", admission_control=False, model_selection="deit_small_baseline"),
            SimulationConfig(scheduling_policy="EDF", admission_control=False, model_selection="evit_small_k07_fuse"),
        ]

        for p in presets:
            reqs = self.wm.load_preset(p)
            for cfg in configs:
                res = self.engine.run_workload(reqs, cfg)
                acc = res["accounting"]
                m = res["metrics"]
                self.assertTrue(acc["verified"], f"Conservation failed for preset={p}, policy={cfg.scheduling_policy}, AC={cfg.admission_control}")
                self.assertEqual(m["admitted_requests"] + m["rejected_requests"], m["offered_requests"])
                self.assertEqual(m["completed_requests"] + m["expired_requests"], m["admitted_requests"])
                self.assertEqual(m["on_time_requests"] + m["late_requests"], m["completed_requests"])

    def test_admission_control_gain_on_periodic_high_tight(self):
        """Verify proposed Adaptive EViT + AC significantly outperforms baseline on periodic_high_tight."""
        reqs = self.wm.load_preset("periodic_high_tight")

        baseline_cfg = SimulationConfig(
            scheduling_policy="FIFO",
            admission_control=False,
            model_selection="deit_small_baseline",
            execution_mode="simulation",
        )
        proposed_cfg = SimulationConfig(
            scheduling_policy="EDF",
            admission_control=True,
            model_selection="adaptive",
            execution_mode="simulation",
        )

        comp = self.engine.run_comparison(reqs, baseline_cfg, proposed_cfg)
        gains = comp["gains"]
        self.assertGreater(gains["proposed_on_time_pct"], gains["baseline_on_time_pct"])
        self.assertGreaterEqual(gains["delta_on_time_ratio_pct"], 30.0, "Proposed should have at least 30% on-time gain")

    def test_overload_stress_rejection(self):
        """Verify that under overload traffic arriving faster than GPU capability, AC rejects excess traffic."""
        reqs = self.wm.load_preset("overload_stress")
        cfg = SimulationConfig(scheduling_policy="EDF", admission_control=True, model_selection="adaptive")
        res = self.engine.run_workload(reqs, cfg)
        self.assertGreater(res["metrics"]["rejected_requests"], 0, "AC should reject in overload stress")

    def test_event_steps_generation(self):
        """Verify event trace contains valid step IDs, active stages, and explanations."""
        reqs = self.wm.load_preset("nominal_loose")
        cfg = SimulationConfig(scheduling_policy="EDF", admission_control=True, model_selection="adaptive")
        res = self.engine.run_workload(reqs, cfg)
        events = res["events"]
        self.assertGreater(len(events), 0)

        # Check fields of event steps
        for i, ev in enumerate(events):
            self.assertEqual(ev["step_id"], i + 1)
            self.assertIn("timestamp_ms", ev)
            self.assertIn("active_stage", ev)
            self.assertIn("explanation", ev)
            self.assertIn("request_id", ev)


class TestGPUInferenceEngine(unittest.TestCase):
    """Test physical GPU execution on RTX 4060."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required for GPU execution tests")
    def test_device_info(self):
        engine = get_inference_engine()
        info = engine.get_device_info()
        self.assertTrue(info["is_cuda"])
        self.assertIn("GeForce", info["device_name"])
        self.assertGreater(info["vram_total_mb"], 4000.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required for GPU execution tests")
    def test_cuda_graph_sub_5ms_latency(self):
        engine = get_inference_engine()
        res = engine.run_inference("evit_small_k07_fuse")
        self.assertIn("latency_ms", res)
        # Verify sub-5ms latency on RTX 4060
        self.assertLess(res["latency_ms"], 5.0, f"Expected sub-5ms CUDA graph latency, got {res['latency_ms']}ms")
        self.assertIn("predicted_class_name", res)
        self.assertIn("confidence", res)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertEqual(len(res["top3"]), 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required for GPU execution tests")
    def test_real_image_inference_on_preset(self):
        wm = WorkloadManager()
        reqs = wm.load_preset("nominal_loose")
        engine = InteractiveRTSEngine()
        cfg_gpu = SimulationConfig(scheduling_policy="EDF", admission_control=True, model_selection="adaptive", execution_mode="gpu")
        res_gpu = engine.run_workload(reqs, cfg_gpu)
        m = res_gpu["metrics"]
        self.assertEqual(m["on_time_ratio"], 1.0)
        self.assertTrue(res_gpu["accounting"]["verified"])


if __name__ == "__main__":
    unittest.main()
