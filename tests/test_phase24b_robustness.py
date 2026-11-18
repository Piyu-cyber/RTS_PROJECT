"""Unit tests for Phase 24B experiment engine, robustness suite, and conservation laws."""

import unittest
import numpy as np

from src.phase24.model_profiles import load_model_profiles
from src.phase24.request import Request, RequestOutcome
from src.phase24.discrete_event_simulator import Policy
from src.phase24b.experiment_engine import CanonicalSimulator, CanonicalResult
from src.phase24b.robustness_suite import get_canonical_suite, get_robustness_suite, WorkloadConfig, generate_trace_exact


class TestPhase24BRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = load_model_profiles("outputs/phase24/model_profiles.json")
        cls.sim = CanonicalSimulator(profiles=cls.profiles, latency_metric="p95", backend="cuda_graph")

    def test_conservation_identities_enforcement(self):
        """Test that CanonicalResult enforces exact conservation laws."""
        # Valid result should pass
        res = CanonicalResult(
            policy="TEST",
            workload_name="wl",
            workload_category="test",
            load_level="mod",
            deadline_tightness="mod",
            seed=42,
            experiment_type="SIM",
            num_offered=100,
            num_admitted=75,
            num_rejected=25,
            num_expired=10,
            num_completed=65,
            num_completed_on_time=60,
            num_completed_late=5,
            on_time_ratio_offered=0.60,
            on_time_ratio_accepted=0.80,
            deadline_miss_ratio_completed=0.0769,
            rejection_ratio=0.25,
            expiration_ratio=0.10,
            mean_response_time_ms=4.0,
            median_response_time_ms=4.0,
            p95_response_time_ms=4.5,
            p99_response_time_ms=4.8,
            mean_waiting_time_ms=1.0,
            median_waiting_time_ms=1.0,
            mean_queue_length=1.2,
            max_queue_length=4,
            total_gpu_busy_time_ms=200.0,
            total_makespan_ms=250.0,
            gpu_utilization_pct=80.0,
            mean_val_accuracy=93.5,
            synthetic_expected_test_accuracy=92.5,
            empirical_test_accuracy=None,
            count_deit_small=50,
            count_evit_k09=10,
            count_evit_k07=5,
            count_evit_k05=0,
        )
        res.verify_conservation()

        # Violation 1: admitted + rejected != offered
        res.num_offered = 101
        with self.assertRaises(ValueError):
            res.verify_conservation()
        res.num_offered = 100

        # Violation 2: completed + expired != admitted
        res.num_completed = 66
        with self.assertRaises(ValueError):
            res.verify_conservation()
        res.num_completed = 65

        # Violation 3: on_time + late != completed
        res.num_completed_on_time = 61
        with self.assertRaises(ValueError):
            res.verify_conservation()

    def test_multi_seed_generation_determinism(self):
        """Test that same config and seed produces bitwise identical request sequences."""
        cfg = WorkloadConfig("test_trace", "bursty", "high", "tight", seed=999)
        trace_a = generate_trace_exact(cfg)
        trace_b = generate_trace_exact(cfg)

        self.assertEqual(len(trace_a), len(trace_b))
        for ra, rb in zip(trace_a, trace_b):
            self.assertEqual(ra.request_id, rb.request_id)
            self.assertEqual(ra.arrival_time_ms, rb.arrival_time_ms)
            self.assertEqual(ra.relative_deadline_ms, rb.relative_deadline_ms)

    def test_sensitivity_metric_switch(self):
        """Verify that simulator changes latency profile between p95 and median."""
        sim_p95 = CanonicalSimulator(profiles=self.profiles, latency_metric="p95", backend="cuda_graph")
        sim_med = CanonicalSimulator(profiles=self.profiles, latency_metric="median", backend="cuda_graph")

        cfg = WorkloadConfig("burst_test", "bursty", "high", "tight", seed=48)
        trace_1 = generate_trace_exact(cfg)
        trace_2 = generate_trace_exact(cfg)

        res_p95 = sim_p95.run(Policy.FIFO_FIXED_DEIT, trace_1, "burst_test")
        res_med = sim_med.run(Policy.FIFO_FIXED_DEIT, trace_2, "burst_test")

        # Median latency (3.15ms) clears queue faster than p95 (3.78ms)
        self.assertGreater(res_med.on_time_ratio_offered, res_p95.on_time_ratio_offered)
        self.assertLess(res_med.total_gpu_busy_time_ms, res_p95.total_gpu_busy_time_ms)

    def test_accuracy_metrics_separation(self):
        """Ensure validation accuracy and synthetic test accuracy are tracked distinctly."""
        cfg = WorkloadConfig("acc_test", "periodic", "high", "tight", seed=44)
        trace = generate_trace_exact(cfg)
        res = self.sim.run(Policy.PROPOSED_ADAPTIVE_AC, trace, "acc_test")

        # Validation accuracy should be distinct and strictly >= threshold
        self.assertGreaterEqual(res.mean_val_accuracy, 92.0)
        self.assertGreater(res.synthetic_expected_test_accuracy, 90.0)
        self.assertIsNone(res.empirical_test_accuracy)  # Must be None in simulation


if __name__ == "__main__":
    unittest.main()
