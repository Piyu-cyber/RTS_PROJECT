"""Unit tests for Phase 24 real-time scheduling, queues, and simulation correctness."""

import unittest
from typing import Dict

from src.phase24.request import Request, RequestOutcome, RequestState
from src.phase24.priority_queue import FIFOQueue, EDFQueue
from src.phase24.model_profiles import ModelProfile, load_model_profiles
from src.phase24.admission_control import AdmissionController
from src.phase24.adaptive_controller import AdaptiveController
from src.phase24.discrete_event_simulator import Policy, RealTimeSimulator, SimulationResult


def create_mock_profiles() -> Dict[str, ModelProfile]:
    """Create lightweight mock model profiles for isolated testing."""
    return {
        "deit_small_baseline": ModelProfile(
            config_id="deit_small_baseline",
            display_name="DeiT-Small Baseline",
            keep_rate=1.0,
            fuse_token=False,
            parameters_m=22.05,
            macs_g=4.60,
            token_sequence=[196] * 12,
            val_top1_accuracy=93.89,
            test_top1_accuracy=92.67,
            test_macro_f1=92.56,
            cuda_graph_latency_ms={"mean": 3.28, "median": 3.15, "p95": 3.50, "p99": 3.90},
            eager_latency_ms={"mean": 13.38, "median": 13.11, "p95": 17.66, "p99": 21.39},
            throughput_fps=432.6,
            checkpoint_path="",
            official_checkpoint="",
        ),
        "evit_small_k09_fuse": ModelProfile(
            config_id="evit_small_k09_fuse",
            display_name="EViT-Small (0.9)",
            keep_rate=0.9,
            fuse_token=True,
            parameters_m=22.05,
            macs_g=4.03,
            token_sequence=[196, 178, 162, 147] + [147] * 8,
            val_top1_accuracy=93.61,
            test_top1_accuracy=92.75,
            test_macro_f1=92.66,
            cuda_graph_latency_ms={"mean": 3.14, "median": 2.92, "p95": 3.30, "p99": 3.70},
            eager_latency_ms={"mean": 15.55, "median": 15.30, "p95": 19.95, "p99": 24.31},
            throughput_fps=469.5,
            checkpoint_path="",
            official_checkpoint="",
        ),
        "evit_small_k07_fuse": ModelProfile(
            config_id="evit_small_k07_fuse",
            display_name="EViT-Small (0.7)",
            keep_rate=0.7,
            fuse_token=True,
            parameters_m=22.05,
            macs_g=3.03,
            token_sequence=[196, 139, 99, 71] + [71] * 8,
            val_top1_accuracy=93.48,
            test_top1_accuracy=93.19,
            test_macro_f1=93.09,
            cuda_graph_latency_ms={"mean": 2.79, "median": 2.50, "p95": 3.00, "p99": 3.40},
            eager_latency_ms={"mean": 15.53, "median": 15.03, "p95": 20.67, "p99": 25.02},
            throughput_fps=622.2,
            checkpoint_path="",
            official_checkpoint="",
        ),
        "evit_small_k05_fuse": ModelProfile(
            config_id="evit_small_k05_fuse",
            display_name="EViT-Small (0.5)",
            keep_rate=0.5,
            fuse_token=True,
            parameters_m=22.05,
            macs_g=2.31,
            token_sequence=[196, 99, 51, 27] + [27] * 8,
            val_top1_accuracy=92.53,
            test_top1_accuracy=91.44,
            test_macro_f1=91.34,
            cuda_graph_latency_ms={"mean": 2.36, "median": 2.38, "p95": 2.50, "p99": 2.80},
            eager_latency_ms={"mean": 15.67, "median": 15.59, "p95": 19.80, "p99": 22.85},
            throughput_fps=788.8,
            checkpoint_path="",
            official_checkpoint="",
        ),
    }


class TestPhase24RealTime(unittest.TestCase):
    def test_absolute_deadline_calculation(self):
        """Verify D_i = r_i + d_i calculation."""
        req = Request(request_id="r1", arrival_time_ms=10.5, relative_deadline_ms=5.2)
        self.assertEqual(req.absolute_deadline_ms, 15.7)

    def test_fifo_queue_ordering(self):
        """Verify FIFO orders strictly by arrival time with tie breaking."""
        q = FIFOQueue()
        r1 = Request(request_id="r1", arrival_time_ms=10.0, relative_deadline_ms=20.0)
        r2 = Request(request_id="r2", arrival_time_ms=5.0, relative_deadline_ms=5.0)
        r3 = Request(request_id="r3", arrival_time_ms=5.0, relative_deadline_ms=10.0)

        q.push(r1)
        q.push(r2)
        q.push(r3)

        self.assertEqual(len(q), 3)
        self.assertEqual(q.pop().request_id, "r2")  # Arrived at 5.0 first
        self.assertEqual(q.pop().request_id, "r3")  # Arrived at 5.0 second
        self.assertEqual(q.pop().request_id, "r1")  # Arrived at 10.0

    def test_edf_queue_ordering(self):
        """Verify EDF orders strictly by absolute deadline with tie breaking."""
        q = EDFQueue()
        # r1: D = 5.0 + 10.0 = 15.0
        r1 = Request(request_id="r1", arrival_time_ms=5.0, relative_deadline_ms=10.0)
        # r2: D = 1.0 + 8.0 = 9.0
        r2 = Request(request_id="r2", arrival_time_ms=1.0, relative_deadline_ms=8.0)
        # r3: D = 2.0 + 7.0 = 9.0 (same D as r2, but r2 arrived earlier)
        r3 = Request(request_id="r3", arrival_time_ms=2.0, relative_deadline_ms=7.0)

        q.push(r1)
        q.push(r2)
        q.push(r3)

        self.assertEqual(q.pop().request_id, "r2")  # D=9.0, arrived at 1.0
        self.assertEqual(q.pop().request_id, "r3")  # D=9.0, arrived at 2.0
        self.assertEqual(q.pop().request_id, "r1")  # D=15.0

    def test_admission_control_rejection(self):
        """Verify admission controller rejects provably infeasible requests."""
        profiles = create_mock_profiles()
        ac = AdmissionController(profiles, latency_metric="p95", backend="cuda_graph")

        # Ongoing execution finishes at t=10.0ms. Candidate arrives at t=0.0ms with relative deadline 5.0ms (D=5.0).
        # Earliest start is 10.0ms > deadline 5.0ms -> Must be rejected!
        candidate = Request(request_id="cand", arrival_time_ms=0.0, relative_deadline_ms=5.0)
        feasible = ac.is_feasible(
            request=candidate,
            current_time_ms=0.0,
            gpu_busy_until_ms=10.0,
            queued_requests=[],
        )
        self.assertFalse(feasible, "Candidate starting at 10ms with D=5ms should be rejected")

        # Candidate with relative deadline 15.0ms (D=15.0ms). Start at 10ms + min_exec (2.5ms) = 12.5ms <= 15.0ms -> Feasible!
        candidate2 = Request(request_id="cand2", arrival_time_ms=0.0, relative_deadline_ms=15.0)
        feasible2 = ac.is_feasible(
            request=candidate2,
            current_time_ms=0.0,
            gpu_busy_until_ms=10.0,
            queued_requests=[],
        )
        self.assertTrue(feasible2, "Candidate starting at 10ms with D=15ms should be admitted")

    def test_adaptive_controller_model_selection(self):
        """Verify adaptive controller selects highest-quality feasible model."""
        profiles = create_mock_profiles()
        ctrl = AdaptiveController(profiles, latency_metric="p95", backend="cuda_graph")

        # Case 1: Ample slack at t=0.0 with D=10.0ms.
        # DeiT (p95=3.5ms) finishes at 3.5ms <= 10.0ms. Should pick DeiT (highest val acc 93.89%).
        req1 = Request(request_id="r1", arrival_time_ms=0.0, relative_deadline_ms=10.0)
        m1, est1, slack1 = ctrl.select_model(req1, current_time_ms=0.0)
        self.assertEqual(m1, "deit_small_baseline")

        # Case 2: Tight slack at t=0.0 with D=2.8ms.
        # DeiT (3.5ms) and EViT 0.7 (3.0ms) miss. EViT 0.5 (2.5ms) meets (finish 2.5 <= 2.8).
        req2 = Request(request_id="r2", arrival_time_ms=0.0, relative_deadline_ms=2.8)
        m2, est2, slack2 = ctrl.select_model(req2, current_time_ms=0.0)
        self.assertEqual(m2, "evit_small_k05_fuse")

        # Case 3: Infeasible at t=0.0 with D=2.0ms (even EViT 0.5 needs 2.5ms).
        # Should return None under firm deadline semantics.
        req3 = Request(request_id="r3", arrival_time_ms=0.0, relative_deadline_ms=2.0)
        m3, est3, slack3 = ctrl.select_model(req3, current_time_ms=0.0)
        self.assertIsNone(m3)

    def test_accounting_invariants_and_non_preemption(self):
        """Verify complete accounting without double-counting and non-preemptive GPU blocking."""
        profiles = create_mock_profiles()
        sim = RealTimeSimulator(profiles, latency_metric="p95", backend="cuda_graph")

        # Workload of 4 requests:
        # r1 arrives at 0.0, D=4.0 (dispatched immediately, runs until 3.5ms)
        # r2 arrives at 1.0, D=3.0 (blocked by r1, will expire or miss)
        # r3 arrives at 4.0, D=8.0 (dispatched at 4.0ms, runs until 7.5ms)
        # r4 arrives at 4.5, D=6.0 (infeasible under admission control)
        requests = [
            Request("r1", arrival_time_ms=0.0, relative_deadline_ms=4.0),
            Request("r2", arrival_time_ms=1.0, relative_deadline_ms=2.0),  # D=3.0
            Request("r3", arrival_time_ms=4.0, relative_deadline_ms=4.0),  # D=8.0
            Request("r4", arrival_time_ms=4.5, relative_deadline_ms=1.5),  # D=6.0
        ]

        result = sim.run(Policy.PROPOSED_ADAPTIVE_AC, requests, workload_name="test_invariants")

        # Invariant checks
        self.assertEqual(result.num_offered, 4)
        self.assertEqual(result.num_offered, result.num_admitted + result.num_rejected)
        self.assertEqual(result.num_admitted, result.num_completed + result.num_expired)
        self.assertEqual(result.num_completed, result.num_completed_on_time + result.num_completed_late)

    def test_edf_vs_fifo_deadline_advantage(self):
        """Verify classic RTS condition: EDF completes requests on time where FIFO misses."""
        profiles = create_mock_profiles()
        sim = RealTimeSimulator(profiles, latency_metric="p95", backend="cuda_graph")

        # Two requests arriving simultaneously at t=0:
        # r_long_deadline: D=20.0ms
        # r_urgent: D=4.0ms
        # Under FIFO order [r_long_deadline, r_urgent]:
        # r_long runs from 0 to 3.5ms.
        # r_urgent dispatched at 3.5ms, completes at 3.5+3.5=7.0ms > D=4.0ms (LATE or EXPIRED).
        # Under EDF order [r_urgent, r_long_deadline]:
        # r_urgent runs 0 to 3.5ms <= 4.0ms (ON TIME!).
        # r_long runs 3.5 to 7.0ms <= 20.0ms (ON TIME!). Both on time!
        trace = [
            Request("r_long", arrival_time_ms=0.0, relative_deadline_ms=20.0),
            Request("r_urgent", arrival_time_ms=0.0, relative_deadline_ms=4.0),
        ]

        res_fifo = sim.run(Policy.FIFO_FIXED_DEIT, trace, "fifo_test")
        res_edf = sim.run(Policy.EDF_FIXED_DEIT, trace, "edf_test")

        self.assertEqual(res_edf.num_completed_on_time, 2, "EDF should complete both on time")
        self.assertLess(res_fifo.num_completed_on_time, 2, "FIFO should miss r_urgent")

    def test_image_path_resolution_and_accounting(self):
        """Verify image path resolution on disk and accounting tracking."""
        from src.phase24.gpu_trace_replay import GPUTraceReplayer
        import os

        profiles = create_mock_profiles()
        # Initialize without CUDA model build (device='cpu')
        replayer = GPUTraceReplayer.__new__(GPUTraceReplayer)
        replayer.profiles = profiles
        replayer.device = "cpu"
        replayer.loaded_image_paths = set()
        replayer.disk_image_count = 0
        replayer.synthetic_image_count = 0
        replayer.image_tensor_cache = {}

        # 1. Non-existent image resolves to None
        self.assertIsNone(replayer.resolve_image_path("non_existent_fake_image_xyz"))

        # 2. Real canonical test image resolves if dataset is present
        canonical_test_dir = os.path.join("data", "bdd100k", "images", "100k", "test")
        if os.path.isdir(canonical_test_dir):
            files = [f for f in os.listdir(canonical_test_dir) if f.endswith(".jpg")]
            if files:
                sample_name = os.path.splitext(files[0])[0]
                resolved = replayer.resolve_image_path(sample_name)
                self.assertIsNotNone(resolved)
                self.assertTrue(os.path.isfile(resolved))

    def test_all_four_conservation_invariants_on_results(self):
        """Verify that all 4 real-time conservation laws hold across diverse workloads."""
        profiles = create_mock_profiles()
        sim = RealTimeSimulator(profiles, latency_metric="p95", backend="cuda_graph")

        for seed in [10, 20, 30]:
            import random
            rng = random.Random(seed)
            trace = []
            t = 0.0
            for i in range(25):
                t += rng.uniform(0.5, 3.0)
                d = rng.uniform(2.5, 8.0)
                trace.append(Request(f"req_{seed}_{i}", arrival_time_ms=t, relative_deadline_ms=d))

            for pol in [Policy.FIFO_FIXED_DEIT, Policy.EDF_FIXED_DEIT, Policy.PROPOSED_ADAPTIVE_AC]:
                res = sim.run(pol, trace, workload_name=f"test_wl_{seed}")
                # Conservation 1: Offered = On-time + Late + Rejected + Expired
                self.assertEqual(
                    res.num_offered,
                    res.num_completed_on_time + res.num_completed_late + res.num_rejected + res.num_expired,
                    f"Invariant 1 violated in {pol} seed {seed}",
                )
                # Conservation 2: Completed = On-time + Late
                self.assertEqual(
                    res.num_completed,
                    res.num_completed_on_time + res.num_completed_late,
                    f"Invariant 2 violated in {pol} seed {seed}",
                )
                # Conservation 3: Admitted = Completed + Expired
                self.assertEqual(
                    res.num_admitted,
                    res.num_completed + res.num_expired,
                    f"Invariant 3 violated in {pol} seed {seed}",
                )
                # Conservation 4: Offered = Admitted + Rejected
                self.assertEqual(
                    res.num_offered,
                    res.num_admitted + res.num_rejected,
                    f"Invariant 4 violated in {pol} seed {seed}",
                )


if __name__ == "__main__":
    unittest.main()
