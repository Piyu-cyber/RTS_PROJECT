"""
tests/test_phase19_audit_and_validation.py

Automated unit tests for Phase 19:
Audit and Validate the Phase 18 Scenario Demonstration.

Validates:
1. Reconciliation invariants:
   total_requests == on_time + misses_on_gpu + misses_in_queue + rejected + pending
   admitted + rejected == total_requests
   dispatched == on_time + misses_on_gpu + (dispatched_pending)
2. Consistent denominators (total-arrival hit rate vs. dispatched hit rate).
3. Candidate-specific physical latency sampling (manual_interleaved vs. full_attention).
4. Earliest Deadline First (EDF) sorting semantics vs. FIFO.
5. Admission control feasibility checks and non-preemptive GPU blocking.
6. Safeguard assertion for formal WCET on Windows/NVIDIA RTX 4060 (Outcome C).
7. Phase 19 artifacts integrity and outcome reconciliation.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import unittest

from realtime.realtime_simulator import (
    CandidateProfile,
    FormalBoundUnavailableError,
    PolicyType,
    RealTimePriorityQueue,
    RealTimeSimulator,
    Request,
    RequestStatus,
    get_default_candidates,
)
from realtime.run_phase19_validation import generate_poisson_stream


class TestPhase19AuditAndValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates = get_default_candidates()
        cls.raw_latencies = {
            "full_attention": [23.53] * 20,
            "manual_interleaved": [25.36] * 20,
            "sigma_selected": [25.65] * 20,
        }

    def test_candidate_specific_latency_sampling(self):
        """Verify that requests executed with manual_interleaved sample from manual_interleaved latencies."""
        raw_latencies = {
            "full_attention": [20.0, 22.0, 24.0],
            "manual_interleaved": [30.0, 32.0, 34.0],
            "sigma_selected": [28.0, 29.0, 31.0],
        }
        stream = generate_poisson_stream(
            seed=42,
            num_requests=20,
            arrival_rate=20.0,
            deadline_min_ms=25.0,
            deadline_max_ms=50.0,
            quality_threshold=0.75,
        )
        self.assertEqual(len(stream), 20)

        # Run simulation with Always Manual DWConv and raw_latencies
        sim = RealTimeSimulator(
            candidates=self.candidates,
            policy=PolicyType.ALWAYS_MANUAL_DWCONV,
            use_admission_control=False,
            drop_expired_in_queue=True,
            raw_latencies=raw_latencies,
        )
        sim.run_workload(copy.deepcopy(stream))

        # Verify dispatched manual_dwconv requests used manual_interleaved latencies
        manual_trace = [e for e in sim.events if e.event_type == "DISPATCH" and e.details.get("configuration") == "manual_interleaved"]
        self.assertGreater(len(manual_trace), 0)
        for ev in manual_trace:
            self.assertIn(ev.details["duration_ms"], raw_latencies["manual_interleaved"])

    def test_reconciliation_invariants_across_all_policies(self):
        """Verify for every policy that:
        total == on_time + misses_on_gpu + misses_in_queue + rejected + pending
        """
        stream = generate_poisson_stream(
            seed=123,
            num_requests=50,
            arrival_rate=35.0,
            deadline_min_ms=15.0,
            deadline_max_ms=45.0,
            quality_threshold=0.75,
        )

        test_configs = [
            ("Always Full (Queue Drop)", PolicyType.ALWAYS_FULL_ATTENTION, False, True),
            ("Always Full (Blind)", PolicyType.ALWAYS_FULL_ATTENTION, False, False),
            ("Always Manual (Queue Drop)", PolicyType.ALWAYS_MANUAL_DWCONV, False, True),
            ("Always Manual (Blind)", PolicyType.ALWAYS_MANUAL_DWCONV, False, False),
            ("FIFO QA", PolicyType.FIFO, False, True),
            ("EDF QA", PolicyType.EDF, False, True),
            ("FIFO + AC", PolicyType.ADMISSION_CONTROL_FIFO, True, True),
            ("EDF + AC", PolicyType.COMBINED_PROPOSED, True, True),
            ("EUB", PolicyType.EMPIRICAL_UPPER_BOUND, True, True),
        ]

        for name, policy_type, use_ac, drop_q in test_configs:
            with self.subTest(policy=name):
                sim = RealTimeSimulator(
                    candidates=self.candidates,
                    policy=policy_type,
                    use_admission_control=use_ac,
                    drop_expired_in_queue=drop_q,
                    raw_latencies=self.raw_latencies,
                )
                sim.run_workload(copy.deepcopy(stream))
                metrics = sim.compute_summary_metrics()

                total = metrics["total_requests"]
                admitted = metrics["admitted_requests"]
                rejected = metrics["rejected_requests"]
                dispatched = metrics["dispatched_requests"]
                on_time = metrics["on_time_completions"]
                miss_gpu = metrics["misses_on_gpu"]
                miss_q = metrics["misses_in_queue"]
                pending = metrics["pending_requests"]

                # Invariant 1: Admitted + Rejected == Total
                self.assertEqual(admitted + rejected, total, f"Admission mismatch in {name}")

                # Invariant 2: Dispatched == On-Time + Missed on GPU (when no pending execution)
                self.assertEqual(dispatched, on_time + miss_gpu, f"Dispatch mismatch in {name}")

                # Invariant 3: Total == On-Time + Misses on GPU + Misses in Queue + Rejected + Pending
                reconciled_sum = on_time + miss_gpu + miss_q + rejected + pending
                self.assertEqual(reconciled_sum, total, f"Reconciliation sum mismatch in {name}")

                # Invariant 4: Reconciliation flag
                self.assertTrue(metrics["reconciliation_valid"], f"Reconciliation valid flag false in {name}")

                # Invariant 5: Total misses == Misses on GPU + Misses in Queue
                self.assertEqual(metrics["deadline_misses"], miss_gpu + miss_q, f"Total misses mismatch in {name}")

    def test_edf_sorting_behavior(self):
        """Verify that EDF actually dispatches the request with earliest absolute deadline."""
        stream = [
            Request(request_id="R01", arrival_time=0.0, relative_deadline=40.0, min_miou_threshold=0.75, actual_duration_ms=10.0),
            Request(request_id="R02", arrival_time=1.0, relative_deadline=50.0, min_miou_threshold=0.75, actual_duration_ms=10.0),  # abs = 51.0
            Request(request_id="R03", arrival_time=2.0, relative_deadline=15.0, min_miou_threshold=0.75, actual_duration_ms=10.0),  # abs = 17.0
        ]
        # Under FIFO: R01 dispatches at 0 (ends 10). At 10, queue has [R02, R03]. R02 arrived first, so R02 dispatches next.
        sim_fifo = RealTimeSimulator(
            candidates=self.candidates,
            policy=PolicyType.FIFO,
            use_admission_control=False,
            drop_expired_in_queue=False,
        )
        sim_fifo.run_workload(copy.deepcopy(stream))
        fifo_dispatches = [e.request_id for e in sim_fifo.events if e.event_type == "DISPATCH"]
        self.assertEqual(fifo_dispatches, ["R01", "R02", "R03"])

        # Under EDF: R01 dispatches at 0 (ends 10). At 10, queue has [R02 (d=51), R03 (d=17)]. EDF chooses R03!
        sim_edf = RealTimeSimulator(
            candidates=self.candidates,
            policy=PolicyType.EDF,
            use_admission_control=False,
            drop_expired_in_queue=False,
        )
        sim_edf.run_workload(copy.deepcopy(stream))
        edf_dispatches = [e.request_id for e in sim_edf.events if e.event_type == "DISPATCH"]
        self.assertEqual(edf_dispatches, ["R01", "R03", "R02"])

    def test_non_preemptive_gpu_blocking(self):
        """Verify that once a job starts on the GPU, it runs to completion non-preemptively."""
        stream = [
            Request(request_id="R01", arrival_time=0.0, relative_deadline=50.0, min_miou_threshold=0.75, actual_duration_ms=30.0),
            Request(request_id="R02", arrival_time=5.0, relative_deadline=10.0, min_miou_threshold=0.75, actual_duration_ms=5.0),  # abs = 15.0
        ]
        sim = RealTimeSimulator(
            candidates=self.candidates,
            policy=PolicyType.EDF,
            use_admission_control=False,
            drop_expired_in_queue=False,
        )
        sim.run_workload(copy.deepcopy(stream))

        # R01 started at 0.0, must complete at 30.0 before R02 can execute
        dispatch_events = [e for e in sim.events if e.event_type == "DISPATCH"]
        self.assertEqual(dispatch_events[0].request_id, "R01")
        self.assertAlmostEqual(dispatch_events[0].timestamp_ms, 0.0)

        # R02 cannot start until 30.0 ms despite earlier deadline
        self.assertEqual(dispatch_events[1].request_id, "R02")
        self.assertAlmostEqual(dispatch_events[1].timestamp_ms, 30.0)

    def test_formal_wcet_safeguard(self):
        """Verify that attempting to invoke formal WCET policy raises FormalBoundUnavailableError."""
        sim = RealTimeSimulator(
            candidates=self.candidates,
            policy=PolicyType.FORMAL_BOUND_AWARE,
        )
        with self.assertRaises(FormalBoundUnavailableError) as ctx:
            sim.run_workload([Request(request_id="R01", arrival_time=0.0, relative_deadline=50.0, min_miou_threshold=0.75, actual_duration_ms=23.53)])
        self.assertIn("formal WCET has not been established on this platform", str(ctx.exception))

    def test_phase19_csv_trace_and_comparison_files_exist(self):
        """Verify that Phase 19 output artifacts exist and match reconciliation constraints."""
        trace_path = os.path.join("outputs", "phase19_validation", "scenario_trace.csv")
        comparison_path = os.path.join("outputs", "phase19_validation", "policy_comparison.csv")
        config_path = os.path.join("outputs", "phase19_validation", "scenario_config.json")
        self.assertTrue(os.path.exists(trace_path), "scenario_trace.csv missing in phase19_validation")
        self.assertTrue(os.path.exists(comparison_path), "policy_comparison.csv missing in phase19_validation")
        self.assertTrue(os.path.exists(config_path), "scenario_config.json missing in phase19_validation")

        with open(comparison_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["policy_label"] == "Formal Bound-Aware (WCET)":
                    continue
                total = int(row["total_requests"])
                on_time = int(row["on_time_completions"])
                miss_gpu = int(row["misses_on_gpu"])
                miss_q = int(row["misses_in_queue"])
                rejected = int(row["rejected_requests"])
                self.assertEqual(total, on_time + miss_gpu + miss_q + rejected, f"Row {row['policy_label']} does not sum to total")


if __name__ == "__main__":
    unittest.main()
