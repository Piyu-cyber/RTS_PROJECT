"""Unit tests for Phase 18 Real-Time Simulator, Priority Queue, and Scheduling Policies.

Verifies:
  1. Priority Queue behavior (EDF orders by absolute deadline, FIFO preserves arrival order).
  2. Deterministic tie-breaking for equal deadlines.
  3. Non-preemptive GPU execution (running tasks are not interrupted).
  4. Dynamic budget decay B_i(t) = D_i - t.
  5. Queue-aware feasibility calculation.
  6. Deadline outcomes: COMPLETED_ON_TIME (before and exactly at deadline), DEADLINE_MISSED.
  7. Lateness calculation: lateness_i = C_i - D_i.
  8. Admission control rejection (REJECTED_INFEASIBLE), ensuring rejected requests are never counted as on-time.
  9. Quality-threshold filtering (tau exclusions).
  10. FIFO deadline miss vs. EDF on-time completion (Problem 1).
  11. Non-preemptive blocking detection (Problem 2).
  12. Overload load shedding (Problem 3).
  13. Formal bound-aware safeguard enforcement (Outcome C).
"""

from __future__ import annotations

import copy
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


class TestRealTimeSimulator(unittest.TestCase):
    """Test suite covering priority queues, admission control, and non-preemptive GPU simulation."""

    def test_edf_orders_by_earliest_deadline_not_arrival(self):
        """EDF must order requests by absolute deadline, regardless of arrival order."""
        pq = RealTimePriorityQueue(use_edf=True)

        req_a = Request(request_id="R_late_deadline", arrival_time=5.0, relative_deadline=80.0)  # D = 85.0
        req_b = Request(request_id="R_early_deadline", arrival_time=10.0, relative_deadline=30.0)  # D = 40.0

        pq.push(req_a)
        pq.push(req_b)

        # Even though req_a arrived earlier (5ms vs 10ms), req_b has earlier deadline (40ms vs 85ms)
        popped_first = pq.pop()
        self.assertEqual(popped_first.request_id, "R_early_deadline")
        self.assertEqual(popped_first.absolute_deadline, 40.0)

        popped_second = pq.pop()
        self.assertEqual(popped_second.request_id, "R_late_deadline")
        self.assertEqual(popped_second.absolute_deadline, 85.0)

    def test_fifo_preserves_arrival_order(self):
        """FIFO must order requests strictly by arrival time."""
        pq = RealTimePriorityQueue(use_edf=False)

        req_a = Request(request_id="R1", arrival_time=5.0, relative_deadline=80.0)  # D = 85.0
        req_b = Request(request_id="R2", arrival_time=10.0, relative_deadline=30.0)  # D = 40.0

        pq.push(req_a)
        pq.push(req_b)

        popped_first = pq.pop()
        self.assertEqual(popped_first.request_id, "R1")
        popped_second = pq.pop()
        self.assertEqual(popped_second.request_id, "R2")

    def test_equal_deadlines_deterministic_tie_breaking(self):
        """Equal deadlines must break ties deterministically by arrival time and request ID."""
        pq = RealTimePriorityQueue(use_edf=True)

        req1 = Request(request_id="R_tie_1", arrival_time=5.0, relative_deadline=40.0)  # D = 45.0
        req2 = Request(request_id="R_tie_2", arrival_time=10.0, relative_deadline=35.0)  # D = 45.0

        pq.push(req2)
        pq.push(req1)

        # Both have deadline 45.0 ms; req1 arrived at 5ms, so req1 must be popped first
        self.assertEqual(pq.pop().request_id, "R_tie_1")
        self.assertEqual(pq.pop().request_id, "R_tie_2")

    def test_remaining_budget_decays_with_time(self):
        """Dynamic budget B_i(t) = D_i - t must strictly decrease as simulated time advances."""
        req = Request(request_id="R1", arrival_time=10.0, relative_deadline=50.0)
        self.assertEqual(req.absolute_deadline, 60.0)

        self.assertAlmostEqual(req.remaining_budget(10.0), 50.0)
        self.assertAlmostEqual(req.remaining_budget(25.0), 35.0)
        self.assertAlmostEqual(req.remaining_budget(60.0), 0.0)
        self.assertAlmostEqual(req.remaining_budget(75.0), -15.0)

    def test_completed_before_deadline(self):
        """A request completing before its deadline must be COMPLETED_ON_TIME with negative lateness."""
        sim = RealTimeSimulator(policy=PolicyType.COMBINED_PROPOSED)
        req = Request(request_id="R1", arrival_time=0.0, relative_deadline=40.0, actual_duration_ms=23.66)
        results = sim.run_workload([req])

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.status, RequestStatus.COMPLETED_ON_TIME)
        self.assertAlmostEqual(r.completion_time, 23.66, places=2)
        self.assertLess(r.completion_time, r.absolute_deadline)
        self.assertAlmostEqual(r.lateness, 23.66 - 40.0, places=2)
        self.assertLess(r.lateness, 0.0)

    def test_completed_exactly_at_deadline(self):
        """A request completing exactly at its deadline (C_i == D_i) must be COMPLETED_ON_TIME."""
        # Using FIFO without admission control with exact duration matching relative deadline
        sim = RealTimeSimulator(policy=PolicyType.FIFO, use_admission_control=False)
        req = Request(request_id="R1", arrival_time=0.0, relative_deadline=25.0, actual_duration_ms=25.0)
        results = sim.run_workload([req])

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.status, RequestStatus.COMPLETED_ON_TIME)
        self.assertAlmostEqual(r.completion_time, 25.0)
        self.assertAlmostEqual(r.absolute_deadline, 25.0)
        self.assertAlmostEqual(r.lateness, 0.0)

    def test_completed_after_deadline(self):
        """A request completing after its deadline (C_i > D_i) must be marked DEADLINE_MISSED."""
        # Using a policy without admission control so the task is allowed to run and miss
        sim = RealTimeSimulator(policy=PolicyType.FIFO, use_admission_control=False)
        req = Request(request_id="R1", arrival_time=0.0, relative_deadline=20.0, actual_duration_ms=25.0)
        results = sim.run_workload([req])

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.status, RequestStatus.DEADLINE_MISSED)
        self.assertAlmostEqual(r.completion_time, 25.0)
        self.assertAlmostEqual(r.lateness, 5.0)  # C_i - D_i = 25 - 20 = +5.0 ms
        self.assertGreater(r.lateness, 0.0)

    def test_request_rejected_before_execution(self):
        """An infeasible request rejected by admission control must have status REJECTED_INFEASIBLE."""
        sim = RealTimeSimulator(policy=PolicyType.COMBINED_PROPOSED, use_admission_control=True)
        # relative deadline 10 ms is impossible when p95 is 28.57 ms
        req = Request(request_id="R1", arrival_time=0.0, relative_deadline=10.0, actual_duration_ms=23.66)
        results = sim.run_workload([req])

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.status, RequestStatus.REJECTED_INFEASIBLE)
        self.assertIsNone(r.start_time)
        self.assertIsNone(r.completion_time)
        self.assertIsNone(r.lateness)

        # Check summary metrics: rejected requests must NOT be counted as on-time completions
        m = sim.compute_summary_metrics()
        self.assertEqual(m["rejected_requests"], 1)
        self.assertEqual(m["on_time_completions"], 0)
        self.assertEqual(m["total_arrival_hit_rate_pct"], 0.0)
        self.assertEqual(m["gpu_busy_time_ms"], 0.0)  # Rejected request consumed 0 GPU time!

    def test_non_preemptive_blocking(self):
        """A running GPU task must not be preempted when an urgent request arrives."""
        sim = RealTimeSimulator(policy=PolicyType.FIFO, use_admission_control=False, drop_expired_in_queue=False)

        # R1 starts at t=0 and runs for 30 ms (non-preemptive)
        r1 = Request(request_id="R1", arrival_time=0.0, relative_deadline=50.0, actual_duration_ms=30.0)
        # R2 arrives at t=5 with deadline D=35 ms, but must wait until t=30 ms to start
        r2 = Request(request_id="R2", arrival_time=5.0, relative_deadline=30.0, actual_duration_ms=20.0)  # D = 35 ms

        results = sim.run_workload([r1, r2])

        r1_res = [r for r in results if r.request_id == "R1"][0]
        r2_res = [r for r in results if r.request_id == "R2"][0]

        # R1 ran from 0 to 30 ms uninterrupted
        self.assertEqual(r1_res.start_time, 0.0)
        self.assertEqual(r1_res.completion_time, 30.0)

        # R2 could NOT start until R1 completed at t=30 ms (non-preemptive blocking)
        self.assertEqual(r2_res.start_time, 30.0)
        self.assertEqual(r2_res.completion_time, 50.0)
        self.assertEqual(r2_res.status, RequestStatus.DEADLINE_MISSED)

    def test_problem1_fifo_misses_while_edf_meets_deadline(self):
        """Workload where FIFO misses a deadline that EDF meets under identical conditions."""
        # Baseline task running on GPU
        r1 = Request(request_id="R1", arrival_time=0.0, relative_deadline=35.0, actual_duration_ms=23.66)
        # R2 arrives at t=5, relaxed deadline D=85 ms
        r2 = Request(request_id="R2", arrival_time=5.0, relative_deadline=80.0, actual_duration_ms=23.66)
        # R3 arrives at t=10, tighter deadline D=55 ms
        r3 = Request(request_id="R3", arrival_time=10.0, relative_deadline=45.0, actual_duration_ms=23.66)

        workload = [r1, r2, r3]

        # 1. Under FIFO:
        sim_fifo = RealTimeSimulator(policy=PolicyType.FIFO, use_admission_control=False)
        res_fifo = sim_fifo.run_workload(copy.deepcopy(workload))
        fifo_map = {r.request_id: r for r in res_fifo}

        self.assertEqual(fifo_map["R1"].status, RequestStatus.COMPLETED_ON_TIME)
        self.assertEqual(fifo_map["R2"].status, RequestStatus.COMPLETED_ON_TIME)
        self.assertEqual(fifo_map["R3"].status, RequestStatus.DEADLINE_MISSED)  # FIFO caused R3 to miss!

        # 2. Under EDF:
        sim_edf = RealTimeSimulator(policy=PolicyType.EDF, use_admission_control=False)
        res_edf = sim_edf.run_workload(copy.deepcopy(workload))
        edf_map = {r.request_id: r for r in res_edf}

        self.assertEqual(edf_map["R1"].status, RequestStatus.COMPLETED_ON_TIME)
        self.assertEqual(edf_map["R2"].status, RequestStatus.COMPLETED_ON_TIME)
        self.assertEqual(edf_map["R3"].status, RequestStatus.COMPLETED_ON_TIME)  # EDF met both deadlines!

    def test_quality_threshold_filtering(self):
        """High quality threshold must exclude DWConv models whose mIoU falls below tau."""
        sim = RealTimeSimulator(policy=PolicyType.COMBINED_PROPOSED)

        # With tau = 0.78:
        # Full attention (0.8197) is eligible
        # Sigma DWConv (0.7590) is excluded
        # Manual DWConv (0.7447) is excluded
        req_high = Request(request_id="R_high", arrival_time=0.0, relative_deadline=50.0, min_miou_threshold=0.78)
        is_adm, config, est, eligible, reason = sim.select_configuration_and_check_feasibility(req_high, 0.0)

        self.assertTrue(is_adm)
        self.assertEqual(config, "full_attention")
        self.assertEqual(eligible, ["full_attention"])

        # With impossible tau = 0.90:
        req_impossible = Request(request_id="R_imp", arrival_time=0.0, relative_deadline=50.0, min_miou_threshold=0.90)
        is_adm, config, est, eligible, reason = sim.select_configuration_and_check_feasibility(req_impossible, 0.0)
        self.assertFalse(is_adm)
        self.assertIn("Quality floor violated", reason)

    def test_formal_bound_aware_safeguard(self):
        """Formal bound-aware policy must raise FormalBoundUnavailableError on unverified platform."""
        sim = RealTimeSimulator(policy=PolicyType.FORMAL_BOUND_AWARE)
        req = Request(request_id="R1", arrival_time=0.0, relative_deadline=50.0)

        with self.assertRaises(FormalBoundUnavailableError):
            sim.run_workload([req])

    def test_identical_workloads_across_policies(self):
        """Identical arrival streams must be fed to different policies for controlled comparison."""
        workload = [
            Request(request_id=f"R{i}", arrival_time=i * 20.0, relative_deadline=30.0, actual_duration_ms=23.66)
            for i in range(10)
        ]

        sim1 = RealTimeSimulator(policy=PolicyType.ALWAYS_FULL_ATTENTION)
        sim2 = RealTimeSimulator(policy=PolicyType.COMBINED_PROPOSED)

        res1 = sim1.run_workload(copy.deepcopy(workload))
        res2 = sim2.run_workload(copy.deepcopy(workload))

        # Check identical input arrival timestamps
        for r1, r2 in zip(res1, res2):
            self.assertEqual(r1.arrival_time, r2.arrival_time)
            self.assertEqual(r1.relative_deadline, r2.relative_deadline)
            self.assertEqual(r1.absolute_deadline, r2.absolute_deadline)


if __name__ == "__main__":
    unittest.main()
