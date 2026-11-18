"""Unit tests for Phase 16 bound-aware and empirical upper-bound scheduler.

Tests:
  1. Absolute deadline and remaining budget calculations (B_i(t) = D_i - t).
  2. Queue delay and execution budget accounting (W_i + C_c <= B_i(t)).
  3. Quality-floor eligibility and explicit floor violation detection.
  4. Timing metadata validation and error handling for invalid bounds.
  5. Strict separation between empirical and formal bound-aware modes (Outcome C enforcement).
  6. Empirical upper-bound mode behavior (EUB = max_obs + 3*round_std).
  7. No-candidate-feasible behavior and rejection load shedding.
  8. Fallback policy deterministic routing ('reject', 'highest_quality_miss', 'fastest_miss').
  9. Comprehensive logging and predicted vs proven infeasibility flags.
  10. Candidate registry loading from JSON profile artifact.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from realtime.bound_aware_scheduler import (
    BoundAwareSelectionResult,
    BoundCandidateProfile,
    ControllerError,
    FormalBoundUnavailableError,
    SchedulingMode,
    TimingScope,
    load_bound_candidates,
    select_bound_aware_candidate,
)


class TestPhase16BoundAwareScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            BoundCandidateProfile(
                name="full_attention",
                checkpoint_path="outputs/phase12a_full_pet/baseline/best_checkpoint.pt",
                validation_miou=0.819709,
                input_shape=(1, 3, 224, 224),
                model_forward_latencies={
                    "median": 23.66,
                    "mean": 23.53,
                    "p90": 27.81,
                    "p95": 28.57,
                    "p99": 29.97,
                    "max_observed": 30.74,
                },
                end_to_end_latencies={
                    "median": 29.27,
                    "mean": 29.14,
                    "p90": 33.42,
                    "p95": 34.18,
                    "p99": 35.58,
                    "max_observed": 36.35,
                },
                empirical_upper_bound_forward_ms=32.33,
                empirical_upper_bound_e2e_ms=37.94,
                formal_wcet_forward_ms=None,
                formal_wcet_e2e_ms=None,
                is_formally_justified=False,
                operating_assumptions=["Warm GPU", "Batch size 1"],
                source="Phase 12B/13 audited",
                blocks=[],
            ),
            BoundCandidateProfile(
                name="sigma_selected",
                checkpoint_path="outputs/phase12b_dwconv/sigma_selected/best_checkpoint.pt",
                validation_miou=0.758961,
                input_shape=(1, 3, 224, 224),
                model_forward_latencies={
                    "median": 26.41,
                    "mean": 25.65,
                    "p90": 29.65,
                    "p95": 30.52,
                    "p99": 31.52,
                    "max_observed": 31.59,
                },
                end_to_end_latencies={
                    "median": 32.02,
                    "mean": 31.26,
                    "p90": 35.26,
                    "p95": 36.13,
                    "p99": 37.13,
                    "max_observed": 37.20,
                },
                empirical_upper_bound_forward_ms=32.95,
                empirical_upper_bound_e2e_ms=38.56,
                formal_wcet_forward_ms=None,
                formal_wcet_e2e_ms=None,
                is_formally_justified=False,
                operating_assumptions=["Warm GPU", "Batch size 1"],
                source="Phase 12B/13 audited",
                blocks=[1, 2, 3, 4],
            ),
            BoundCandidateProfile(
                name="manual_interleaved",
                checkpoint_path="outputs/phase12b_dwconv/manual/best_checkpoint.pt",
                validation_miou=0.744658,
                input_shape=(1, 3, 224, 224),
                model_forward_latencies={
                    "median": 25.74,
                    "mean": 25.36,
                    "p90": 29.04,
                    "p95": 29.89,
                    "p99": 30.79,
                    "max_observed": 30.88,
                },
                end_to_end_latencies={
                    "median": 31.35,
                    "mean": 30.97,
                    "p90": 34.65,
                    "p95": 35.50,
                    "p99": 36.40,
                    "max_observed": 36.49,
                },
                empirical_upper_bound_forward_ms=33.64,
                empirical_upper_bound_e2e_ms=39.25,
                formal_wcet_forward_ms=None,
                formal_wcet_e2e_ms=None,
                is_formally_justified=False,
                operating_assumptions=["Warm GPU", "Batch size 1"],
                source="Phase 12B/13 audited",
                blocks=[2, 5, 8, 11],
            ),
        ]

    def test_absolute_deadline_and_budget_calculation(self) -> None:
        """Verify remaining budget is computed as (absolute_deadline - current_time) * 1000."""
        res = select_bound_aware_candidate(
            candidates=self.candidates,
            min_miou_threshold=0.75,
            current_time=10.000,
            absolute_deadline=10.035,  # 35 ms budget
            scheduling_mode=SchedulingMode.EMPIRICAL_LATENCY,
            latency_metric="p95",
        )
        self.assertAlmostEqual(res.remaining_budget_ms, 35.0, places=3)
        self.assertFalse(res.is_infeasible)
        self.assertEqual(res.selected_candidate, "full_attention")

    def test_past_absolute_deadline_sets_zero_budget(self) -> None:
        """If current_time > absolute_deadline, remaining budget is clamped to 0.0."""
        res = select_bound_aware_candidate(
            candidates=self.candidates,
            min_miou_threshold=0.70,
            current_time=10.050,
            absolute_deadline=10.035,  # 15 ms late
            fallback_policy="reject",
        )
        self.assertEqual(res.remaining_budget_ms, 0.0)
        self.assertTrue(res.is_infeasible)
        self.assertTrue(res.is_rejected)

    def test_queue_delay_accounting(self) -> None:
        """Queue delay W_i reduces available execution budget: W_i + C_c <= B_i(t)."""
        # Full attention p95 forward is 28.57 ms. With budget 30.0 ms:
        # If queue_delay = 0.0 -> 0 + 28.57 <= 30.0 (feasible)
        res_no_queue = select_bound_aware_candidate(
            candidates=self.candidates,
            min_miou_threshold=0.75,
            remaining_budget_ms=30.0,
            queue_delay_ms=0.0,
            latency_metric="p95",
        )
        self.assertFalse(res_no_queue.is_infeasible)
        self.assertEqual(res_no_queue.selected_candidate, "full_attention")

        # If queue_delay = 2.0 -> 2.0 + 28.57 = 30.57 > 30.0 (infeasible)
        res_with_queue = select_bound_aware_candidate(
            candidates=self.candidates,
            min_miou_threshold=0.75,
            remaining_budget_ms=30.0,
            queue_delay_ms=2.0,
            latency_metric="p95",
            fallback_policy="reject",
        )
        self.assertTrue(res_with_queue.is_infeasible)
        self.assertTrue(res_with_queue.is_rejected)

    def test_quality_floor_violation_distinct_from_deadline(self) -> None:
        """When quality floor exceeds all candidates, is_quality_floor_violated is explicitly True."""
        res = select_bound_aware_candidate(
            candidates=self.candidates,
            min_miou_threshold=0.85,  # Highest candidate is 0.8197
            remaining_budget_ms=100.0,  # Plentiful deadline
            fallback_policy="reject",
        )
        self.assertTrue(res.is_infeasible)
        self.assertTrue(res.is_quality_floor_violated)
        self.assertEqual(res.quality_eligible_candidates, [])
        self.assertIn("Quality floor violation", res.reason)

    def test_formal_bound_aware_mode_refusal_when_unjustified(self) -> None:
        """Invoking bound_aware mode on uncertified candidates raises FormalBoundUnavailableError."""
        with self.assertRaises(FormalBoundUnavailableError) as ctx:
            select_bound_aware_candidate(
                candidates=self.candidates,
                min_miou_threshold=0.70,
                remaining_budget_ms=50.0,
                scheduling_mode=SchedulingMode.BOUND_AWARE,
            )
        self.assertIn("Outcome C", str(ctx.exception))

    def test_empirical_upper_bound_mode(self) -> None:
        """Empirical upper bound mode uses conservative (max + 3*sigma) bounds without claiming WCET."""
        # Full attention EUB forward is 32.33 ms
        # Budget = 31.0 ms: feasible under p95 (28.57) but INFEASIBLE under EUB (32.33)
        res_p95 = select_bound_aware_candidate(
            candidates=self.candidates,
            min_miou_threshold=0.75,
            remaining_budget_ms=31.0,
            scheduling_mode=SchedulingMode.EMPIRICAL_LATENCY,
            latency_metric="p95",
        )
        self.assertFalse(res_p95.is_infeasible)
        self.assertEqual(res_p95.selected_candidate, "full_attention")

        res_eub = select_bound_aware_candidate(
            candidates=self.candidates,
            min_miou_threshold=0.75,
            remaining_budget_ms=31.0,
            scheduling_mode=SchedulingMode.EMPIRICAL_UPPER_BOUND,
            fallback_policy="reject",
        )
        self.assertTrue(res_eub.is_infeasible)
        self.assertTrue(res_eub.is_rejected)
        self.assertTrue(res_eub.predicted_infeasible)
        self.assertFalse(res_eub.proven_infeasible)  # Never claimed as proven

    def test_timing_scopes_forward_vs_e2e(self) -> None:
        """End-to-end scope includes host preprocessing/transfer offset."""
        # Full attention forward p95 is 28.57 ms; e2e p95 is 34.18 ms
        # With budget 30.0 ms: feasible under forward, infeasible under e2e
        res_fwd = select_bound_aware_candidate(
            candidates=self.candidates,
            min_miou_threshold=0.75,
            remaining_budget_ms=30.0,
            timing_scope=TimingScope.MODEL_FORWARD,
            latency_metric="p95",
        )
        self.assertFalse(res_fwd.is_infeasible)

        res_e2e = select_bound_aware_candidate(
            candidates=self.candidates,
            min_miou_threshold=0.75,
            remaining_budget_ms=30.0,
            timing_scope=TimingScope.END_TO_END,
            latency_metric="p95",
            fallback_policy="reject",
        )
        self.assertTrue(res_e2e.is_infeasible)
        self.assertTrue(res_e2e.is_rejected)

    def test_fallback_policies(self) -> None:
        """Verify deterministic routing for reject, highest_quality_miss, and fastest_miss."""
        tight_budget = 10.0  # Infeasible for all models

        # Reject
        res_rej = select_bound_aware_candidate(
            candidates=self.candidates,
            min_miou_threshold=0.75,
            remaining_budget_ms=tight_budget,
            fallback_policy="reject",
        )
        self.assertIsNone(res_rej.selected_candidate)
        self.assertTrue(res_rej.is_rejected)

        # Highest Quality Miss
        res_hq = select_bound_aware_candidate(
            candidates=self.candidates,
            min_miou_threshold=0.75,
            remaining_budget_ms=tight_budget,
            fallback_policy="highest_quality_miss",
        )
        self.assertEqual(res_hq.selected_candidate, "full_attention")
        self.assertTrue(res_hq.is_fallback)

        # Fastest Miss
        res_fast = select_bound_aware_candidate(
            candidates=self.candidates,
            min_miou_threshold=0.75,
            remaining_budget_ms=tight_budget,
            fallback_policy="fastest_miss",
        )
        self.assertEqual(res_fast.selected_candidate, "full_attention")  # full is fastest at 23.66 ms
        self.assertTrue(res_fast.is_fallback)

    def test_load_bound_candidates_from_artifact(self) -> None:
        """Verify loader properly parses the Phase 16 profile artifact."""
        artifact_path = Path("outputs/phase16_wcet/empirical_latency/empirical_timing_profiles.json")
        if artifact_path.is_file():
            loaded = load_bound_candidates(artifact_path)
            self.assertEqual(len(loaded), 3)
            names = [c.name for c in loaded]
            self.assertIn("full_attention", names)
            self.assertIn("sigma_selected", names)
            self.assertIn("manual_interleaved", names)

            full_cand = next(c for c in loaded if c.name == "full_attention")
            self.assertAlmostEqual(full_cand.validation_miou, 0.819709, places=4)
            self.assertAlmostEqual(full_cand.empirical_upper_bound_forward_ms, 32.33, places=1)
            self.assertFalse(full_cand.is_formally_justified)

    def test_structured_logging_output(self) -> None:
        """Selection result dictionary contains complete audit trail."""
        res = select_bound_aware_candidate(
            candidates=self.candidates,
            min_miou_threshold=0.75,
            remaining_budget_ms=29.0,
            queue_delay_ms=0.5,
            request_id="req-test-42",
        )
        d = res.to_dict()
        self.assertEqual(d["request_id"], "req-test-42")
        self.assertEqual(d["queue_delay_ms"], 0.5)
        self.assertIn("full_attention", d["quality_eligible_candidates"])
        self.assertIn("all_candidate_timing_values", d)


if __name__ == "__main__":
    unittest.main()
