"""Unit tests for the Phase 10C accuracy-constrained adaptive scheduler."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from realtime.accuracy_scheduler import (
    AccuracyCandidateProfile,
    AccuracySelectionResult,
    ControllerError,
    load_accuracy_candidates,
    select_accuracy_constrained_mode,
)


class TestAccuracyScheduler(unittest.TestCase):
    def setUp(self) -> None:
        # Standard candidate set modeled on empirical results
        self.candidates = [
            AccuracyCandidateProfile(
                name="full_attention",
                validation_miou=0.7430,
                p90_ms=14.85,
                p95_ms=15.65,
                p99_ms=18.62,
                max_observed_ms=19.98,
                median_ms=13.08,
                blocks=[],
            ),
            AccuracyCandidateProfile(
                name="moderate_dwconv",
                validation_miou=0.7036,
                p90_ms=15.88,
                p95_ms=16.41,
                p99_ms=19.99,
                max_observed_ms=23.85,
                median_ms=14.19,
                blocks=[2, 5, 8, 11],
            ),
            AccuracyCandidateProfile(
                name="sigma_b_dwconv",
                validation_miou=0.4801,
                p90_ms=15.61,
                p95_ms=17.14,
                p99_ms=18.62,
                max_observed_ms=20.06,
                median_ms=13.97,
                blocks=[1, 2, 3, 4],
            ),
        ]

    def test_quality_filtering_low_threshold(self) -> None:
        """With low threshold (0.40), all three candidates are quality-eligible."""
        res = select_accuracy_constrained_mode(
            remaining_deadline_ms=25.0,
            candidates=self.candidates,
            min_miou_threshold=0.40,
            latency_metric="p95",
        )
        self.assertFalse(res.is_infeasible)
        self.assertEqual(res.selected_candidate, "full_attention")
        self.assertEqual(len(res.quality_eligible_candidates), 3)
        self.assertEqual(len(res.jointly_feasible_candidates), 3)

    def test_quality_filtering_moderate_threshold(self) -> None:
        """With threshold 0.65, sigma_b_dwconv (0.4801) is filtered out."""
        res = select_accuracy_constrained_mode(
            remaining_deadline_ms=25.0,
            candidates=self.candidates,
            min_miou_threshold=0.65,
            latency_metric="p95",
        )
        self.assertFalse(res.is_infeasible)
        self.assertEqual(res.selected_candidate, "full_attention")
        self.assertEqual(set(res.quality_eligible_candidates), {"full_attention", "moderate_dwconv"})
        self.assertNotIn("sigma_b_dwconv", res.jointly_feasible_candidates)

    def test_quality_filtering_high_threshold(self) -> None:
        """With threshold 0.72, only full_attention (0.7430) is quality-eligible."""
        res = select_accuracy_constrained_mode(
            remaining_deadline_ms=20.0,
            candidates=self.candidates,
            min_miou_threshold=0.72,
            latency_metric="p95",
        )
        self.assertFalse(res.is_infeasible)
        self.assertEqual(res.selected_candidate, "full_attention")
        self.assertEqual(res.quality_eligible_candidates, ["full_attention"])

    def test_deadline_filtering(self) -> None:
        """When deadline is tight (16.0ms), moderate (16.41) and sigma (17.14) are deadline-infeasible under p95."""
        res = select_accuracy_constrained_mode(
            remaining_deadline_ms=16.0,
            candidates=self.candidates,
            min_miou_threshold=0.65,
            latency_metric="p95",
        )
        self.assertFalse(res.is_infeasible)
        self.assertEqual(res.selected_candidate, "full_attention")
        self.assertEqual(res.deadline_eligible_candidates, ["full_attention"])

    def test_infeasibility_and_highest_quality_fallback(self) -> None:
        """When deadline is tighter than all candidates (10.0ms), fallback selects highest quality."""
        res = select_accuracy_constrained_mode(
            remaining_deadline_ms=10.0,
            candidates=self.candidates,
            min_miou_threshold=0.65,
            latency_metric="p95",
            fallback_policy="highest_quality_miss",
        )
        self.assertTrue(res.is_infeasible)
        self.assertTrue(res.is_fallback)
        self.assertFalse(res.is_rejected)
        self.assertEqual(res.selected_candidate, "full_attention")

    def test_infeasibility_and_fastest_fallback(self) -> None:
        """When deadline is tighter than all (10.0ms), fastest_miss selects candidate with lowest p95 latency."""
        res = select_accuracy_constrained_mode(
            remaining_deadline_ms=10.0,
            candidates=self.candidates,
            min_miou_threshold=0.65,
            latency_metric="p95",
            fallback_policy="fastest_miss",
        )
        self.assertTrue(res.is_infeasible)
        self.assertTrue(res.is_fallback)
        self.assertEqual(res.selected_candidate, "full_attention")  # full_attention is lowest p95 (15.65)

    def test_infeasibility_and_reject_policy(self) -> None:
        """When fallback is reject, request is rejected with None selected."""
        res = select_accuracy_constrained_mode(
            remaining_deadline_ms=10.0,
            candidates=self.candidates,
            min_miou_threshold=0.65,
            latency_metric="p95",
            fallback_policy="reject",
        )
        self.assertTrue(res.is_infeasible)
        self.assertTrue(res.is_fallback)
        self.assertTrue(res.is_rejected)
        self.assertIsNone(res.selected_candidate)
        self.assertIsNone(res.selected_miou)
        self.assertIsNone(res.selected_latency_bound_ms)

    def test_tie_breaking(self) -> None:
        """Tie in mIoU breaks by lowest latency bound, then name."""
        candidates = [
            AccuracyCandidateProfile("cand_b", 0.70, 15.0, 16.0, 18.0, 20.0, 14.0),
            AccuracyCandidateProfile("cand_a", 0.70, 14.0, 15.0, 17.0, 19.0, 13.0),
        ]
        res = select_accuracy_constrained_mode(
            remaining_deadline_ms=20.0,
            candidates=candidates,
            min_miou_threshold=0.65,
            latency_metric="p95",
        )
        self.assertEqual(res.selected_candidate, "cand_a")

    def test_invalid_threshold_raises_error(self) -> None:
        with self.assertRaises(ControllerError):
            select_accuracy_constrained_mode(20.0, self.candidates, min_miou_threshold=-0.1)
        with self.assertRaises(ControllerError):
            select_accuracy_constrained_mode(20.0, self.candidates, min_miou_threshold=1.5)
        with self.assertRaises(ControllerError):
            select_accuracy_constrained_mode(20.0, self.candidates, min_miou_threshold=float("nan"))

    def test_invalid_deadline_raises_error(self) -> None:
        with self.assertRaises(ControllerError):
            select_accuracy_constrained_mode(-5.0, self.candidates, min_miou_threshold=0.65)

    def test_invalid_fallback_policy_raises_error(self) -> None:
        with self.assertRaises(ControllerError):
            select_accuracy_constrained_mode(
                20.0, self.candidates, min_miou_threshold=0.65, fallback_policy="invalid_policy"
            )

    def test_empty_candidates_raises_error(self) -> None:
        with self.assertRaises(ControllerError):
            select_accuracy_constrained_mode(20.0, [], min_miou_threshold=0.65)

    def test_load_accuracy_candidates_from_file(self) -> None:
        bench_path = Path("outputs/phase10c_adaptive_scheduler/benchmark_results.json")
        if bench_path.is_file():
            candidates = load_accuracy_candidates(bench_path)
            self.assertEqual(len(candidates), 3)
            names = {c.name for c in candidates}
            self.assertEqual(names, {"full_attention", "moderate_dwconv", "sigma_b_dwconv"})


if __name__ == "__main__":
    unittest.main()
