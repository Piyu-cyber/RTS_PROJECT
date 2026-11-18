"""Unit tests for Phase 12C candidate registry schema, eligibility filtering, and scheduler invariants."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from realtime.accuracy_scheduler import (
    AccuracyCandidateProfile,
    select_accuracy_constrained_mode,
)


class TestPhase12CScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.registry_path = self.project_root / "outputs" / "phase12c_scheduler" / "candidate_registry.json"
        self.assertTrue(self.registry_path.is_file(), f"Missing registry at {self.registry_path}")

        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.modes = data["modes"]

        self.candidates = []
        for name, m in self.modes.items():
            f_lat = m["model_forward"]
            self.candidates.append(
                AccuracyCandidateProfile(
                    name=name,
                    validation_miou=m["validation_miou"],
                    p90_ms=f_lat["p90_ms"],
                    p95_ms=f_lat["p95_ms"],
                    p99_ms=f_lat["p99_ms"],
                    max_observed_ms=f_lat["max_observed_ms"],
                    median_ms=f_lat["median_ms"],
                    blocks=m["blocks"],
                )
            )

    def test_registry_schema_and_provenance(self) -> None:
        """Verify candidate registry has exact 3 candidates and verified Phase 12B metrics."""
        self.assertEqual(len(self.modes), 3)
        self.assertIn("full_attention", self.modes)
        self.assertIn("sigma_selected", self.modes)
        self.assertIn("manual_interleaved", self.modes)

        self.assertAlmostEqual(self.modes["full_attention"]["validation_miou"], 0.819709, places=4)
        self.assertAlmostEqual(self.modes["sigma_selected"]["validation_miou"], 0.7590, places=3)
        self.assertAlmostEqual(self.modes["manual_interleaved"]["validation_miou"], 0.7447, places=3)

    def test_accuracy_floor_eligibility_scenarios(self) -> None:
        """Verify candidate eligibility under declared quality floors."""
        # Floor 0.80 -> Only full_attention eligible
        res_080 = select_accuracy_constrained_mode(50.0, self.candidates, min_miou_threshold=0.80)
        self.assertEqual(res_080.quality_eligible_candidates, ["full_attention"])
        self.assertEqual(res_080.selected_candidate, "full_attention")

        # Floor 0.75 -> full_attention and sigma_selected eligible
        res_075 = select_accuracy_constrained_mode(50.0, self.candidates, min_miou_threshold=0.75)
        self.assertEqual(res_075.quality_eligible_candidates, ["full_attention", "sigma_selected"])
        self.assertEqual(res_075.selected_candidate, "full_attention")

        # Floor 0.74 -> All 3 eligible
        res_074 = select_accuracy_constrained_mode(50.0, self.candidates, min_miou_threshold=0.74)
        self.assertEqual(sorted(res_074.quality_eligible_candidates), ["full_attention", "manual_interleaved", "sigma_selected"])

    def test_rejection_policy_on_tight_deadline(self) -> None:
        """Verify request is rejected when deadline cannot be satisfied by any candidate."""
        # Tight deadline 15ms (all candidates have p95 > 28ms)
        res = select_accuracy_constrained_mode(
            remaining_deadline_ms=15.0,
            candidates=self.candidates,
            min_miou_threshold=0.74,
            latency_metric="p95",
            fallback_policy="reject",
        )
        self.assertTrue(res.is_infeasible)
        self.assertTrue(res.is_rejected)
        self.assertIsNone(res.selected_candidate)
        self.assertIsNone(res.selected_miou)

    def test_fallback_policies_on_tight_deadline(self) -> None:
        """Verify fallback behavior under highest_quality_miss vs fastest_miss."""
        # 1. highest_quality_miss -> selects full_attention (highest mIoU = 0.8197)
        res_hq = select_accuracy_constrained_mode(
            remaining_deadline_ms=15.0,
            candidates=self.candidates,
            min_miou_threshold=0.74,
            latency_metric="p95",
            fallback_policy="highest_quality_miss",
        )
        self.assertTrue(res_hq.is_infeasible)
        self.assertFalse(res_hq.is_rejected)
        self.assertEqual(res_hq.selected_candidate, "full_attention")

        # 2. fastest_miss -> selects full_attention (also lowest latency bound p95=28.57ms)
        res_fast = select_accuracy_constrained_mode(
            remaining_deadline_ms=15.0,
            candidates=self.candidates,
            min_miou_threshold=0.74,
            latency_metric="p95",
            fallback_policy="fastest_miss",
        )
        self.assertTrue(res_fast.is_infeasible)
        self.assertEqual(res_fast.selected_candidate, "full_attention")

    def test_no_accuracy_floor_violations_on_feasible_dispatch(self) -> None:
        """Verify that any feasible dispatch strictly satisfies mIoU >= tau."""
        for tau in [0.74, 0.75, 0.76, 0.80]:
            for d in [20.0, 25.0, 30.0, 35.0, 40.0]:
                res = select_accuracy_constrained_mode(
                    remaining_deadline_ms=d,
                    candidates=self.candidates,
                    min_miou_threshold=tau,
                    latency_metric="p95",
                    fallback_policy="reject",
                )
                if not res.is_rejected:
                    self.assertGreaterEqual(res.selected_miou, tau)


if __name__ == "__main__":
    unittest.main()
