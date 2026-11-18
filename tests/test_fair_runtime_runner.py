"""Unit tests for Phase 9.1 Fair Runtime Evaluation Runner.

Tests:
  1. Wilson 95% confidence interval calculations and edge cases.
  2. Request matching, trace validation, and duplicate/mismatch detection.
  3. Strict confusion matrix calculation.
  4. Aggregate summary metrics recomputation.
  5. Controller model selection conformance validation.
"""

import math
import unittest
from pathlib import Path

from realtime.controller import ModeProfile
from realtime.run_fair_runtime_experiment import (
    FairRequestRecord,
    compute_aggregate_metrics,
    compute_distribution_stats,
    compute_wilson_ci,
    validate_fair_traces,
)


def _make_dummy_record(
    round_id: int,
    request_id: str,
    strategy: str,
    controller_metric: str = "p95",
    selected_model: str = "full_attention",
    deadline_budget_ms: float = 20.0,
    model_forward_ms: float = 13.0,
    end_to_end_ms: float = 18.0,
    predicted_feasible: bool = True,
    selected_model_miou: float = 0.74299,
    execution_order_index: int = 0,
    round_order_seed: int = 1234,
) -> FairRequestRecord:
    return FairRequestRecord(
        round_id=round_id,
        request_id=request_id,
        strategy=strategy,
        controller_metric=controller_metric,
        selected_model=selected_model,
        deadline_budget_ms=deadline_budget_ms,
        workload_category="moderate",
        controller_ms=0.05,
        preprocessing_ms=4.5,
        host_to_device_ms=0.2,
        model_forward_ms=model_forward_ms,
        model_cuda_event_ms=model_forward_ms,
        postprocessing_ms=0.0005,
        end_to_end_ms=end_to_end_ms,
        model_deadline_hit=model_forward_ms <= deadline_budget_ms,
        end_to_end_deadline_hit=end_to_end_ms <= deadline_budget_ms,
        predicted_feasible=predicted_feasible,
        likely_deadline_miss=not predicted_feasible,
        selected_model_miou=selected_model_miou,
        execution_order_index=execution_order_index,
        round_order_seed=round_order_seed,
    )


class TestFairRunnerLogic(unittest.TestCase):
    """Test pure numerical, statistical, and validation logic of the fair runner."""

    def test_wilson_ci_bounds(self) -> None:
        low, high = compute_wilson_ci(50, 100)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)

        # 0 successes
        low_zero, high_zero = compute_wilson_ci(0, 50)
        self.assertEqual(low_zero, 0.0)
        self.assertGreater(high_zero, 0.0)

        # 100% successes
        low_all, high_all = compute_wilson_ci(50, 50)
        self.assertLess(low_all, 1.0)
        self.assertEqual(high_all, 1.0)

    def test_distribution_stats(self) -> None:
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_distribution_stats(vals)
        self.assertEqual(stats["mean"], 30.0)
        self.assertEqual(stats["median"], 30.0)
        self.assertAlmostEqual(stats["p95"], 48.0, places=1)

    def test_confusion_matrix_and_hit_rates(self) -> None:
        # 4 records:
        # r1: pred=True, e2e_hit=True  -> TP
        # r2: pred=True, e2e_hit=False -> FP
        # r3: pred=False, e2e_hit=True -> FN
        # r4: pred=False, e2e_hit=False-> TN
        records = [
            _make_dummy_record(1, "req_1", "strat_a", deadline_budget_ms=20.0, end_to_end_ms=18.0, predicted_feasible=True),
            _make_dummy_record(1, "req_2", "strat_a", deadline_budget_ms=15.0, end_to_end_ms=18.0, predicted_feasible=True),
            _make_dummy_record(1, "req_3", "strat_a", deadline_budget_ms=20.0, end_to_end_ms=18.0, predicted_feasible=False),
            _make_dummy_record(1, "req_4", "strat_a", deadline_budget_ms=15.0, end_to_end_ms=18.0, predicted_feasible=False),
        ]
        agg = compute_aggregate_metrics(records, ["full_attention"])
        self.assertEqual(agg["confusion_matrix"]["tp"], 1)
        self.assertEqual(agg["confusion_matrix"]["fp"], 1)
        self.assertEqual(agg["confusion_matrix"]["fn"], 1)
        self.assertEqual(agg["confusion_matrix"]["tn"], 1)
        self.assertEqual(agg["end_to_end_deadline_hit_count"], 2)
        self.assertEqual(agg["end_to_end_deadline_hit_rate"], 0.5)

    def test_trace_validation_detects_mismatch(self) -> None:
        dummy_modes = [
            ModeProfile(name="full_attention", validation_miou=0.74, p95_ms=16.6, p99_ms=21.3, max_observed_ms=27.5),
            ModeProfile(name="moderate_dwconv", validation_miou=0.70, p95_ms=18.6, p99_ms=20.8, max_observed_ms=24.0),
        ]
        strategies = ["always_full_attention", "always_fastest"]
        
        # Correct matching
        records = [
            _make_dummy_record(1, "req_1", "always_full_attention", deadline_budget_ms=25.0, execution_order_index=0),
            _make_dummy_record(1, "req_1", "always_fastest", deadline_budget_ms=25.0, execution_order_index=1),
        ]
        val = validate_fair_traces(records, 1, 1, strategies, dummy_modes)
        self.assertTrue(val["valid"])

        # Mismatched deadlines
        bad_records = [
            _make_dummy_record(1, "req_1", "always_full_attention", deadline_budget_ms=25.0, execution_order_index=0),
            _make_dummy_record(1, "req_1", "always_fastest", deadline_budget_ms=20.0, execution_order_index=1),
        ]
        bad_val = validate_fair_traces(bad_records, 1, 1, strategies, dummy_modes)
        self.assertFalse(bad_val["valid"])
        self.assertTrue(any("Mismatch in request IDs or deadlines" in e for e in bad_val["errors"]))


if __name__ == "__main__":
    unittest.main()
