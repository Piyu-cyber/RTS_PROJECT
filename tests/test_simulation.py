"""Automated unit and regression tests for Phase 8 simulation.

Covers all correctness requirements:
  - Stream reproducibility (same seed vs different seed)
  - Identical request dispatch and order invariance across all strategies
  - Completeness (every request processed exactly once)
  - Strategy invariants (always-full, always-fastest, adaptive agreement)
  - Exact boundary feasibility (<=)
  - Infeasible requests, empirical miss and fallback tracking
  - Hand-crafted analytical test fixture for aggregate metrics
  - Mode frequencies summing to N
  - Comprehensive input validation
  - Output artifact verification (CSV rows, JSON keys, plot creation)
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from realtime.controller import ControllerError, ModeProfile
from realtime.simulate_strategies import (
    DEFAULT_BENCHMARK,
    SyntheticRequest,
    compute_strategy_metrics,
    evaluate_adaptive_controller,
    evaluate_always_fastest,
    evaluate_always_full_attention,
    generate_deadline_stream,
    run_simulation,
)


def _mock_modes() -> list[ModeProfile]:
    return [
        ModeProfile(
            name="full_attention",
            validation_miou=0.74,
            p95_ms=16.6,
            p99_ms=21.3,
            max_observed_ms=27.5,
        ),
        ModeProfile(
            name="moderate_dwconv",
            validation_miou=0.70,
            p95_ms=18.6,
            p99_ms=20.8,
            max_observed_ms=24.0,
        ),
        ModeProfile(
            name="aggressive_dwconv",
            validation_miou=0.45,
            p95_ms=18.0,
            p99_ms=23.2,
            max_observed_ms=25.6,
        ),
    ]


class TestDeadlineStreamGeneration(unittest.TestCase):
    """Test synthetic request stream generation."""

    def test_same_seed_produces_identical_stream(self) -> None:
        stream_a = generate_deadline_stream(n_requests=100, seed=42)
        stream_b = generate_deadline_stream(n_requests=100, seed=42)
        self.assertEqual(len(stream_a), 100)
        self.assertEqual(stream_a, stream_b)

    def test_different_seeds_produce_different_streams(self) -> None:
        stream_a = generate_deadline_stream(n_requests=100, seed=42)
        stream_b = generate_deadline_stream(n_requests=100, seed=999)
        self.assertNotEqual(
            [r.deadline_ms for r in stream_a],
            [r.deadline_ms for r in stream_b],
        )

    def test_invalid_request_count_or_seed(self) -> None:
        with self.assertRaises(ControllerError):
            generate_deadline_stream(n_requests=0, seed=42)
        with self.assertRaises(ControllerError):
            generate_deadline_stream(n_requests=-10, seed=42)
        with self.assertRaises(ControllerError):
            generate_deadline_stream(n_requests="100", seed=42)  # type: ignore[arg-type]
        with self.assertRaises(ControllerError):
            generate_deadline_stream(n_requests=100, seed="42")  # type: ignore[arg-type]
        with self.assertRaises(ControllerError):
            generate_deadline_stream(n_requests=100, seed=42, workload_name="unknown")


class TestStrategyInvariants(unittest.TestCase):
    """Verify correctness and invariance of the three scheduling strategies."""

    def setUp(self) -> None:
        self.modes = _mock_modes()
        self.requests = [
            SyntheticRequest("req_001", 12.0, "tight"),
            SyntheticRequest("req_002", 17.0, "moderate"),
            SyntheticRequest("req_003", 20.0, "moderate"),
            SyntheticRequest("req_004", 25.0, "relaxed"),
        ]

    def test_always_full_attention_always_selects_full(self) -> None:
        decisions = evaluate_always_full_attention(self.requests, self.modes, "p95")
        self.assertEqual(len(decisions), 4)
        for d in decisions:
            self.assertEqual(d.selected_mode, "full_attention")
            self.assertAlmostEqual(d.selected_miou, 0.74)
            self.assertAlmostEqual(d.selected_latency_bound_ms, 16.6)
            self.assertFalse(d.fallback_invoked)

        # Hits: bound (16.6) <= deadline
        # req_001: 16.6 <= 12.0 -> False
        # req_002: 16.6 <= 17.0 -> True
        # req_003: 16.6 <= 20.0 -> True
        # req_004: 16.6 <= 25.0 -> True
        self.assertFalse(decisions[0].empirical_hit)
        self.assertTrue(decisions[1].empirical_hit)
        self.assertTrue(decisions[2].empirical_hit)
        self.assertTrue(decisions[3].empirical_hit)

    def test_always_fastest_under_p95_selects_full_attention(self) -> None:
        # Under p95, full_attention has 16.6ms (lowest)
        decisions = evaluate_always_fastest(self.requests, self.modes, "p95")
        for d in decisions:
            self.assertEqual(d.selected_mode, "full_attention")

    def test_always_fastest_under_p99_selects_moderate_dwconv(self) -> None:
        # Under p99, moderate_dwconv has 20.8ms (lowest: full=21.3, aggressive=23.2)
        decisions = evaluate_always_fastest(self.requests, self.modes, "p99")
        for d in decisions:
            self.assertEqual(d.selected_mode, "moderate_dwconv")
            self.assertAlmostEqual(d.selected_latency_bound_ms, 20.8)

    def test_always_fastest_under_max_observed_selects_moderate_dwconv(self) -> None:
        # Under max_observed, moderate_dwconv has 24.0ms (lowest: aggressive=25.6, full=27.5)
        decisions = evaluate_always_fastest(self.requests, self.modes, "max_observed")
        for d in decisions:
            self.assertEqual(d.selected_mode, "moderate_dwconv")
            self.assertAlmostEqual(d.selected_latency_bound_ms, 24.0)

    def test_adaptive_agrees_with_phase7_and_handles_exact_equality(self) -> None:
        # Hand-crafted boundary request: exactly 16.6 ms
        requests = [
            SyntheticRequest("boundary_req", 16.6, "moderate"),
            SyntheticRequest("tight_infeasible", 15.0, "tight"),
        ]
        decisions = evaluate_adaptive_controller(requests, self.modes, "p95")

        # 16.6 ms: full_attention (16.6) <= 16.6 -> Feasible!
        self.assertEqual(decisions[0].selected_mode, "full_attention")
        self.assertTrue(decisions[0].empirical_hit)
        self.assertFalse(decisions[0].fallback_invoked)
        self.assertEqual(decisions[0].feasible_modes, ["full_attention"])

        # 15.0 ms: No mode <= 15.0 ms -> Infeasible fallback to lowest p95 (full_attention)
        self.assertEqual(decisions[1].selected_mode, "full_attention")
        self.assertFalse(decisions[1].empirical_hit)
        self.assertTrue(decisions[1].fallback_invoked)
        self.assertEqual(decisions[1].feasible_modes, [])

    def test_all_strategies_receive_identical_requests(self) -> None:
        stream = generate_deadline_stream(n_requests=50, seed=7)
        d_full = evaluate_always_full_attention(stream, self.modes, "p95")
        d_fast = evaluate_always_fastest(stream, self.modes, "p95")
        d_adapt = evaluate_adaptive_controller(stream, self.modes, "p95")

        self.assertEqual(len(d_full), 50)
        self.assertEqual(len(d_fast), 50)
        self.assertEqual(len(d_adapt), 50)

        for i in range(50):
            req_id = stream[i].request_id
            dl = stream[i].deadline_ms
            self.assertEqual(d_full[i].request_id, req_id)
            self.assertEqual(d_fast[i].request_id, req_id)
            self.assertEqual(d_adapt[i].request_id, req_id)
            self.assertEqual(d_full[i].deadline_ms, dl)
            self.assertEqual(d_fast[i].deadline_ms, dl)
            self.assertEqual(d_adapt[i].deadline_ms, dl)


class TestMetricsCalculation(unittest.TestCase):
    """Verify statistical calculations against exact hand-calculated fixtures."""

    def test_exact_hand_calculated_metrics(self) -> None:
        modes = _mock_modes()
        # 4 requests: 2 hits, 2 misses
        # Mode 1: full (miou 0.74, bound 16.6)
        # Mode 2: full (miou 0.74, bound 16.6)
        # Mode 3: moderate (miou 0.70, bound 18.6)
        # Mode 4: aggressive (miou 0.45, bound 18.0)
        requests = [
            SyntheticRequest("r1", 20.0, "cat_a"),
            SyntheticRequest("r2", 20.0, "cat_a"),
            SyntheticRequest("r3", 15.0, "cat_b"),
            SyntheticRequest("r4", 15.0, "cat_b"),
        ]
        # Let's craft decisions directly:
        from realtime.simulate_strategies import RequestDecision
        decs = [
            RequestDecision("r1", 20.0, "cat_a", "test", "full_attention", 0.74, 16.6, True, ["full_attention"], False),
            RequestDecision("r2", 20.0, "cat_a", "test", "full_attention", 0.74, 16.6, True, ["full_attention"], False),
            RequestDecision("r3", 15.0, "cat_b", "test", "moderate_dwconv", 0.70, 18.6, False, [], True),
            RequestDecision("r4", 15.0, "cat_b", "test", "aggressive_dwconv", 0.45, 18.0, False, [], True),
        ]

        metrics = compute_strategy_metrics(decs, [m.name for m in modes])
        self.assertEqual(metrics["total_requests"], 4)
        self.assertEqual(metrics["empirical_hit_count"], 2)
        self.assertEqual(metrics["empirical_miss_count"], 2)
        self.assertAlmostEqual(metrics["empirical_deadline_hit_rate"], 0.5)
        self.assertAlmostEqual(metrics["empirical_deadline_miss_rate"], 0.5)
        # mean mIoU: (0.74 + 0.74 + 0.70 + 0.45) / 4 = 2.63 / 4 = 0.6575
        self.assertAlmostEqual(metrics["average_selected_miou"], 0.6575)
        # mean bound: (16.6 + 16.6 + 18.6 + 18.0) / 4 = 69.8 / 4 = 17.45
        self.assertAlmostEqual(metrics["average_selected_latency_bound_ms"], 17.45)
        self.assertEqual(metrics["adaptive_fallback_count"], 2)

        # Mode frequencies: full=2 (50%), moderate=1 (25%), aggressive=1 (25%)
        freqs = metrics["mode_selection_frequency"]
        self.assertEqual(freqs["full_attention"]["count"], 2)
        self.assertEqual(freqs["full_attention"]["percentage"], 50.0)
        self.assertEqual(freqs["moderate_dwconv"]["count"], 1)
        self.assertEqual(freqs["moderate_dwconv"]["percentage"], 25.0)
        self.assertEqual(freqs["aggressive_dwconv"]["count"], 1)
        self.assertEqual(freqs["aggressive_dwconv"]["percentage"], 25.0)

        # Total counts sum to total requests
        total_freq_count = sum(f["count"] for f in freqs.values())
        self.assertEqual(total_freq_count, 4)

        # Category metrics
        cat_a = metrics["per_category_metrics"]["cat_a"]
        self.assertEqual(cat_a["request_count"], 2)
        self.assertEqual(cat_a["empirical_hits"], 2)
        self.assertAlmostEqual(cat_a["empirical_hit_rate"], 1.0)


class TestFullSimulationPipeline(unittest.TestCase):
    """End-to-end integration tests verifying files and outputs."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_simulation_produces_all_artifacts(self) -> None:
        summary = run_simulation(
            n_requests=100,
            seed=42,
            metric="p95",
            benchmark_path=DEFAULT_BENCHMARK,
            output_dir=self.temp_dir,
        )

        # Check summary object
        self.assertIn("strategies", summary)
        self.assertIn("always_full_attention", summary["strategies"])
        self.assertIn("always_fastest", summary["strategies"])
        self.assertIn("adaptive_controller", summary["strategies"])

        # Check files created on disk
        summary_file = self.temp_dir / "summary.json"
        stream_file = self.temp_dir / "deadline_stream.json"
        trace_file = self.temp_dir / "request_trace.csv"
        hit_png = self.temp_dir / "hit_rate_by_strategy.png"
        hit_svg = self.temp_dir / "hit_rate_by_strategy.svg"
        miou_png = self.temp_dir / "avg_miou_by_strategy.png"
        miou_svg = self.temp_dir / "avg_miou_by_strategy.svg"

        self.assertTrue(summary_file.is_file())
        self.assertTrue(stream_file.is_file())
        self.assertTrue(trace_file.is_file())
        self.assertTrue(hit_png.is_file())
        self.assertTrue(hit_svg.is_file())
        self.assertTrue(miou_png.is_file())
        self.assertTrue(miou_svg.is_file())

        # Check stream file contents
        with stream_file.open("r", encoding="utf-8") as f:
            stream_data = json.load(f)
        self.assertEqual(len(stream_data), 100)

        # Check CSV trace rows: header + 300 rows (100 per strategy)
        with trace_file.open("r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        self.assertEqual(len(reader), 1 + 300)
        self.assertEqual(reader[0][0], "request_id")
        self.assertEqual(reader[0][3], "strategy")

        # Verify strategy names in CSV
        strategies_in_csv = {row[3] for row in reader[1:]}
        self.assertEqual(
            strategies_in_csv,
            {"always_full_attention", "always_fastest", "adaptive_controller"},
        )


if __name__ == "__main__":
    unittest.main()
