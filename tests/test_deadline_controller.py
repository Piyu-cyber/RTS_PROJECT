"""Unit tests for the deadline-aware inference-mode controller.

Covers all 22 required test cases:
  Selection behavior:  tests 1-8
  Input validation:    tests 9-20
  Benchmark JSON:      tests 21-22

Run from the project root::

    .\.venv\Scripts\python.exe -m unittest tests.test_deadline_controller -v
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from realtime.controller import (
    ControllerError,
    ModeProfile,
    SelectionResult,
    load_benchmark_modes,
    select_mode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mode(
    name: str,
    miou: float = 0.5,
    p95: float = 10.0,
    p99: float = 15.0,
    max_obs: float = 20.0,
) -> ModeProfile:
    """Shorthand for creating a ModeProfile."""
    return ModeProfile(
        name=name,
        validation_miou=miou,
        p95_ms=p95,
        p99_ms=p99,
        max_observed_ms=max_obs,
    )


# ===================================================================
# Test group 1: Selection behavior (tests 1–8)
# ===================================================================
class TestSelectionBehavior(unittest.TestCase):
    """Tests 1-8: deterministic selection algorithm."""

    def test_01_all_modes_feasible_selects_highest_miou(self) -> None:
        """When all modes fit the deadline, select the highest mIoU."""
        modes = [
            _mode("fast", miou=0.4, p95=5.0),
            _mode("medium", miou=0.7, p95=10.0),
            _mode("accurate", miou=0.9, p95=15.0),
        ]
        result = select_mode(20.0, modes, "p95")
        self.assertEqual(result.selected_mode, "accurate")
        self.assertAlmostEqual(result.selected_miou, 0.9)
        self.assertFalse(result.likely_deadline_miss)
        self.assertEqual(len(result.feasible_modes), 3)

    def test_02_exactly_one_mode_feasible(self) -> None:
        """When exactly one mode fits, select that mode."""
        modes = [
            _mode("fast", miou=0.4, p95=5.0),
            _mode("medium", miou=0.7, p95=10.0),
            _mode("accurate", miou=0.9, p95=15.0),
        ]
        result = select_mode(6.0, modes, "p95")
        self.assertEqual(result.selected_mode, "fast")
        self.assertAlmostEqual(result.selected_miou, 0.4)
        self.assertFalse(result.likely_deadline_miss)
        self.assertEqual(result.feasible_modes, ["fast"])

    def test_03_no_modes_feasible_selects_fastest_with_flag(self) -> None:
        """When no mode fits, fall back to lowest latency and flag miss."""
        modes = [
            _mode("fast", miou=0.4, p95=5.0),
            _mode("medium", miou=0.7, p95=10.0),
            _mode("accurate", miou=0.9, p95=15.0),
        ]
        result = select_mode(3.0, modes, "p95")
        self.assertEqual(result.selected_mode, "fast")
        self.assertTrue(result.likely_deadline_miss)
        self.assertEqual(result.feasible_modes, [])

    def test_04_deadline_exactly_equals_bound(self) -> None:
        """A mode whose bound equals the deadline is feasible."""
        modes = [
            _mode("a", miou=0.6, p95=10.0),
            _mode("b", miou=0.8, p95=20.0),
        ]
        # Deadline exactly equals b's p95
        result = select_mode(20.0, modes, "p95")
        self.assertIn("b", result.feasible_modes)
        self.assertIn("a", result.feasible_modes)
        self.assertEqual(result.selected_mode, "b")
        self.assertFalse(result.likely_deadline_miss)

        # Deadline exactly equals a's p95 — only a is feasible
        result2 = select_mode(10.0, modes, "p95")
        self.assertEqual(result2.feasible_modes, ["a"])
        self.assertEqual(result2.selected_mode, "a")
        self.assertFalse(result2.likely_deadline_miss)

    def test_05_equal_miou_selects_lower_latency(self) -> None:
        """With equal mIoU, choose the mode with lower latency bound."""
        modes = [
            _mode("slow", miou=0.7, p95=15.0),
            _mode("fast", miou=0.7, p95=8.0),
        ]
        result = select_mode(20.0, modes, "p95")
        self.assertEqual(result.selected_mode, "fast")

    def test_06_equal_latency_and_miou_uses_name_tiebreak(self) -> None:
        """With equal mIoU and latency, choose lexicographically smaller name."""
        modes = [
            _mode("charlie", miou=0.7, p95=10.0),
            _mode("alpha", miou=0.7, p95=10.0),
            _mode("bravo", miou=0.7, p95=10.0),
        ]
        result = select_mode(20.0, modes, "p95")
        self.assertEqual(result.selected_mode, "alpha")

    def test_07_fallback_tie_selects_higher_miou(self) -> None:
        """In fallback, among modes with equal lowest bound, choose higher mIoU."""
        modes = [
            _mode("a", miou=0.3, p95=10.0),
            _mode("b", miou=0.6, p95=10.0),
            _mode("c", miou=0.9, p95=10.0),
        ]
        result = select_mode(5.0, modes, "p95")
        self.assertTrue(result.likely_deadline_miss)
        self.assertEqual(result.selected_mode, "c")

    def test_08_reordering_modes_does_not_change_result(self) -> None:
        """The result must be the same regardless of input order."""
        modes_v1 = [
            _mode("full", miou=0.9, p95=15.0),
            _mode("moderate", miou=0.7, p95=12.0),
            _mode("aggressive", miou=0.4, p95=10.0),
        ]
        modes_v2 = list(reversed(modes_v1))
        modes_v3 = [modes_v1[1], modes_v1[2], modes_v1[0]]

        r1 = select_mode(13.0, modes_v1, "p95")
        r2 = select_mode(13.0, modes_v2, "p95")
        r3 = select_mode(13.0, modes_v3, "p95")

        self.assertEqual(r1.selected_mode, r2.selected_mode)
        self.assertEqual(r1.selected_mode, r3.selected_mode)
        self.assertEqual(r1.likely_deadline_miss, r2.likely_deadline_miss)
        self.assertEqual(r1.feasible_modes, r2.feasible_modes)
        self.assertEqual(r1.feasible_modes, r3.feasible_modes)


# ===================================================================
# Test group 2: Input validation (tests 9–20)
# ===================================================================
class TestInputValidation(unittest.TestCase):
    """Tests 9-20: informative validation for invalid inputs."""

    def _good_modes(self) -> list[ModeProfile]:
        return [_mode("a", miou=0.5, p95=10.0)]

    def test_09_negative_deadline(self) -> None:
        with self.assertRaises(ControllerError) as ctx:
            select_mode(-1.0, self._good_modes(), "p95")
        self.assertIn("non-negative", str(ctx.exception))

    def test_10_nan_deadline(self) -> None:
        with self.assertRaises(ControllerError) as ctx:
            select_mode(float("nan"), self._good_modes(), "p95")
        self.assertIn("NaN", str(ctx.exception))

    def test_11_infinite_deadline(self) -> None:
        with self.assertRaises(ControllerError) as ctx:
            select_mode(float("inf"), self._good_modes(), "p95")
        self.assertIn("finite", str(ctx.exception))

        with self.assertRaises(ControllerError):
            select_mode(float("-inf"), self._good_modes(), "p95")

    def test_12_empty_mode_collection(self) -> None:
        with self.assertRaises(ControllerError) as ctx:
            select_mode(20.0, [], "p95")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_13_missing_mode_name(self) -> None:
        """Empty-string name must be rejected."""
        mode = _mode("", miou=0.5, p95=10.0)
        with self.assertRaises(ControllerError) as ctx:
            select_mode(20.0, [mode], "p95")
        self.assertIn("non-empty", str(ctx.exception))

    def test_14_duplicate_mode_names(self) -> None:
        modes = [_mode("same", miou=0.5, p95=10.0), _mode("same", miou=0.6, p95=12.0)]
        with self.assertRaises(ControllerError) as ctx:
            select_mode(20.0, modes, "p95")
        self.assertIn("Duplicate", str(ctx.exception))

    def test_15_invalid_miou(self) -> None:
        """Non-numeric mIoU must be rejected."""
        mode = ModeProfile(
            name="bad", validation_miou="not_a_number",  # type: ignore[arg-type]
            p95_ms=10.0, p99_ms=15.0, max_observed_ms=20.0,
        )
        with self.assertRaises(ControllerError) as ctx:
            select_mode(20.0, [mode], "p95")
        self.assertIn("numeric", str(ctx.exception))

    def test_16_miou_out_of_range(self) -> None:
        """mIoU below 0 or above 1 must be rejected."""
        mode_neg = _mode("neg", miou=-0.1, p95=10.0)
        with self.assertRaises(ControllerError) as ctx:
            select_mode(20.0, [mode_neg], "p95")
        self.assertIn("[0, 1]", str(ctx.exception))

        mode_high = _mode("high", miou=1.5, p95=10.0)
        with self.assertRaises(ControllerError):
            select_mode(20.0, [mode_high], "p95")

    def test_17_negative_latency(self) -> None:
        mode = _mode("neg_lat", miou=0.5, p95=-1.0)
        with self.assertRaises(ControllerError) as ctx:
            select_mode(20.0, [mode], "p95")
        self.assertIn("non-negative", str(ctx.exception))

    def test_18_nan_or_infinite_latency(self) -> None:
        mode_nan = ModeProfile(
            name="nan_lat", validation_miou=0.5,
            p95_ms=float("nan"), p99_ms=15.0, max_observed_ms=20.0,
        )
        with self.assertRaises(ControllerError):
            select_mode(20.0, [mode_nan], "p95")

        mode_inf = ModeProfile(
            name="inf_lat", validation_miou=0.5,
            p95_ms=float("inf"), p99_ms=15.0, max_observed_ms=20.0,
        )
        with self.assertRaises(ControllerError):
            select_mode(20.0, [mode_inf], "p95")

    def test_19_missing_selected_latency_metric(self) -> None:
        """If the selected metric's value is None (missing), reject it."""
        mode = ModeProfile(
            name="missing", validation_miou=0.5,
            p95_ms=None,  # type: ignore[arg-type]
            p99_ms=15.0, max_observed_ms=20.0,
        )
        with self.assertRaises(ControllerError) as ctx:
            select_mode(20.0, [mode], "p95")
        self.assertIn("numeric", str(ctx.exception))

    def test_20_unsupported_metric_name(self) -> None:
        with self.assertRaises(ControllerError) as ctx:
            select_mode(20.0, self._good_modes(), "p50")
        self.assertIn("Unsupported", str(ctx.exception))

        with self.assertRaises(ControllerError):
            select_mode(20.0, self._good_modes(), "median")


# ===================================================================
# Test group 3: Benchmark JSON loader (tests 21–22)
# ===================================================================
class TestBenchmarkJsonLoader(unittest.TestCase):
    """Tests 21-22: JSON file loading and validation."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_json(self, data: object, filename: str = "bench.json") -> Path:
        path = self.temp_dir / filename
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _valid_benchmark(self) -> dict:
        """Return a minimal valid benchmark JSON structure."""
        return {
            "modes": {
                "mode_a": {
                    "accuracy": {"mean_iou": 0.8},
                    "latency_ms": {
                        "p95_ms": 12.0,
                        "p99_ms": 18.0,
                        "max_observed_ms": 25.0,
                    },
                },
            },
        }

    def test_21a_missing_benchmark_file(self) -> None:
        """Loading a non-existent file must fail."""
        with self.assertRaises(ControllerError) as ctx:
            load_benchmark_modes(self.temp_dir / "does_not_exist.json")
        self.assertIn("not found", str(ctx.exception))

    def test_21b_malformed_json(self) -> None:
        """Loading broken JSON must fail."""
        path = self.temp_dir / "bad.json"
        path.write_text("{invalid json!!!}", encoding="utf-8")
        with self.assertRaises(ControllerError) as ctx:
            load_benchmark_modes(path)
        self.assertIn("Malformed", str(ctx.exception))

    def test_21c_json_root_not_object(self) -> None:
        """JSON root must be an object."""
        path = self._write_json([1, 2, 3])
        with self.assertRaises(ControllerError) as ctx:
            load_benchmark_modes(path)
        self.assertIn("object", str(ctx.exception))

    def test_21d_empty_modes_object(self) -> None:
        """An empty 'modes' object must fail."""
        path = self._write_json({"modes": {}})
        with self.assertRaises(ControllerError):
            load_benchmark_modes(path)

    def test_21e_missing_modes_key(self) -> None:
        """Missing 'modes' key must fail."""
        path = self._write_json({"task": "test"})
        with self.assertRaises(ControllerError):
            load_benchmark_modes(path)

    def test_22a_missing_accuracy(self) -> None:
        """A mode without 'accuracy' must fail."""
        data = self._valid_benchmark()
        del data["modes"]["mode_a"]["accuracy"]
        path = self._write_json(data)
        with self.assertRaises(ControllerError) as ctx:
            load_benchmark_modes(path)
        self.assertIn("accuracy", str(ctx.exception))

    def test_22b_missing_mean_iou(self) -> None:
        """A mode without 'accuracy.mean_iou' must fail."""
        data = self._valid_benchmark()
        del data["modes"]["mode_a"]["accuracy"]["mean_iou"]
        path = self._write_json(data)
        with self.assertRaises(ControllerError) as ctx:
            load_benchmark_modes(path)
        self.assertIn("mean_iou", str(ctx.exception))

    def test_22c_missing_latency_ms(self) -> None:
        """A mode without 'latency_ms' must fail."""
        data = self._valid_benchmark()
        del data["modes"]["mode_a"]["latency_ms"]
        path = self._write_json(data)
        with self.assertRaises(ControllerError) as ctx:
            load_benchmark_modes(path)
        self.assertIn("latency_ms", str(ctx.exception))

    def test_22d_missing_individual_latency_field(self) -> None:
        """A mode missing one latency field (e.g. p99_ms) must fail."""
        data = self._valid_benchmark()
        del data["modes"]["mode_a"]["latency_ms"]["p99_ms"]
        path = self._write_json(data)
        with self.assertRaises(ControllerError) as ctx:
            load_benchmark_modes(path)
        self.assertIn("p99_ms", str(ctx.exception))

    def test_valid_benchmark_loads_correctly(self) -> None:
        """A valid benchmark file must load without errors."""
        path = self._write_json(self._valid_benchmark())
        modes = load_benchmark_modes(path)
        self.assertEqual(len(modes), 1)
        self.assertEqual(modes[0].name, "mode_a")
        self.assertAlmostEqual(modes[0].validation_miou, 0.8)
        self.assertAlmostEqual(modes[0].p95_ms, 12.0)
        self.assertAlmostEqual(modes[0].p99_ms, 18.0)
        self.assertAlmostEqual(modes[0].max_observed_ms, 25.0)


# ===================================================================
# Test group 4: Integration with real Phase 6B JSON
# ===================================================================
class TestPhase6BIntegration(unittest.TestCase):
    """Verify that the real Phase 6B JSON loads and produces valid results."""

    BENCHMARK_PATH = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "inference_benchmark"
        / "phase6_finetuned_results.json"
    )

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "outputs" / "inference_benchmark" / "phase6_finetuned_results.json").is_file(),
        "Phase 6B benchmark file not available",
    )
    def test_load_phase6b(self) -> None:
        modes = load_benchmark_modes(self.BENCHMARK_PATH)
        self.assertEqual(len(modes), 3)
        names = sorted(m.name for m in modes)
        self.assertEqual(names, ["aggressive_dwconv", "full_attention", "moderate_dwconv"])

        for m in modes:
            self.assertGreater(m.validation_miou, 0.0)
            self.assertLessEqual(m.validation_miou, 1.0)
            self.assertGreater(m.p95_ms, 0.0)
            self.assertGreater(m.p99_ms, 0.0)
            self.assertGreater(m.max_observed_ms, 0.0)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "outputs" / "inference_benchmark" / "phase6_finetuned_results.json").is_file(),
        "Phase 6B benchmark file not available",
    )
    def test_deadline_25_p95_all_feasible(self) -> None:
        """25 ms p95 — all modes feasible, select full_attention."""
        modes = load_benchmark_modes(self.BENCHMARK_PATH)
        result = select_mode(25.0, modes, "p95")
        self.assertEqual(result.selected_mode, "full_attention")
        self.assertEqual(len(result.feasible_modes), 3)
        self.assertFalse(result.likely_deadline_miss)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "outputs" / "inference_benchmark" / "phase6_finetuned_results.json").is_file(),
        "Phase 6B benchmark file not available",
    )
    def test_deadline_15_p95_none_feasible(self) -> None:
        """15 ms p95 — no mode feasible, fallback to full_attention (lowest p95)."""
        modes = load_benchmark_modes(self.BENCHMARK_PATH)
        result = select_mode(15.0, modes, "p95")
        self.assertEqual(result.selected_mode, "full_attention")
        self.assertTrue(result.likely_deadline_miss)
        self.assertEqual(result.feasible_modes, [])


if __name__ == "__main__":
    unittest.main()
