"""Unit tests for Phase 9.2 Statistical Validation and Analysis."""

import unittest
from pathlib import Path

from realtime.paired_statistical_analysis import (
    calculate_mcnemar_test,
    paired_difference_ci,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_FILE = PROJECT_ROOT / "outputs" / "phase9_1_fair" / "raw_traces.csv"


class TestPhase92StatisticalAnalysis(unittest.TestCase):
    """Test paired statistical calculations and validation."""

    def test_mcnemar_exact_symmetry(self) -> None:
        # Equal discordant pairs should give chi2 = 0 and p = 1.0
        res = calculate_mcnemar_test(10, 10)
        self.assertEqual(res["chi2_continuity_corrected"], 0.0)
        self.assertEqual(res["p_value_continuity_corrected"], 1.0)
        self.assertFalse(res["statistically_significant_05"])

    def test_mcnemar_known_values(self) -> None:
        # b = 25, c = 5 -> total = 30
        # (|25 - 5| - 1)^2 / 30 = 19^2 / 30 = 361 / 30 = 12.0333
        res = calculate_mcnemar_test(25, 5)
        self.assertAlmostEqual(res["chi2_continuity_corrected"], 12.0333, places=3)
        self.assertLess(res["p_value_continuity_corrected"], 0.001)
        self.assertTrue(res["statistically_significant_05"])

    def test_paired_difference_ci(self) -> None:
        diffs = [1.0, 1.0, 1.0, 1.0, 1.0]
        ci = paired_difference_ci(diffs)
        self.assertEqual(ci["mean"], 1.0)
        self.assertEqual(ci["std"], 0.0)
        self.assertEqual(ci["ci_lower"], 1.0)
        self.assertEqual(ci["ci_upper"], 1.0)


if __name__ == "__main__":
    unittest.main()
