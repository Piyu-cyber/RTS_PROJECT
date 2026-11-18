"""Unit tests for Phase 10A Sigma_h and Sigma_b calculation and block selection."""

import unittest
import numpy as np
import torch

from realtime.sigma_b_selection import (
    WelfordAccumulator,
    rank_and_select_blocks,
)


class TestSigmaBSelection(unittest.TestCase):
    """Test mathematical and algorithmic correctness of Sigma_b selection."""

    def test_welford_accumulator_against_direct_numpy(self) -> None:
        shape = (3, 3)
        acc = WelfordAccumulator(shape)
        np.random.seed(42)
        samples = [np.random.randn(*shape) for _ in range(20)]

        for s in samples:
            acc.update(s)

        stack = np.stack(samples, axis=0)
        direct_std_sample = np.std(stack, axis=0, ddof=1)
        welford_std = acc.std(ddof=1)

        self.assertEqual(acc.count, 20)
        np.testing.assert_allclose(welford_std, direct_std_sample, rtol=1e-7, atol=1e-7)

    def test_sigma_h_and_sigma_b_aggregation(self) -> None:
        # Mock 2 blocks, 2 heads per block, (2, 2) attention maps
        # Block 0:
        # Head 0 std: all 1.0 -> sum = 4.0
        # Head 1 std: all 2.0 -> sum = 8.0
        # -> Sigma_b[0] = (4.0 + 8.0) / 2 = 6.0
        # Block 1:
        # Head 0 std: all 0.5 -> sum = 2.0
        # Head 1 std: all 1.5 -> sum = 6.0
        # -> Sigma_b[1] = (2.0 + 6.0) / 2 = 4.0
        sigma_b = {0: 6.0, 1: 4.0}
        selected = rank_and_select_blocks(sigma_b, k=1)
        # Block 1 has lower Sigma_b (4.0 < 6.0), so it should be selected
        self.assertEqual(selected, [1])

    def test_ascending_ordering(self) -> None:
        sigma_b = {0: 10.5, 1: 3.2, 2: 7.8, 3: 1.1, 4: 5.4}
        # Ascending: 3 (1.1), 1 (3.2), 4 (5.4), 2 (7.8), 0 (10.5)
        top_2 = rank_and_select_blocks(sigma_b, k=2)
        self.assertEqual(top_2, [1, 3])  # sorted index order [1, 3]

        top_4 = rank_and_select_blocks(sigma_b, k=4)
        self.assertEqual(top_4, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
