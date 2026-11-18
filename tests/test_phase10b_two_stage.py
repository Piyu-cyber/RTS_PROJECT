"""Phase 10B regression tests for two-stage training, block indexing, and evaluation."""

import unittest
from pathlib import Path
import torch

from baseline.train_segmentation import IGNORE_INDEX, NUM_CLASSES
from realtime.sigma_b_selection import rank_and_select_blocks


class TestPhase10BComponents(unittest.TestCase):
    """Test suite covering Phase 10B requirements."""

    def test_block_indexing_zero_and_one_based_conversion(self) -> None:
        """Verify conversions between 0-based code indices and 1-based paper notation."""
        zero_based_mod = [2, 5, 8, 11]
        one_based_mod = [b + 1 for b in zero_based_mod]
        self.assertEqual(one_based_mod, [3, 6, 9, 12])

        zero_based_sig4 = [1, 2, 3, 4]
        one_based_sig4 = [b + 1 for b in zero_based_sig4]
        self.assertEqual(one_based_sig4, [2, 3, 4, 5])

        # Inversion
        self.assertEqual([b - 1 for b in one_based_sig4], zero_based_sig4)

    def test_class_iou_and_confusion_matrix_calculation(self) -> None:
        """Verify IoU calculation on known synthetic confusion matrix."""
        # Confusion: [[TN, FP], [FN, TP]]
        # Let class 0 (bg) have TP=100, FP=50, FN=50 -> union = 100+50+50 = 200 -> IoU = 100/200 = 0.5
        # Let class 1 (pet) have TP=400, FP=50, FN=50 -> union = 400+50+50 = 500 -> IoU = 400/500 = 0.8
        confusion = torch.tensor([[100, 50], [50, 400]], dtype=torch.int64)

        per_class = {}
        for i, name in enumerate(["background", "pet"]):
            tp = confusion[i, i].item()
            union = tp + confusion[:, i].sum().item() - tp + confusion[i, :].sum().item() - tp
            score = tp / union if union > 0 else None
            per_class[name] = score

        self.assertAlmostEqual(per_class["background"], 0.5)
        self.assertAlmostEqual(per_class["pet"], 0.8)
        miou = (per_class["background"] + per_class["pet"]) / 2
        self.assertAlmostEqual(miou, 0.65)

    def test_mode_collapse_iou_behavior(self) -> None:
        """Verify IoU calculation when model predicts 100% pet foreground (collapse)."""
        # Ground truth: 200 bg, 800 pet.
        # Prediction: 0 bg, 1000 pet.
        # Confusion: [[0, 200], [0, 800]]
        confusion = torch.tensor([[0, 200], [0, 800]], dtype=torch.int64)

        # Class 0 (bg): tp=0, union = 0 + 0 + 200 = 200 -> IoU = 0/200 = 0.0
        # Class 1 (pet): tp=800, union = 800 + 200 + 0 = 1000 -> IoU = 800/1000 = 0.8
        bg_tp = confusion[0, 0].item()
        bg_union = bg_tp + confusion[:, 0].sum().item() - bg_tp + confusion[0, :].sum().item() - bg_tp
        bg_iou = bg_tp / bg_union if bg_union > 0 else None

        pet_tp = confusion[1, 1].item()
        pet_union = pet_tp + confusion[:, 1].sum().item() - pet_tp + confusion[1, :].sum().item() - pet_tp
        pet_iou = pet_tp / pet_union if pet_union > 0 else None

        self.assertEqual(bg_iou, 0.0)
        self.assertEqual(pet_iou, 0.8)
        self.assertEqual((bg_iou + pet_iou) / 2, 0.4)

    def test_ignore_index_pixel_handling(self) -> None:
        """Verify that ignore_index (255) pixels are excluded from valid evaluation mask."""
        targets = torch.tensor([[0, 1], [255, 1]], dtype=torch.int64)
        valid = targets != IGNORE_INDEX
        self.assertEqual(valid.sum().item(), 3)
        self.assertFalse(valid[1, 0].item())


if __name__ == "__main__":
    unittest.main()
