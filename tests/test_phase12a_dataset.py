"""Unit tests for Phase 12A: Dataset loading, mask mapping, ignore-index semantics, and split disjointness."""
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets.oxford_pet_loader import (
    IGNORE_INDEX,
    map_trimap_to_binary,
    load_dataset_from_manifest,
)


class TestPhase12AFullPetDataset(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.manifest_path = self.project_root / "outputs" / "phase12a_full_pet" / "split_manifest.json"

    def test_split_manifest_exists_and_valid(self):
        self.assertTrue(self.manifest_path.is_file(), f"Missing split manifest at {self.manifest_path}")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["train_count"], 3680)
        self.assertEqual(manifest["val_count"], 3669)
        self.assertEqual(manifest["total_samples"], 7349)
        self.assertEqual(manifest["num_categories"], 37)

    def test_split_disjointness(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        train_stems = {s["stem"] for s in manifest["train_samples"]}
        val_stems = {s["stem"] for s in manifest["val_samples"]}
        overlap = train_stems & val_stems
        self.assertEqual(len(overlap), 0, f"Detected data leakage between train and val: {overlap}")
        self.assertEqual(len(train_stems), 3680)
        self.assertEqual(len(val_stems), 3669)

    def test_trimap_mapping_semantics(self):
        # Oxford-IIIT Pet trimap: 1=pet, 2=boundary, 3=background
        raw = np.array([
            [1, 2, 3],
            [3, 1, 2],
        ], dtype=np.uint8)

        mapped = map_trimap_to_binary(raw)
        expected = np.array([
            [1, IGNORE_INDEX, 0],
            [0, 1, IGNORE_INDEX],
        ], dtype=np.int64)

        np.testing.assert_array_equal(mapped, expected)
        # Check that boundary (2) NEVER becomes background (0) or foreground (1)
        self.assertTrue(np.all(mapped[raw == 2] == IGNORE_INDEX))
        self.assertTrue(np.all(mapped[raw == 1] == 1))
        self.assertTrue(np.all(mapped[raw == 3] == 0))

    def test_dataset_item_shapes_and_types(self):
        train_ds, val_ds, manifest = load_dataset_from_manifest(
            self.manifest_path, self.project_root, image_size=224
        )
        self.assertEqual(len(train_ds), 3680)
        self.assertEqual(len(val_ds), 3669)

        # Inspect first sample of train
        img, mask = train_ds[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(img.shape, torch.Size([3, 224, 224]))
        self.assertEqual(mask.shape, torch.Size([224, 224]))
        self.assertEqual(img.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.int64)

        # Check unique mask values
        unique_vals = set(torch.unique(mask).tolist())
        self.assertTrue(unique_vals.issubset({0, 1, IGNORE_INDEX}))

        # Inspect first sample of val
        img_val, mask_val = val_ds[0]
        self.assertEqual(img_val.shape, torch.Size([3, 224, 224]))
        self.assertEqual(mask_val.shape, torch.Size([224, 224]))
        unique_vals_val = set(torch.unique(mask_val).tolist())
        self.assertTrue(unique_vals_val.issubset({0, 1, IGNORE_INDEX}))


if __name__ == "__main__":
    unittest.main()
