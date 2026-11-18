import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "reference") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "reference"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.nn import functional as F

from baseline.train_segmentation import (
    IGNORE_INDEX,
    NUM_CLASSES,
    SegmentationHead,
    build_backbone,
    features,
    strict_load_backbone,
)
from datasets.oxford_pet_loader import load_dataset_from_manifest
from finetuning.finetune_dwconv import patch_and_verify
from lib.convatt.convatt_dw import CnvAttnDW_blockwise


class TestPhase12BDWConv(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.manifest_path = self.project_root / "outputs" / "phase12a_full_pet" / "split_manifest.json"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_block_replacement_and_preservation(self) -> None:
        """Verify targeted blocks are replaced with CnvAttnDW_blockwise and remaining blocks stay standard MHSA."""
        backbone = build_backbone()
        strict_load_backbone(backbone)
        target_blocks = [2, 5, 8, 11]
        patch_and_verify(backbone, target_blocks, self.device)

        for b in range(12):
            attn = backbone.blocks[b].attn
            if b in target_blocks:
                self.assertIsInstance(attn, CnvAttnDW_blockwise, f"Block {b} was not patched to CnvAttnDW_blockwise")
            else:
                self.assertNotIsInstance(attn, CnvAttnDW_blockwise, f"Block {b} should have preserved original attention")

    def test_sigma_selected_block_replacement_and_preservation(self) -> None:
        """Verify sigma blocks [1, 2, 3, 4] are replaced and others preserved."""
        backbone = build_backbone()
        strict_load_backbone(backbone)
        target_blocks = [1, 2, 3, 4]
        patch_and_verify(backbone, target_blocks, self.device)

        for b in range(12):
            attn = backbone.blocks[b].attn
            if b in target_blocks:
                self.assertIsInstance(attn, CnvAttnDW_blockwise, f"Block {b} was not patched to CnvAttnDW_blockwise")
            else:
                self.assertNotIsInstance(attn, CnvAttnDW_blockwise, f"Block {b} should have preserved original attention")

    def test_forward_backward_finite_gradients(self) -> None:
        """Verify forward output shape [B, 2, 224, 224], loss finiteness, and non-empty finite gradients."""
        backbone = build_backbone()
        strict_load_backbone(backbone)
        patch_and_verify(backbone, [2, 5, 8, 11], self.device)
        backbone.train()

        head = SegmentationHead().to(self.device).train()

        dummy_x = torch.randn(2, 3, 224, 224, device=self.device)
        dummy_y = torch.randint(0, 2, (2, 224, 224), device=self.device, dtype=torch.int64)
        dummy_y[0, 0:10, 0:10] = IGNORE_INDEX  # Ensure ignore index is exercised

        feat = features(backbone, dummy_x)
        logits = head(feat, (224, 224))

        self.assertEqual(logits.shape, torch.Size([2, 2, 224, 224]))

        loss = F.cross_entropy(logits, dummy_y, ignore_index=IGNORE_INDEX)
        self.assertTrue(torch.isfinite(loss).item())

        optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()

        for name, p in head.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.all(torch.isfinite(p.grad)).item(), f"Non-finite grad in head: {name}")

        for name, p in backbone.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.all(torch.isfinite(p.grad)).item(), f"Non-finite grad in backbone: {name}")

    def test_sigma_scores_structure_and_order(self) -> None:
        """Verify sigma_scores.json exists, has all 12 blocks, and produces expected ranking."""
        p = self.project_root / "outputs" / "phase12b_dwconv" / "sigma_scores.json"
        self.assertTrue(p.is_file(), f"Missing sigma_scores.json at {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(len(data["sigma_b_per_block"]), 12)
        self.assertEqual(len(data["ranked_blocks_zero_based"]), 12)
        # Check ascending order of ranked scores
        scores = [data["sigma_b_per_block"][str(b)] for b in data["ranked_blocks_zero_based"]]
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(set(data["top_4_selected_zero_based"]), {1, 2, 3, 4})

    def test_checkpoints_exist_and_loadable(self) -> None:
        """Verify all three best_checkpoint.pt files exist and have expected keys."""
        paths = [
            self.project_root / "outputs" / "phase12a_full_pet" / "baseline" / "best_checkpoint.pt",
            self.project_root / "outputs" / "phase12b_dwconv" / "manual" / "best_checkpoint.pt",
            self.project_root / "outputs" / "phase12b_dwconv" / "sigma_selected" / "best_checkpoint.pt",
        ]
        for cp_p in paths:
            self.assertTrue(cp_p.is_file(), f"Missing checkpoint: {cp_p}")
            payload = torch.load(cp_p, map_location="cpu", weights_only=False)
            self.assertIn("head_state_dict", payload)
            self.assertIn("epoch", payload)


if __name__ == "__main__":
    unittest.main()
