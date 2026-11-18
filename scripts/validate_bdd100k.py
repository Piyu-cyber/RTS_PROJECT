"""Validation script for BDD100K dataset assets.

Verifies:
1. Existence of required annotation files and class list.
2. Integrity of all referenced images via PIL header checks.
3. Class balance and absence of empty or corrupted files.
4. Non-leakage: disjoint train, val, and test splits.

Usage:
    python scripts/validate_bdd100k.py [--data-dir data/bdd100k]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Set, Any
from PIL import Image


def validate_bdd100k(data_dir: str = "data/bdd100k") -> bool:
    print(f"=== Validating BDD100K Dataset in '{data_dir}' ===")
    annot_dir = os.path.join(data_dir, "annotations")
    images_dir = os.path.join(data_dir, "images")

    if not os.path.isdir(annot_dir):
        print(f"FAILED: Annotations directory missing at {annot_dir}", file=sys.stderr)
        return False
    if not os.path.isdir(images_dir):
        print(f"FAILED: Images directory missing at {images_dir}", file=sys.stderr)
        return False

    # Check class names
    class_names_path = os.path.join(annot_dir, "class_names.json")
    if not os.path.isfile(class_names_path):
        print(f"FAILED: class_names.json missing at {class_names_path}", file=sys.stderr)
        return False

    with open(class_names_path, "r", encoding="utf-8") as f:
        classes = json.load(f)
    print(f"Class names ({len(classes)}): {classes}")

    splits = ["train", "val", "test"]
    split_sets: Dict[str, Set[str]] = {}
    total_images_checked = 0
    corrupt_images = 0

    for split in splits:
        json_path = os.path.join(annot_dir, f"{split}.json")
        txt_path = os.path.join(annot_dir, f"{split}.txt")

        if not os.path.isfile(json_path):
            print(f"FAILED: Missing split file {json_path}", file=sys.stderr)
            return False
        if not os.path.isfile(txt_path):
            print(f"FAILED: Missing split file {txt_path}", file=sys.stderr)
            return False

        with open(json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        print(f"\nChecking split '{split}' ({len(samples)} samples)...")
        image_names = set()
        class_counts = {c: 0 for c in classes}

        for s in samples:
            filename = s.get("filename")
            image_name = s.get("image_name")
            class_id = s.get("class_id")
            scene = s.get("scene")

            if not filename or not image_name or class_id is None or scene is None:
                print(f"FAILED: Incomplete record in {split}: {s}", file=sys.stderr)
                return False

            if class_id < 0 or class_id >= len(classes):
                print(f"FAILED: Invalid class_id {class_id} for {classes}", file=sys.stderr)
                return False

            if classes[class_id] != scene:
                print(f"FAILED: Mismatch between class_id {class_id} and scene '{scene}'", file=sys.stderr)
                return False

            image_path = os.path.join(images_dir, filename)
            if not os.path.isfile(image_path):
                for sub in ("train", "val", "test"):
                    cand = os.path.join(images_dir, "100k", sub, filename)
                    if os.path.isfile(cand):
                        image_path = cand
                        break
            if not os.path.isfile(image_path):
                print(f"FAILED: Referenced image does not exist: {image_path}", file=sys.stderr)
                return False

            # Verify image file
            try:
                with Image.open(image_path) as img:
                    img.verify()
                with Image.open(image_path) as img:
                    w, h = img.size
                    if w < 100 or h < 100:
                        print(f"WARNING: Suspiciously small image {filename}: {w}x{h}")
            except Exception as e:
                print(f"FAILED: Corrupted image file {image_path}: {e}", file=sys.stderr)
                corrupt_images += 1
                return False

            image_names.add(image_name)
            class_counts[scene] += 1
            total_images_checked += 1

        split_sets[split] = image_names
        print(f"  Distribution: {class_counts}")

    # Verify disjoint splits
    train_val_overlap = split_sets["train"].intersection(split_sets["val"])
    train_test_overlap = split_sets["train"].intersection(split_sets["test"])
    val_test_overlap = split_sets["val"].intersection(split_sets["test"])

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(f"FAILED: Data leakage detected across splits!", file=sys.stderr)
        print(f"  Train/Val overlap: {len(train_val_overlap)}")
        print(f"  Train/Test overlap: {len(train_test_overlap)}")
        print(f"  Val/Test overlap: {len(val_test_overlap)}")
        return False

    print("\n[OK] Split Disjointness: Verified (0 image overlap between train, val, test).")
    print(f"[OK] Image Integrity: Verified {total_images_checked} images (0 corrupted).")
    print("[OK] Label Consistency: Verified 100% correspondence between annotations and class definitions.")
    print("\nBDD100K DATASET VALIDATION: PASSED")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate BDD100K dataset assets.")
    parser.add_argument("--data-dir", type=str, default="data/bdd100k", help="Dataset root directory")
    args = parser.parse_args()

    success = validate_bdd100k(args.data_dir)
    sys.exit(0 if success else 1)
