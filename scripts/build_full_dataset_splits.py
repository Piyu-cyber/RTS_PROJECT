"""Build stratified Train / Val / Test splits from all verified BDD100K images on disk."""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Any
from PIL import Image

BDD_SCENE_CLASSES = [
    "city street",
    "highway",
    "residential",
    "parking lot",
    "gas stations",
    "tunnel",
]


def verify_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
            if w < 100 or h < 100:
                return False
        return True
    except Exception:
        return False


def build_splits(
    data_dir: str = "data/bdd100k",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Any]:
    random.seed(seed)
    images_dir = os.path.join(data_dir, "images")
    annot_dir = os.path.join(data_dir, "annotations")

    # Load 100K full manifest
    manifest_path = "outputs/bdd100k_migration/full_dataset_manifest.json"
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing {manifest_path}. Run scripts/parse_full_100k_annotations.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    samples_map = {s["image_name"]: s for s in manifest_data["samples"]}
    print(f"Loaded {len(samples_map)} valid annotations from full 100K manifest.")

    # Scan available images on disk
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Found {len(image_files)} image files on disk in {images_dir}.")

    verified_samples: List[Dict[str, Any]] = []
    unannotated_count = 0
    corrupt_count = 0

    for f in image_files:
        path = os.path.join(images_dir, f)
        img_id = os.path.splitext(f)[0]
        if img_id not in samples_map:
            unannotated_count += 1
            continue

        if not verify_image(path):
            corrupt_count += 1
            continue

        annot = samples_map[img_id]
        verified_samples.append({
            "image_name": img_id,
            "filename": f,
            "scene": annot["scene"],
            "class_id": annot["class_id"],
            "weather": annot.get("weather"),
            "timeofday": annot.get("timeofday"),
            "official_split": annot.get("split"),
        })

    print(f"\n--- Verified Local Inventory ---")
    print(f"Total eligible verified images on disk: {len(verified_samples)}")
    print(f"Unannotated / excluded frames: {unannotated_count}")
    print(f"Corrupt / unreadable frames: {corrupt_count}")

    # Group by class to ensure stratified splitting
    by_class: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(BDD_SCENE_CLASSES))}
    for s in verified_samples:
        by_class[s["class_id"]].append(s)

    train_samples = []
    val_samples = []
    test_samples = []

    print("\nClass breakdown of verified images on disk:")
    for c_id, name in enumerate(BDD_SCENE_CLASSES):
        items = by_class[c_id]
        random.shuffle(items)
        n = len(items)
        n_val = max(1, int(round(val_ratio * n))) if n >= 3 else 0
        n_test = max(1, int(round(test_ratio * n))) if n >= 3 else 0
        n_train = n - n_val - n_test
        if n_train < 1 and n > 0:
            n_train = 1
            if n_val > 0: n_val -= 1
            elif n_test > 0: n_test -= 1

        c_train = items[:n_train]
        c_val = items[n_train:n_train + n_val]
        c_test = items[n_train + n_val:]

        train_samples.extend(c_train)
        val_samples.extend(c_val)
        test_samples.extend(c_test)

        print(f"  Class {c_id} ({name:<12}): total={n:<4} | train={len(c_train):<4} | val={len(c_val):<3} | test={len(c_test):<3}")

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    # Save splits
    def save_json(obj: Any, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    save_json(BDD_SCENE_CLASSES, os.path.join(annot_dir, "class_names.json"))
    save_json(train_samples, os.path.join(annot_dir, "train.json"))
    save_json(val_samples, os.path.join(annot_dir, "val.json"))
    save_json(test_samples, os.path.join(annot_dir, "test.json"))

    for split_name, split_list in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        txt_path = os.path.join(annot_dir, f"{split_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for s in split_list:
                f.write(f"{s['image_name']} {s['class_id']}\n")

    summary = {
        "dataset": "BDD100K",
        "task": "driving_scene_classification",
        "classes": BDD_SCENE_CLASSES,
        "num_classes": len(BDD_SCENE_CLASSES),
        "total_verified_images": len(verified_samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "full_official_catalog_count": len(samples_map),
        "annotations_saved": [
            os.path.join(annot_dir, "class_names.json"),
            os.path.join(annot_dir, "train.json"),
            os.path.join(annot_dir, "val.json"),
            os.path.join(annot_dir, "test.json"),
            os.path.join(annot_dir, "train.txt"),
            os.path.join(annot_dir, "val.txt"),
            os.path.join(annot_dir, "test.txt"),
        ],
    }
    save_json(summary, os.path.join(annot_dir, "dataset_summary.json"))

    print("\n--- BDD100K Stratified Split Construction Complete ---")
    print(f"Total Local Verified Images: {summary['total_verified_images']}")
    print(f"Train: {summary['train_count']} | Val: {summary['val_count']} | Test: {summary['test_count']}")
    return summary


if __name__ == "__main__":
    build_splits()
