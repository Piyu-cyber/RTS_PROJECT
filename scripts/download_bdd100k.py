"""BDD100K Dataset Downloader & Annotation Formatter.

Downloads official BDD100K images and annotations, formats them for
Driving Scene Category Classification, and sets up reproducible splits.

Usage:
    python scripts/download_bdd100k.py [--num-images 600] [--data-dir data/bdd100k]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
from typing import Dict, List, Any, Tuple
import requests
from PIL import Image

BDD_SCENE_CLASSES = [
    "city street",
    "highway",
    "residential",
    "parking lot",
    "gas stations",
    "tunnel",
]

SAMPLES_JSON_URL = "https://huggingface.co/datasets/dgural/bdd100k/resolve/main/samples.json"
IMAGE_BASE_URL = "https://huggingface.co/datasets/dgural/bdd100k/resolve/main/data"


def download_file(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> bool:
    """Download a file with progress and error handling."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)
        return True
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return False


def fetch_and_parse_annotations(data_dir: str) -> List[Dict[str, Any]]:
    """Fetch BDD100K sample records and extract scene classification metadata."""
    annot_dir = os.path.join(data_dir, "annotations")
    os.makedirs(annot_dir, exist_ok=True)
    cache_path = os.path.join(annot_dir, "bdd100k_raw_samples.json")

    if not os.path.isfile(cache_path):
        print(f"Downloading BDD100K annotations index (~71 MB) from {SAMPLES_JSON_URL}...")
        success = download_file(SAMPLES_JSON_URL, cache_path)
        if not success:
            raise RuntimeError("Failed to download BDD100K annotations index.")
        print(f"Saved raw annotations index to {cache_path}")
    else:
        print(f"Using cached raw annotations index from {cache_path}")

    print("Parsing BDD100K samples and extracting driving scene metadata...")
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_samples = data.get("samples", [])
    valid_samples = []

    for s in raw_samples:
        filepath = s.get("filepath", "")
        filename = os.path.basename(filepath)
        scene_info = s.get("scene", {})
        scene_label = scene_info.get("label") if isinstance(scene_info, dict) else None

        if not filename or not scene_label:
            continue

        if scene_label in BDD_SCENE_CLASSES:
            class_id = BDD_SCENE_CLASSES.index(scene_label)
            valid_samples.append({
                "image_name": os.path.splitext(filename)[0],
                "filename": filename,
                "scene": scene_label,
                "class_id": class_id,
                "weather": s.get("weather", {}).get("label") if isinstance(s.get("weather"), dict) else None,
                "timeofday": s.get("timeofday", {}).get("label") if isinstance(s.get("timeofday"), dict) else None,
            })

    print(f"Extracted {len(valid_samples)} valid driving scene samples across {len(BDD_SCENE_CLASSES)} classes.")
    return valid_samples


def download_single_image(args: Tuple[str, str, str]) -> Tuple[str, bool]:
    """Worker function to download and verify an image."""
    filename, image_dir, image_url = args
    dest_path = os.path.join(image_dir, filename)

    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1000:
        # Check integrity
        try:
            with Image.open(dest_path) as img:
                img.verify()
            return filename, True
        except Exception:
            pass

    success = download_file(image_url, dest_path)
    if success:
        try:
            with Image.open(dest_path) as img:
                img.verify()
            return filename, True
        except Exception:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return filename, False
    return filename, False


def setup_bdd100k(
    data_dir: str = "data/bdd100k",
    num_images: int = 600,
    seed: int = 42,
    max_workers: int = 8,
) -> Dict[str, Any]:
    """Download balanced subset of BDD100K images and generate reproducible splits."""
    random.seed(seed)
    images_dir = os.path.join(data_dir, "images")
    annot_dir = os.path.join(data_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annot_dir, exist_ok=True)

    # 1. Fetch metadata
    all_samples = fetch_and_parse_annotations(data_dir)

    # Group by class
    by_class: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(BDD_SCENE_CLASSES))}
    for s in all_samples:
        by_class[s["class_id"]].append(s)

    for c_id, name in enumerate(BDD_SCENE_CLASSES):
        print(f"  Class {c_id} ({name}): {len(by_class[c_id])} available frames")

    # Select balanced target count per class
    target_per_class = num_images // len(BDD_SCENE_CLASSES)
    selected_samples: List[Dict[str, Any]] = []

    for c_id in range(len(BDD_SCENE_CLASSES)):
        available = by_class[c_id]
        random.shuffle(available)
        chosen = available[:target_per_class]
        selected_samples.extend(chosen)

    print(f"\nTargeting {len(selected_samples)} total images ({target_per_class} per class)...")

    # 2. Parallel image downloads
    download_tasks = [
        (s["filename"], images_dir, f"{IMAGE_BASE_URL}/{s['filename']}")
        for s in selected_samples
    ]

    print(f"Downloading {len(download_tasks)} images into {images_dir} using {max_workers} threads...")
    t0 = time.time()
    successful_filenames = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for filename, ok in executor.map(download_single_image, download_tasks):
            if ok:
                successful_filenames.add(filename)

    elapsed = time.time() - t0
    print(f"Downloaded and verified {len(successful_filenames)} / {len(download_tasks)} images in {elapsed:.1f}s.")

    # Filter to only downloaded and verified samples
    final_samples = [s for s in selected_samples if s["filename"] in successful_filenames]

    # 3. Create reproducible train / val / test splits (70% train, 15% val, 15% test)
    # Ensure stratification across classes
    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []

    final_by_class: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(BDD_SCENE_CLASSES))}
    for s in final_samples:
        final_by_class[s["class_id"]].append(s)

    for c_id, class_items in final_by_class.items():
        random.shuffle(class_items)
        n = len(class_items)
        n_val = max(1, int(0.15 * n))
        n_test = max(1, int(0.15 * n))
        n_train = n - n_val - n_test

        train_samples.extend(class_items[:n_train])
        val_samples.extend(class_items[n_train:n_train + n_val])
        test_samples.extend(class_items[n_train + n_val:])

    # Shuffle final splits
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    # 4. Save annotation files
    def save_json(obj: Any, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    save_json(BDD_SCENE_CLASSES, os.path.join(annot_dir, "class_names.json"))
    save_json(train_samples, os.path.join(annot_dir, "train.json"))
    save_json(val_samples, os.path.join(annot_dir, "val.json"))
    save_json(test_samples, os.path.join(annot_dir, "test.json"))

    # Also save standard txt format for compatibility with evaluation routines
    for split_name, split_list in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        txt_path = os.path.join(annot_dir, f"{split_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for s in split_list:
                f.write(f"{s['image_name']} {s['class_id']}\n")

    summary = {
        "dataset": "BDD100K",
        "task": "driving_scene_classification",
        "data_dir": data_dir,
        "classes": BDD_SCENE_CLASSES,
        "num_classes": len(BDD_SCENE_CLASSES),
        "total_images": len(final_samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
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

    print("\n--- BDD100K Setup Complete ---")
    print(f"Total verified images: {summary['total_images']}")
    print(f"Train: {summary['train_count']} | Val: {summary['val_count']} | Test: {summary['test_count']}")
    print(f"Classes: {BDD_SCENE_CLASSES}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare BDD100K driving scene dataset.")
    parser.add_argument("--data-dir", type=str, default="data/bdd100k", help="Destination root directory")
    parser.add_argument("--num-images", type=int, default=600, help="Target total images across all classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--threads", type=int, default=8, help="Parallel download threads")
    args = parser.parse_args()

    setup_bdd100k(
        data_dir=args.data_dir,
        num_images=args.num_images,
        seed=args.seed,
        max_workers=args.threads,
    )
