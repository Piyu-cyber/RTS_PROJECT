"""High-performance parser for all 100,000 official BDD100K JSON annotations."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple

BDD_SCENE_CLASSES = [
    "city street",
    "highway",
    "residential",
    "parking lot",
    "gas stations",
    "tunnel",
]

scene_pattern = re.compile(r'"scene":\s*"([^"]+)"')
weather_pattern = re.compile(r'"weather":\s*"([^"]+)"')
tod_pattern = re.compile(r'"timeofday":\s*"([^"]+)"')


def parse_single_file(args: Tuple[str, str, str]) -> Dict[str, Any]:
    folder, f, split_name = args
    path = os.path.join(folder, f)
    image_name = os.path.splitext(f)[0]
    try:
        with open(path, "rb") as fp:
            fp.seek(0, 2)
            sz = fp.tell()
            fp.seek(max(0, sz - 1024), 0)
            tail = fp.read().decode("utf-8", errors="ignore")
        
        s = scene_pattern.search(tail)
        w = weather_pattern.search(tail)
        t = tod_pattern.search(tail)

        scene = s.group(1) if s else None
        weather = w.group(1) if w else None
        timeofday = t.group(1) if t else None

        valid_class_id = BDD_SCENE_CLASSES.index(scene) if (scene and scene in BDD_SCENE_CLASSES) else None

        return {
            "image_name": image_name,
            "filename": f"{image_name}.jpg",
            "split": split_name,
            "scene": scene,
            "class_id": valid_class_id,
            "weather": weather,
            "timeofday": timeofday,
        }
    except Exception as e:
        return {
            "image_name": image_name,
            "filename": f"{image_name}.jpg",
            "split": split_name,
            "scene": None,
            "class_id": None,
            "weather": None,
            "timeofday": None,
            "error": str(e),
        }


def parse_split(folder: str, split_name: str, max_workers: int = 16) -> Dict[str, Any]:
    print(f"\nScanning split '{split_name}' in {folder}...", flush=True)
    t0 = time.time()
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    print(f"  Found {len(files)} JSON files. Parsing with {max_workers} threads...", flush=True)

    tasks = [(folder, f, split_name) for f in files]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(parse_single_file, tasks))

    elapsed = time.time() - t0
    scenes = Counter()
    weathers = Counter()
    times = Counter()
    valid_records = []
    undefined_count = 0
    missing_count = 0

    for r in results:
        scene = r.get("scene")
        weather = r.get("weather")
        tod = r.get("timeofday")

        if scene:
            scenes[scene] += 1
            if r.get("class_id") is not None:
                valid_records.append(r)
            else:
                undefined_count += 1
        else:
            missing_count += 1

        if weather:
            weathers[weather] += 1
        if tod:
            times[tod] += 1

    print(f"  Parsed {len(files)} files in {elapsed:.2f} s ({len(files)/elapsed:.0f} files/s).", flush=True)
    print(f"  Valid 6-class driving scenes: {len(valid_records)} / {len(files)} ({len(valid_records)/len(files)*100:.2f}%)", flush=True)
    print(f"  Undefined scenes: {undefined_count} | Missing scenes: {missing_count}", flush=True)
    print(f"  Scene distribution: {dict(scenes)}", flush=True)

    return {
        "split": split_name,
        "total_files": len(files),
        "valid_count": len(valid_records),
        "undefined_count": undefined_count,
        "missing_count": missing_count,
        "scene_distribution": dict(scenes),
        "weather_distribution": dict(weathers),
        "timeofday_distribution": dict(times),
        "records": valid_records,
    }


def main():
    root = "data/bdd100k/annotations/official_labels/100k"
    train_res = parse_split(os.path.join(root, "train"), "train")
    val_res = parse_split(os.path.join(root, "val"), "val")
    test_res = parse_split(os.path.join(root, "test"), "test")

    summary = {
        "dataset": "BDD100K",
        "task": "driving_scene_category_classification",
        "official_source": "http://128.32.162.150/bdd100k/bdd100k_labels.zip (UC Berkeley)",
        "classes": BDD_SCENE_CLASSES,
        "splits": {
            "train": {
                "total_files": train_res["total_files"],
                "valid_6class_scenes": train_res["valid_count"],
                "undefined_scenes": train_res["undefined_count"],
                "missing_scenes": train_res["missing_count"],
                "scenes": train_res["scene_distribution"],
                "weather": train_res["weather_distribution"],
                "timeofday": train_res["timeofday_distribution"],
            },
            "val": {
                "total_files": val_res["total_files"],
                "valid_6class_scenes": val_res["valid_count"],
                "undefined_scenes": val_res["undefined_count"],
                "missing_scenes": val_res["missing_count"],
                "scenes": val_res["scene_distribution"],
                "weather": val_res["weather_distribution"],
                "timeofday": val_res["timeofday_distribution"],
            },
            "test": {
                "total_files": test_res["total_files"],
                "valid_6class_scenes": test_res["valid_count"],
                "undefined_scenes": test_res["undefined_count"],
                "missing_scenes": test_res["missing_count"],
                "scenes": test_res["scene_distribution"],
                "weather": test_res["weather_distribution"],
                "timeofday": test_res["timeofday_distribution"],
            },
        },
        "totals": {
            "total_annotations": train_res["total_files"] + val_res["total_files"] + test_res["total_files"],
            "total_valid_6class": train_res["valid_count"] + val_res["valid_count"] + test_res["valid_count"],
            "total_undefined": train_res["undefined_count"] + val_res["undefined_count"] + test_res["undefined_count"],
            "total_missing": train_res["missing_count"] + val_res["missing_count"] + test_res["missing_count"],
        }
    }

    out_summary = "data/bdd100k/annotations/full_100k_annotation_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Saved full summary to {out_summary}")

    # Save full dataset manifest
    all_valid_records = train_res["records"] + val_res["records"] + test_res["records"]
    out_manifest = "outputs/bdd100k_migration/full_dataset_manifest.json"
    os.makedirs(os.path.dirname(out_manifest), exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump({
            "manifest_version": "1.0",
            "dataset": "BDD100K",
            "task": "driving_scene_classification",
            "classes": BDD_SCENE_CLASSES,
            "total_valid_records": len(all_valid_records),
            "summary": summary,
            "samples": all_valid_records,
        }, f, indent=2)
    print(f"[OK] Saved full dataset manifest ({len(all_valid_records)} samples) to {out_manifest}")

    # Save train_full, val_full, test_full json files
    out_dir = "data/bdd100k/annotations"
    for split_name, recs in [("train_full", train_res["records"]), ("val_full", val_res["records"]), ("test_full", test_res["records"])]:
        p = os.path.join(out_dir, f"{split_name}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2)
        print(f"[OK] Saved {len(recs)} records to {p}")


if __name__ == "__main__":
    main()
