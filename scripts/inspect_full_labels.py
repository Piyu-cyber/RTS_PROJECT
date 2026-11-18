"""Audit and parse official BDD100K labels package."""

from __future__ import annotations

import json
import os
import sys
import zipfile
from collections import Counter
from typing import Dict, List, Any

BDD_SCENE_CLASSES = [
    "city street",
    "highway",
    "residential",
    "parking lot",
    "gas stations",
    "tunnel",
]


def audit_labels_zip(zip_path: str, extract_to: str) -> Dict[str, Any]:
    print(f"=== Auditing BDD100K Labels Zip: {zip_path} ===")
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    # Check zip integrity
    print(f"File size: {os.path.getsize(zip_path) / 1024 / 1024:.2f} MB")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        print(f"Zip contains {len(namelist)} items:")
        for name in namelist[:20]:
            print(f"  - {name}")

        print("\nExtracting json annotation files...")
        os.makedirs(extract_to, exist_ok=True)
        for item in namelist:
            if item.endswith(".json"):
                zf.extract(item, extract_to)
                print(f"  Extracted: {item}")

    # Now inspect extracted JSON files
    stats = {}
    for root, _, files in os.walk(extract_to):
        for f in files:
            if not f.endswith(".json") or "samples" in f or "manifest" in f:
                continue
            full_path = os.path.join(root, f)
            print(f"\n--- Inspecting {f} ({os.path.getsize(full_path) / 1024 / 1024:.2f} MB) ---")
            try:
                with open(full_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                if isinstance(data, list):
                    total_frames = len(data)
                    scenes = Counter()
                    weathers = Counter()
                    times = Counter()
                    valid_6class = 0
                    undefined_scene = 0
                    missing_scene = 0

                    for frame in data:
                        attrs = frame.get("attributes", {})
                        scene = attrs.get("scene")
                        weather = attrs.get("weather")
                        tod = attrs.get("timeofday")

                        if scene:
                            scenes[scene] += 1
                            if scene in BDD_SCENE_CLASSES:
                                valid_6class += 1
                            elif scene == "undefined":
                                undefined_scene += 1
                        else:
                            missing_scene += 1

                        if weather:
                            weathers[weather] += 1
                        if tod:
                            times[tod] += 1

                    stats[f] = {
                        "total_frames": total_frames,
                        "valid_6class": valid_6class,
                        "undefined_scene": undefined_scene,
                        "missing_scene": missing_scene,
                        "scenes": dict(scenes),
                        "weathers": dict(weathers),
                        "timeofday": dict(times),
                    }
                    print(f"  Total frames: {total_frames}")
                    print(f"  Valid 6-class scenes: {valid_6class}")
                    print(f"  Undefined scenes: {undefined_scene}")
                    print(f"  Missing scenes: {missing_scene}")
                    print(f"  Scene distribution: {dict(scenes)}")
            except Exception as e:
                print(f"  Error reading {f}: {e}")

    return stats


if __name__ == "__main__":
    zip_p = sys.argv[1] if len(sys.argv) > 1 else "data/bdd100k/downloads/bdd100k_labels.zip"
    ext_p = sys.argv[2] if len(sys.argv) > 2 else "data/bdd100k/annotations/official_labels"
    audit_labels_zip(zip_p, ext_p)
