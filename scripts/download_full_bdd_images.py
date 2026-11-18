"""Download verified BDD100K images from official mirror using multi-threaded streaming."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from typing import List, Dict, Any, Tuple
import requests
from PIL import Image

IMAGE_BASE_URL = "https://huggingface.co/datasets/dgural/bdd100k/resolve/main/data"


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


def download_one(args: Tuple[str, str]) -> Tuple[str, bool]:
    filename, dest_dir = args
    dest_path = os.path.join(dest_dir, filename)

    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1000:
        if verify_image(dest_path):
            return filename, True

    url = f"{IMAGE_BASE_URL}/{filename}"
    temp_path = dest_path + ".tmp"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            with open(temp_path, "wb") as f:
                f.write(r.content)
            if verify_image(temp_path):
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                os.rename(temp_path, dest_path)
                return filename, True
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return filename, False
        else:
            return filename, False
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return filename, False


def download_batch(
    manifest_path: str = "data/bdd100k/annotations/val_full.json",
    dest_dir: str = "data/bdd100k/images",
    max_images: int = 1500,
    threads: int = 32,
) -> int:
    os.makedirs(dest_dir, exist_ok=True)
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Filter out already existing and verified
    needed = []
    already_have = 0
    for r in records:
        fn = r["filename"]
        path = os.path.join(dest_dir, fn)
        if os.path.isfile(path) and os.path.getsize(path) > 1000 and verify_image(path):
            already_have += 1
        else:
            needed.append(fn)

    print(f"Total manifest records: {len(records)}")
    print(f"Already downloaded and verified: {already_have}")
    target_downloads = needed[:max(0, max_images - already_have)]
    print(f"Targeting {len(target_downloads)} new images using {threads} parallel workers...")

    if not target_downloads:
        print("No new downloads required.")
        return already_have

    t0 = time.time()
    success_count = already_have
    tasks = [(fn, dest_dir) for fn in target_downloads]

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        for fn, ok in ex.map(download_one, tasks):
            if ok:
                success_count += 1
            if (success_count) % 100 == 0:
                print(f"  Verified {success_count} images ({time.time() - t0:.1f} s)...", flush=True)

    dt = time.time() - t0
    print(f"Downloaded and verified {success_count} total images in {dt:.1f} s ({len(target_downloads)/dt:.1f} img/s).")
    return success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=1500)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    download_batch(max_images=args.max_images, threads=args.threads)
