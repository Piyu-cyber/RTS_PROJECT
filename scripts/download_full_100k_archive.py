"""High-speed multi-stream downloader with live telemetry for BDD100K 100K archive."""

from __future__ import annotations

import os
import sys
import time
import zipfile
import requests
import concurrent.futures

URL = "http://128.32.162.150/bdd100k/bdd100k_images_100k.zip"
TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "bdd100k"))
TARGET_ZIP = os.path.join(TARGET_DIR, "bdd100k_images_100k.zip")
NUM_WORKERS = 4
CHUNK_BUFFER = 1024 * 256  # 256 KB


def get_remote_file_size(url: str) -> int:
    for attempt in range(5):
        try:
            r = requests.head(url, timeout=30)
            if r.status_code == 200:
                cl = r.headers.get("content-length")
                if cl:
                    return int(cl)
        except Exception as e:
            print(f"HEAD request attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(2)
    raise RuntimeError(f"Unable to retrieve Content-Length from {url}")


def download_part(url: str, start: int, end: int, part_path: str, part_idx: int) -> None:
    part_size = end - start + 1
    existing_bytes = 0

    if os.path.exists(part_path):
        existing_bytes = os.path.getsize(part_path)
        if existing_bytes == part_size:
            print(f"[Part {part_idx}] Already complete ({part_size / (1024**2):.1f} MB).", flush=True)
            return
        elif existing_bytes > part_size:
            os.remove(part_path)
            existing_bytes = 0

    max_retries = 30
    last_logged_mb = existing_bytes // (1024 * 1024 * 25)  # Log every 25 MB

    for attempt in range(max_retries):
        current_existing = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        if current_existing >= part_size:
            break

        current_start = start + current_existing
        headers = {"Range": f"bytes={current_start}-{end}"}

        try:
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as resp:
                if resp.status_code not in (200, 206):
                    raise RuntimeError(f"Server returned HTTP {resp.status_code}")

                with open(part_path, "ab" if current_existing > 0 else "wb") as f:
                    for chunk in resp.iter_content(CHUNK_BUFFER):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            current_existing += len(chunk)
                            cur_mb = current_existing // (1024 * 1024 * 25)
                            if cur_mb > last_logged_mb:
                                last_logged_mb = cur_mb
                                pct = (current_existing / part_size) * 100
                                mb_done = current_existing / (1024 * 1024)
                                mb_total = part_size / (1024 * 1024)
                                print(f"  --> [Part {part_idx}] {mb_done:.1f} MB / {mb_total:.1f} MB ({pct:.1f}%)", flush=True)

            if current_existing >= part_size:
                break
        except Exception as e:
            print(f"[Part {part_idx}] Retry {attempt + 1}/{max_retries} due to: {e}", flush=True)
            time.sleep(3)

    final_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0
    if final_size != part_size:
        raise RuntimeError(f"Part {part_idx} incomplete: expected {part_size} bytes, got {final_size} bytes")
    print(f"[Part {part_idx}] Download completed successfully! ({final_size / (1024**2):.1f} MB)", flush=True)


def download_and_extract_full_archive():
    os.makedirs(TARGET_DIR, exist_ok=True)
    images_dir = os.path.join(TARGET_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("=" * 80, flush=True)
    print(" BDD100K 100K Archive High-Speed Multi-Stream Downloader", flush=True)
    print("=" * 80, flush=True)
    print(f" Source URL:       {URL}", flush=True)
    print(f" Output Location:  {TARGET_ZIP}", flush=True)
    print(f" Parallel Streams: {NUM_WORKERS}", flush=True)

    total_size = get_remote_file_size(URL)
    print(f" Total File Size:  {total_size:,} bytes ({total_size / (1024**3):.2f} GiB / {total_size / (1024**2):.1f} MiB)", flush=True)

    # Clean old 8-part files if switching to 4-part layout
    old_parts = [f for f in os.listdir(TARGET_DIR) if f.startswith("bdd100k_images_100k.zip.part_")]
    if len(old_parts) != NUM_WORKERS:
        print(f" Reconfiguring chunk partitions to {NUM_WORKERS} streams...", flush=True)
        for op in old_parts:
            try:
                os.remove(os.path.join(TARGET_DIR, op))
            except OSError:
                pass

    part_size = total_size // NUM_WORKERS
    parts = []
    part_files = []

    for i in range(NUM_WORKERS):
        start = i * part_size
        end = total_size - 1 if i == NUM_WORKERS - 1 else (i + 1) * part_size - 1
        part_path = f"{TARGET_ZIP}.part_{i:02d}"
        parts.append((start, end, part_path, i))
        part_files.append(part_path)

    t0 = time.time()
    print("\n Starting parallel chunk downloads...", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(download_part, URL, s, e, p, idx): idx
            for s, e, p, idx in parts
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[Part {idx}] FAILED with unrecoverable error: {e}", flush=True)
                sys.exit(1)

    dl_time = round(time.time() - t0, 1)
    avg_speed = (total_size / (1024 * 1024)) / max(dl_time, 1)
    print(f"\n All {NUM_WORKERS} parts downloaded in {dl_time}s ({avg_speed:.2f} MB/s)!", flush=True)

    # Combine parts into final zip
    print("\n Assembling parts into bdd100k_images_100k.zip...", flush=True)
    assemble_t0 = time.time()
    with open(TARGET_ZIP, "wb") as outfile:
        for p in part_files:
            with open(p, "rb") as infile:
                while True:
                    buf = infile.read(1024 * 1024 * 8)  # 8 MB buffer
                    if not buf:
                        break
                    outfile.write(buf)

    print(f" Assembly complete in {time.time() - assemble_t0:.1f}s.", flush=True)

    # Remove temporary part files
    for p in part_files:
        try:
            os.remove(p)
        except OSError:
            pass

    # Validate zip integrity
    print("\n Verifying ZIP integrity...", flush=True)
    if not zipfile.is_zipfile(TARGET_ZIP):
        raise ValueError(" Downloaded file is not a valid zip archive!")
    print(" ZIP archive validated successfully.", flush=True)

    # Extract images
    print("\n Extracting images into data/bdd100k/images/...", flush=True)
    extract_t0 = time.time()
    with zipfile.ZipFile(TARGET_ZIP, "r") as zf:
        zf.extractall(images_dir)
    print(f" Extraction complete in {time.time() - extract_t0:.1f}s.", flush=True)

    total_images = sum(1 for f in os.listdir(images_dir) if f.lower().endswith(".jpg"))
    print(f"\n Total images currently in {images_dir}: {total_images:,}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    download_and_extract_full_archive()
