"""Download full BDD100K dataset using dataset-tools (Dataset Ninja)."""

import os
import sys
import time

# Set MPLCONFIGDIR to temp to avoid matplotlib cache warnings
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/Temp"), "mpl")

print("=" * 80)
print(" Downloading BDD100K: Images 100K via dataset-tools (Dataset Ninja)")
print("=" * 80)

import dataset_tools as dtools

dst_dir = os.path.expanduser("~/dataset-ninja/")
os.makedirs(dst_dir, exist_ok=True)
print(f"Target destination: {dst_dir}")

t0 = time.time()
try:
    extracted_path = dtools.download(
        dataset="BDD100K: Images 100K",
        dst_dir=dst_dir,
        unpack_archive=True,
    )
    elapsed = round(time.time() - t0, 1)
    print(f"\nSuccessfully downloaded and unpacked to: {extracted_path} in {elapsed}s")
except Exception as e:
    print(f"\nDownload / Extraction error: {e}")
    sys.exit(1)

print("\nScanning extracted directory structure...")
for root, dirs, files in os.walk(extracted_path):
    rel = os.path.relpath(root, extracted_path)
    img_count = sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    if img_count > 0:
        print(f"  Folder: {rel} -> {img_count} images ({len(files)} total files)")

print("\nDone!")
