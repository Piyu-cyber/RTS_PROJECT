"""Download official checkpoints for DeiT-Small and EViT models."""

import os
import sys
import urllib.request
import gdown

WEIGHTS_DIR = os.path.abspath("reference/evit_weights")

CHECKPOINTS = {
    "deit_small_patch16_224": {
        "type": "direct",
        "url": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        "filename": "deit_small_patch16_224.pth",
    },
    "evit_small_patch16_0.9_fuse": {
        "type": "gdrive",
        "id": "1CmROmlLV-7nyYY-Bhph8_5ut3wFwqvny",
        "filename": "evit_small_patch16_0.9_fuse.pth",
    },
    "evit_small_patch16_0.7_fuse": {
        "type": "gdrive",
        "id": "1FfvdEBss9f8gexjqEY8vOSUQe_8qwYuL",
        "filename": "evit_small_patch16_0.7_fuse.pth",
    },
    "evit_small_patch16_0.5_fuse": {
        "type": "gdrive",
        "id": "1-nyxAlRWjD2icbgj3Y3YWcCmYj9sTgfD",
        "filename": "evit_small_patch16_0.5_fuse.pth",
    },
}


def download_all(target_dir=WEIGHTS_DIR):
    os.makedirs(target_dir, exist_ok=True)
    status = {}
    for name, info in CHECKPOINTS.items():
        dest = os.path.join(target_dir, info["filename"])
        if os.path.exists(dest) and os.path.getsize(dest) > 1024 * 1024:
            print(f"[EXISTS] {name} -> {dest} ({os.path.getsize(dest)} bytes)")
            status[name] = {"path": dest, "status": "existing", "size": os.path.getsize(dest)}
            continue

        print(f"[DOWNLOADING] {name}...")
        try:
            if info["type"] == "direct":
                urllib.request.urlretrieve(info["url"], dest)
            elif info["type"] == "gdrive":
                gdown.download(id=info["id"], output=dest, quiet=False)

            if os.path.exists(dest) and os.path.getsize(dest) > 1024 * 1024:
                size = os.path.getsize(dest)
                print(f"[SUCCESS] {name} downloaded ({size} bytes)")
                status[name] = {"path": dest, "status": "downloaded", "size": size}
            else:
                print(f"[FAILED] {name} file missing or too small")
                status[name] = {"path": dest, "status": "failed", "size": 0}
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            status[name] = {"path": dest, "status": f"error: {e}", "size": 0}

    return status


if __name__ == "__main__":
    download_all()
