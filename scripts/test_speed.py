import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

folder = "data/bdd100k/annotations/official_labels/100k/train"
files = [f for f in os.listdir(folder) if f.endswith(".json")]
print(f"Total files: {len(files)}")

scene_pattern = re.compile(r'"scene":\s*"([^"]+)"')
weather_pattern = re.compile(r'"weather":\s*"([^"]+)"')
tod_pattern = re.compile(r'"timeofday":\s*"([^"]+)"')

def parse_file(f):
    path = os.path.join(folder, f)
    try:
        with open(path, "rb") as fp:
            fp.seek(0, 2)
            sz = fp.tell()
            fp.seek(max(0, sz - 1024), 0)
            tail = fp.read().decode("utf-8", errors="ignore")
        s = scene_pattern.search(tail)
        w = weather_pattern.search(tail)
        t = tod_pattern.search(tail)
        return (s.group(1) if s else None, w.group(1) if w else None, t.group(1) if t else None)
    except Exception:
        return (None, None, None)

t0 = time.time()
with ThreadPoolExecutor(max_workers=16) as ex:
    results = list(ex.map(parse_file, files[:5000]))
dt = time.time() - t0
print(f"5,000 files parsed in {dt:.2f} s ({5000/dt:.0f} files/s)")
