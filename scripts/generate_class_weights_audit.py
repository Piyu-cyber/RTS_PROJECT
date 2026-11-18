"""Generates exact class-balanced weights audit for BDD100K driving scene classification.

Compares exact dataset counts vs hardcoded approximation and documents normalization rules.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "cmpr_final_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from src.phase23b.bdd_dataset import BDD_CLASSES, NUM_BDD_CLASSES

# Exact ground-truth counts from train_full.json (69,634 total)
exact_counts = np.array([43581, 17414, 8105, 378, 27, 129], dtype=np.float64)
# Hardcoded approximation in earlier run_cmpr_matched_experiments.py
used_counts = np.array([43158, 17816, 8409, 382, 77, 21], dtype=np.float64)
# Test split support (19,902 total)
test_support = np.array([12288, 5069, 2382, 108, 6, 49], dtype=np.int64)

N_exact = np.sum(exact_counts)
N_used = np.sum(used_counts)
C = float(NUM_BDD_CLASSES)

# Exact smoothed inverse-frequency weights: (N / (C * n_c))^0.25
raw_exact = (N_exact / (C * exact_counts)) ** 0.25
norm_exact = raw_exact / np.mean(raw_exact)

# Used smoothed inverse-frequency weights
raw_used = (N_used / (C * used_counts)) ** 0.25
norm_used = raw_used / np.mean(raw_used)

rows = []
for i in range(NUM_BDD_CLASSES):
    diff_pct = ((norm_used[i] - norm_exact[i]) / norm_exact[i]) * 100.0
    rows.append({
        "class_id": i,
        "class_name": BDD_CLASSES[i],
        "exact_train_count": int(exact_counts[i]),
        "train_share_pct": round(float(exact_counts[i] / N_exact * 100.0), 3),
        "test_support": int(test_support[i]),
        "test_share_pct": round(float(test_support[i] / np.sum(test_support) * 100.0), 3),
        "exact_raw_weight": round(float(raw_exact[i]), 4),
        "exact_norm_weight": round(float(norm_exact[i]), 4),
        "used_train_count": int(used_counts[i]),
        "used_raw_weight": round(float(raw_used[i]), 4),
        "used_norm_weight": round(float(norm_used[i]), 4),
        "norm_weight_delta_pct": round(float(diff_pct), 2),
    })

df = pd.DataFrame(rows)
out_csv = os.path.join(OUTPUT_DIR, "class_weights.csv")
df.to_csv(out_csv, index=False)
print(f"[+] Successfully generated class weights audit: {out_csv}")
print(df.to_string(index=False))
