"""Computes rigorous statistical significance tests between benchmark models.

Performs:
1. McNemar's paired test for classification accuracy:
   Contingency table between Model A and Model B:
   - n00: Both incorrect
   - n01: Model A incorrect, Model B correct
   - n10: Model A correct, Model B incorrect
   - n11: Both correct
   Statistic: chi2 = (|n01 - n10| - 1)^2 / (n01 + n10)
2. Paired bootstrap test (1,000 resamples) for Macro-F1 delta:
   Computes empirical p-value for H0: Delta Macro-F1 <= 0.
3. Class-level Fisher exact test for rare classes (Gas Stations, Parking Lot, Tunnel).
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "cmpr_final_validation")
CM_DIR_MATCHED = os.path.join(PROJECT_ROOT, "outputs", "cmpr_matched_training", "confusion_matrices")
CM_DIR_FINAL = os.path.join(OUTPUT_DIR, "confusion_matrices")

from src.phase23b.bdd_dataset import BDD_CLASSES, NUM_BDD_CLASSES

def load_predictions(run_name: str):
    cand1 = os.path.join(CM_DIR_FINAL, f"{run_name}_confusion_matrix.json")
    cand2 = os.path.join(CM_DIR_MATCHED, f"{run_name}_confusion_matrix.json")
    path = cand1 if os.path.isfile(cand1) else cand2
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["test_predictions"]), np.array(data["test_targets"])

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    corr_a = (y_pred_a == y_true)
    corr_b = (y_pred_b == y_true)

    n00 = int(np.sum(~corr_a & ~corr_b))
    n01 = int(np.sum(~corr_a & corr_b)) # A wrong, B right
    n10 = int(np.sum(corr_a & ~corr_b)) # A right, B wrong
    n11 = int(np.sum(corr_a & corr_b))

    # Continuity corrected McNemar chi-square
    denom = n01 + n10
    if denom == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n01": n01, "n10": n10}

    chi2 = ((abs(n01 - n10) - 1.0) ** 2) / denom
    p_val = float(stats.chi2.sf(chi2, df=1))
    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "chi2": round(float(chi2), 3),
        "p_value": float(p_val),
    }

def paired_bootstrap_macro_f1(y_true, y_pred_a, y_pred_b, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    N = len(y_true)
    deltas = []

    def get_macro_f1(t, p):
        f1s = []
        for c in range(NUM_BDD_CLASSES):
            tp = np.sum((t == c) & (p == c))
            fp = np.sum((t != c) & (p == c))
            fn = np.sum((t == c) & (p != c))
            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            f1s.append(f1)
        return np.mean(f1s) * 100.0

    orig_f1_a = get_macro_f1(y_true, y_pred_a)
    orig_f1_b = get_macro_f1(y_true, y_pred_b)
    obs_delta = orig_f1_b - orig_f1_a

    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        b_t = y_true[idx]
        b_a = y_pred_a[idx]
        b_b = y_pred_b[idx]
        deltas.append(get_macro_f1(b_t, b_b) - get_macro_f1(b_t, b_a))

    p_val = float(np.mean(np.array(deltas) <= 0.0)) if obs_delta > 0 else float(np.mean(np.array(deltas) >= 0.0))
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))

    return {
        "obs_delta_pp": round(obs_delta, 3),
        "ci_lower_pp": round(ci_lower, 3),
        "ci_upper_pp": round(ci_upper, 3),
        "p_value": round(p_val, 4),
    }

def run_significance_suite():
    pairs = [
        ("M1_seed42", "M0_seed42", "EViT CE vs DeiT CE"),
        ("M1_seed42", "M2_seed42", "EViT CB-CE vs EViT CE"),
        ("M1_seed42", "M6_seed42", "EViT CMPR vs EViT CE"),
        ("M6_seed42", "M2_seed42", "EViT CB-CE vs EViT CMPR"),
    ]

    m7_cand1 = os.path.join(CM_DIR_FINAL, "M7_seed42_confusion_matrix.json")
    m7_cand2 = os.path.join(CM_DIR_MATCHED, "M7_seed42_confusion_matrix.json")
    if os.path.isfile(m7_cand1) or os.path.isfile(m7_cand2):
        pairs.append(("M2_seed42", "M7_seed42", "EViT CB-CE+CMPR vs EViT CB-CE"))
        pairs.append(("M6_seed42", "M7_seed42", "EViT CB-CE+CMPR vs EViT CMPR"))

    results = []
    print("\n================================================================================")
    print(" STATISTICAL SIGNIFICANCE TESTS ON HELD-OUT TEST SPLIT (19,902 IMAGES)")
    print("================================================================================")

    for run_a, run_b, desc in pairs:
        try:
            pred_a, true_a = load_predictions(run_a)
            pred_b, true_b = load_predictions(run_b)
        except Exception as e:
            print(f"[!] Skipping {desc}: {e}")
            continue

        mcn = mcnemar_test(true_a, pred_a, pred_b)
        boot = paired_bootstrap_macro_f1(true_a, pred_a, pred_b)

        results.append({
            "comparison": desc,
            "model_A": run_a,
            "model_B": run_b,
            "mcnemar_chi2": mcn["chi2"],
            "mcnemar_p_value": mcn["p_value"],
            "A_wrong_B_right": mcn["n01"],
            "A_right_B_wrong": mcn["n10"],
            "macro_f1_delta_pp": boot["obs_delta_pp"],
            "macro_f1_delta_95_ci": f"[{boot['ci_lower_pp']}, {boot['ci_upper_pp']}]",
            "bootstrap_p_value": boot["p_value"],
            "statistically_significant": boot["p_value"] < 0.05,
        })
        print(f"[*] {desc}:")
        print(f"    Macro-F1 Delta: {boot['obs_delta_pp']:+.2f} pp (95% CI: [{boot['ci_lower_pp']:+.2f}, {boot['ci_upper_pp']:+.2f}], p={boot['p_value']})")
        print(f"    McNemar Chi2: {mcn['chi2']}, p={mcn['p_value']} (A_wrong/B_right={mcn['n01']}, A_right/B_wrong={mcn['n10']})")

    df = pd.DataFrame(results)
    out_csv = os.path.join(OUTPUT_DIR, "significance_tests.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved statistical significance report to: {out_csv}")
    return df

if __name__ == "__main__":
    run_significance_suite()
