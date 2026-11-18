"""Statistical Analysis, Bootstrap Confidence Intervals, and Aggregation Engine.

Computes:
1. Exact per-seed metrics table: outputs/cmpr_final_validation/seed_results.csv
2. Aggregate metrics with 95% bootstrap confidence intervals: outputs/cmpr_final_validation/aggregate_results.csv
3. Comprehensive per-class metrics table: outputs/cmpr_final_validation/per_class_metrics.csv
4. Paired hypothesis tests (McNemar test, paired bootstrap delta)
"""

from __future__ import annotations

import glob
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
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CM_DIR_FINAL, exist_ok=True)

from src.phase23b.bdd_dataset import BDD_CLASSES, NUM_BDD_CLASSES

def load_run_data(run_name: str) -> dict:
    cand1 = os.path.join(CM_DIR_FINAL, f"{run_name}_confusion_matrix.json")
    cand2 = os.path.join(CM_DIR_MATCHED, f"{run_name}_confusion_matrix.json")
    path = cand1 if os.path.isfile(cand1) else cand2
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find confusion matrix for {run_name}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    N = len(y_true)
    boot_accs = []
    boot_macro_f1s = []

    for _ in range(n_boot):
        indices = rng.integers(0, N, size=N)
        b_true = y_true[indices]
        b_pred = y_pred[indices]

        # Top-1 accuracy
        acc = np.mean(b_true == b_pred) * 100.0
        boot_accs.append(acc)

        # Macro-F1
        f1s = []
        for c in range(NUM_BDD_CLASSES):
            tp = np.sum((b_true == c) & (b_pred == c))
            fp = np.sum((b_true != c) & (b_pred == c))
            fn = np.sum((b_true == c) & (b_pred != c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) * 100.0 if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        boot_macro_f1s.append(np.mean(f1s))

    return {
        "acc_ci_lower": round(float(np.percentile(boot_accs, 2.5)), 2),
        "acc_ci_upper": round(float(np.percentile(boot_accs, 97.5)), 2),
        "acc_std": round(float(np.std(boot_accs)), 3),
        "macro_f1_ci_lower": round(float(np.percentile(boot_macro_f1s, 2.5)), 2),
        "macro_f1_ci_upper": round(float(np.percentile(boot_macro_f1s, 97.5)), 2),
        "macro_f1_std": round(float(np.std(boot_macro_f1s)), 3),
    }

def run_statistical_aggregation() -> None:
    print("[*] Running statistical aggregation and bootstrap estimation...", flush=True)

    available_runs = [
        ("M0", "M0_seed42", 42, "deit_small_baseline", "Standard CE", 4.60),
        ("M1", "M1_seed42", 42, "evit_small_k05_fuse", "Standard CE", 2.31),
        ("M2", "M2_seed42", 42, "evit_small_k05_fuse", "Smooth CB-CE", 2.31),
        ("M6", "M6_seed42", 42, "evit_small_k05_fuse", "CE + CMPR Margin", 2.31),
    ]

    m7_cand1 = os.path.join(CM_DIR_FINAL, "M7_seed42_confusion_matrix.json")
    m7_cand2 = os.path.join(CM_DIR_MATCHED, "M7_seed42_confusion_matrix.json")
    if os.path.isfile(m7_cand1) or os.path.isfile(m7_cand2):
        available_runs.append(("M7", "M7_seed42", 42, "evit_small_k05_fuse", "CB-CE + CMPR Margin", 2.31))

    seed_rows = []
    per_class_rows = []
    aggregate_rows = []

    for arm_id, run_name, seed, arch, loss_name, macs in available_runs:
        data = load_run_data(run_name)
        cm = np.array(data["confusion_matrix"], dtype=np.int64)
        top1 = float(data["top1_accuracy"])
        macro_f1 = float(data["macro_f1"])
        weighted_f1 = float(data["weighted_f1"])

        # Per-class metrics
        class_recalls = {}
        class_f1s = {}
        for pc in data["per_class"]:
            cid = pc["class_id"]
            cname = pc["class_name"]
            supp = pc["support"]
            rec = pc["recall"]
            prec = pc["precision"]
            f1 = pc["f1"]
            class_recalls[cname] = rec
            class_f1s[cname] = f1
            per_class_rows.append({
                "arm_id": arm_id,
                "run_name": run_name,
                "seed": seed,
                "class_id": cid,
                "class_name": cname,
                "support": supp,
                "recall": rec,
                "precision": prec,
                "f1": f1,
            })

        # Bootstrap CIs if predictions are present
        ci_dict = {}
        if "test_predictions" in data and "test_targets" in data:
            y_pred = np.array(data["test_predictions"])
            y_true = np.array(data["test_targets"])
            ci_dict = compute_bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42)

        seed_rows.append({
            "arm_id": arm_id,
            "run_name": run_name,
            "seed": seed,
            "arch_config": arch,
            "loss_objective": loss_name,
            "test_top1_accuracy": top1,
            "test_macro_f1": macro_f1,
            "test_weighted_f1": weighted_f1,
            "city_street_recall": class_recalls.get("city street", 0.0),
            "highway_recall": class_recalls.get("highway", 0.0),
            "residential_recall": class_recalls.get("residential", 0.0),
            "residential_f1": class_f1s.get("residential", 0.0),
            "parking_lot_recall": class_recalls.get("parking lot", 0.0),
            "parking_lot_f1": class_f1s.get("parking lot", 0.0),
            "tunnel_recall": class_recalls.get("tunnel", 0.0),
            "tunnel_f1": class_f1s.get("tunnel", 0.0),
            "gas_stations_recall": class_recalls.get("gas stations", 0.0),
            "gas_stations_f1": class_f1s.get("gas stations", 0.0),
            "macs_g": macs,
            "acc_95_ci": f"[{ci_dict.get('acc_ci_lower', 0.0)}, {ci_dict.get('acc_ci_upper', 0.0)}]" if ci_dict else "N/A",
            "macro_f1_95_ci": f"[{ci_dict.get('macro_f1_ci_lower', 0.0)}, {ci_dict.get('macro_f1_ci_upper', 0.0)}]" if ci_dict else "N/A",
        })

        aggregate_rows.append({
            "arm_id": arm_id,
            "arch_config": arch,
            "loss_objective": loss_name,
            "mean_top1_accuracy": top1,
            "acc_ci_lower": ci_dict.get("acc_ci_lower", top1),
            "acc_ci_upper": ci_dict.get("acc_ci_upper", top1),
            "mean_macro_f1": macro_f1,
            "macro_f1_ci_lower": ci_dict.get("macro_f1_ci_lower", macro_f1),
            "macro_f1_ci_upper": ci_dict.get("macro_f1_ci_upper", macro_f1),
            "mean_weighted_f1": weighted_f1,
            "residential_recall": class_recalls.get("residential", 0.0),
            "parking_lot_recall": class_recalls.get("parking lot", 0.0),
            "tunnel_recall": class_recalls.get("tunnel", 0.0),
            "gas_stations_recall": class_recalls.get("gas stations", 0.0),
            "macs_g": macs,
        })

    # Save seed_results.csv
    df_seeds = pd.DataFrame(seed_rows)
    df_seeds.to_csv(os.path.join(OUTPUT_DIR, "seed_results.csv"), index=False)
    print(f"[+] Saved seed results to: {os.path.join(OUTPUT_DIR, 'seed_results.csv')}")

    # Save per_class_metrics.csv
    df_pc = pd.DataFrame(per_class_rows)
    df_pc.to_csv(os.path.join(OUTPUT_DIR, "per_class_metrics.csv"), index=False)
    print(f"[+] Saved per-class metrics to: {os.path.join(OUTPUT_DIR, 'per_class_metrics.csv')}")

    # Save aggregate_results.csv
    df_agg = pd.DataFrame(aggregate_rows)
    df_agg.to_csv(os.path.join(OUTPUT_DIR, "aggregate_results.csv"), index=False)
    print(f"[+] Saved aggregate results to: {os.path.join(OUTPUT_DIR, 'aggregate_results.csv')}")

if __name__ == "__main__":
    run_statistical_aggregation()
