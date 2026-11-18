"""Compile all multi-seed validation results across Seeds 42, 123, and 456."""

import glob
import json
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.phase23b.bdd_dataset import BDD_CLASSES, NUM_BDD_CLASSES

STUDY_DIR = os.path.join(PROJECT_ROOT, "outputs", "cross_keep_rate_study")
MULTI_SEED_DIR = os.path.join(STUDY_DIR, "multi_seed_validation")
SEED_123_DIR = os.path.join(MULTI_SEED_DIR, "seed_123")
SEED_456_DIR = os.path.join(MULTI_SEED_DIR, "seed_456")

# Checkpoint paths and confusion matrix paths
RUNS = [
    # (model_id, seed, cm_json_path, config_json_path, ckpt_path)
    ("K07-CB", 42,
     os.path.join(STUDY_DIR, "confusion_matrices", "K07-CB_seed42_confusion_matrix.json"),
     os.path.join(STUDY_DIR, "training_configurations", "K07-CB_seed42_config.json"),
     os.path.join(STUDY_DIR, "checkpoints", "K07-CB_seed42_best.pth")),
    ("K07-CB-CMPR", 42,
     os.path.join(STUDY_DIR, "confusion_matrices", "K07-CB-CMPR_seed42_confusion_matrix.json"),
     os.path.join(STUDY_DIR, "training_configurations", "K07-CB-CMPR_seed42_config.json"),
     os.path.join(STUDY_DIR, "checkpoints", "K07-CB-CMPR_seed42_best.pth")),

    ("K07-CB", 123,
     os.path.join(SEED_123_DIR, "confusion_matrices", "K07-CB_seed123_confusion_matrix.json"),
     os.path.join(SEED_123_DIR, "training_configurations", "K07-CB_seed123_config.json"),
     os.path.join(SEED_123_DIR, "checkpoints", "K07-CB_seed123_best.pth")),
    ("K07-CB-CMPR", 123,
     os.path.join(SEED_123_DIR, "confusion_matrices", "K07-CB-CMPR_seed123_confusion_matrix.json"),
     os.path.join(SEED_123_DIR, "training_configurations", "K07-CB-CMPR_seed123_config.json"),
     os.path.join(SEED_123_DIR, "checkpoints", "K07-CB-CMPR_seed123_best.pth")),

    ("K07-CB", 456,
     os.path.join(SEED_456_DIR, "confusion_matrices", "K07-CB_seed456_confusion_matrix.json"),
     os.path.join(SEED_456_DIR, "training_configurations", "K07-CB_seed456_config.json"),
     os.path.join(SEED_456_DIR, "checkpoints", "K07-CB_seed456_best.pth")),
    ("K07-CB-CMPR", 456,
     os.path.join(SEED_456_DIR, "confusion_matrices", "K07-CB-CMPR_seed456_confusion_matrix.json"),
     os.path.join(SEED_456_DIR, "training_configurations", "K07-CB-CMPR_seed456_config.json"),
     os.path.join(SEED_456_DIR, "checkpoints", "K07-CB-CMPR_seed456_best.pth")),
]


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 2000, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        b_true = y_true[idx]
        b_pred = y_pred[idx]
        accs.append(np.mean(b_true == b_pred) * 100.0)

        b_f1s = []
        for c in range(NUM_BDD_CLASSES):
            tp = np.sum((b_true == c) & (b_pred == c))
            fn = np.sum((b_true == c) & (b_pred != c))
            fp = np.sum((b_true != c) & (b_pred == c))
            supp = tp + fn
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / supp if supp > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec) * 100.0) if (prec + rec) > 0 else 0.0
            b_f1s.append(f1)
        f1s.append(np.mean(b_f1s))

    return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))), \
           (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5)))


def main():
    print("Compiling Multi-Seed Validation Artifacts...")

    manifest_rows = []
    per_seed_rows = []
    per_class_rows = []
    bootstrap_rows = []

    all_data = {}

    for model_id, seed, cm_path, cfg_path, ckpt_path in RUNS:
        run_name = f"{model_id}_seed{seed}"
        print(f"Loading {run_name} from {cm_path}...")
        with open(cm_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        all_data[run_name] = d

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        y_true = np.array(d["test_targets"])
        y_pred = np.array(d["test_predictions"])

        acc_ci, f1_ci = bootstrap_ci(y_true, y_pred, n_boot=2000, seed=seed)

        # Gas station correct out of 6
        mask_gas = (y_true == 4)
        gas_correct = int(np.sum((y_pred == 4) & mask_gas))

        # Manifest row
        manifest_rows.append({
            "run_name": run_name,
            "model_id": model_id,
            "seed": seed,
            "arch_config": cfg["arch_config"],
            "loss_type": cfg["loss_type"],
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "lr_head": cfg["lr_head"],
            "lr_backbone": cfg["lr_backbone"],
            "verified_class_weights": str(cfg.get("verified_class_weights", [])),
            "teacher_checkpoint_hash": cfg.get("teacher_checkpoint_hash", "N/A"),
            "checkpoint_path": ckpt_path,
            "checkpoint_exists": os.path.isfile(ckpt_path),
        })

        # Per seed row
        ps_row = {
            "model_id": model_id,
            "seed": seed,
            "run_name": run_name,
            "top1_accuracy": d["top1_accuracy"],
            "macro_f1": d["macro_f1"],
            "weighted_f1": d["weighted_f1"],
            "gas_station_correct_of_6": gas_correct,
            "gas_station_recall": [pc["recall"] for pc in d["per_class"] if pc["class_id"] == 4][0],
            "gas_station_f1": [pc["f1"] for pc in d["per_class"] if pc["class_id"] == 4][0],
            "city_street_recall": [pc["recall"] for pc in d["per_class"] if pc["class_id"] == 0][0],
            "residential_recall": [pc["recall"] for pc in d["per_class"] if pc["class_id"] == 2][0],
            "parking_lot_recall": [pc["recall"] for pc in d["per_class"] if pc["class_id"] == 3][0],
            "tunnel_f1": [pc["f1"] for pc in d["per_class"] if pc["class_id"] == 5][0],
        }
        per_seed_rows.append(ps_row)

        # Bootstrap row
        bootstrap_rows.append({
            "model_id": model_id,
            "seed": seed,
            "run_name": run_name,
            "top1_accuracy": d["top1_accuracy"],
            "top1_95ci_low": round(acc_ci[0], 2),
            "top1_95ci_high": round(acc_ci[1], 2),
            "macro_f1": d["macro_f1"],
            "macro_f1_95ci_low": round(f1_ci[0], 2),
            "macro_f1_95ci_high": round(f1_ci[1], 2),
            "bootstrap_resamples": 2000,
        })

        # Per class rows
        for pc in d["per_class"]:
            per_class_rows.append({
                "model_id": model_id,
                "seed": seed,
                "run_name": run_name,
                "class_id": pc["class_id"],
                "class_name": pc["class_name"],
                "support": pc["support"],
                "recall": pc["recall"],
                "precision": pc["precision"],
                "f1": pc["f1"],
            })

    # 1. experiment_manifest.csv
    df_man = pd.DataFrame(manifest_rows)
    df_man.to_csv(os.path.join(MULTI_SEED_DIR, "experiment_manifest.csv"), index=False)

    # 2. per_seed_results.csv
    df_ps = pd.DataFrame(per_seed_rows)
    df_ps.to_csv(os.path.join(MULTI_SEED_DIR, "per_seed_results.csv"), index=False)

    # 3. per_class_seed_results.csv
    df_pc = pd.DataFrame(per_class_rows)
    df_pc.to_csv(os.path.join(MULTI_SEED_DIR, "per_class_seed_results.csv"), index=False)

    # 4. bootstrap_analysis.csv
    df_boot = pd.DataFrame(bootstrap_rows)
    df_boot.to_csv(os.path.join(MULTI_SEED_DIR, "bootstrap_analysis.csv"), index=False)

    # 5. paired_seed_differences.csv (CMPR minus CB for each seed)
    paired_rows = []
    for s in [42, 123, 456]:
        cb = all_data[f"K07-CB_seed{s}"]
        cmpr = all_data[f"K07-CB-CMPR_seed{s}"]
        y_true = np.array(cb["test_targets"])
        p_cb = np.array(cb["test_predictions"])
        p_cmpr = np.array(cmpr["test_predictions"])

        d_top1 = cmpr["top1_accuracy"] - cb["top1_accuracy"]
        d_macro = cmpr["macro_f1"] - cb["macro_f1"]
        d_weighted = cmpr["weighted_f1"] - cb["weighted_f1"]

        # Gas station correct
        mask_gas = (y_true == 4)
        gas_cb = int(np.sum((p_cb == 4) & mask_gas))
        gas_cmpr = int(np.sum((p_cmpr == 4) & mask_gas))

        paired_rows.append({
            "seed": s,
            "comparison": "K07-CB-CMPR minus K07-CB",
            "top1_cb": cb["top1_accuracy"],
            "top1_cmpr": cmpr["top1_accuracy"],
            "top1_diff": round(d_top1, 2),
            "macro_f1_cb": cb["macro_f1"],
            "macro_f1_cmpr": cmpr["macro_f1"],
            "macro_f1_diff": round(d_macro, 2),
            "weighted_f1_cb": cb["weighted_f1"],
            "weighted_f1_cmpr": cmpr["weighted_f1"],
            "weighted_f1_diff": round(d_weighted, 2),
            "gas_station_correct_cb_of_6": gas_cb,
            "gas_station_correct_cmpr_of_6": gas_cmpr,
            "city_street_recall_cb": [pc["recall"] for pc in cb["per_class"] if pc["class_id"] == 0][0],
            "city_street_recall_cmpr": [pc["recall"] for pc in cmpr["per_class"] if pc["class_id"] == 0][0],
            "city_street_recall_diff": round([pc["recall"] for pc in cmpr["per_class"] if pc["class_id"] == 0][0] - [pc["recall"] for pc in cb["per_class"] if pc["class_id"] == 0][0], 2),
        })

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(os.path.join(MULTI_SEED_DIR, "paired_seed_differences.csv"), index=False)

    # 6. aggregate_results.csv (mean +/- std)
    agg_rows = []
    for mid in ["K07-CB", "K07-CB-CMPR"]:
        sub = df_ps[df_ps["model_id"] == mid]
        top1_vals = sub["top1_accuracy"].values
        macro_vals = sub["macro_f1"].values
        weighted_vals = sub["weighted_f1"].values
        gas_rec_vals = sub["gas_station_recall"].values
        city_rec_vals = sub["city_street_recall"].values
        tunn_f1_vals = sub["tunnel_f1"].values

        agg_rows.append({
            "model_id": mid,
            "num_seeds": len(sub),
            "seeds": str(sub["seed"].tolist()),
            "top1_accuracy_mean": round(float(np.mean(top1_vals)), 2),
            "top1_accuracy_std": round(float(np.std(top1_vals, ddof=1)), 2),
            "top1_accuracy_str": f"{np.mean(top1_vals):.2f}% ± {np.std(top1_vals, ddof=1):.2f}%",
            "macro_f1_mean": round(float(np.mean(macro_vals)), 2),
            "macro_f1_std": round(float(np.std(macro_vals, ddof=1)), 2),
            "macro_f1_str": f"{np.mean(macro_vals):.2f}% ± {np.std(macro_vals, ddof=1):.2f}%",
            "weighted_f1_mean": round(float(np.mean(weighted_vals)), 2),
            "weighted_f1_std": round(float(np.std(weighted_vals, ddof=1)), 2),
            "weighted_f1_str": f"{np.mean(weighted_vals):.2f}% ± {np.std(weighted_vals, ddof=1):.2f}%",
            "city_street_recall_mean": round(float(np.mean(city_rec_vals)), 2),
            "city_street_recall_std": round(float(np.std(city_rec_vals, ddof=1)), 2),
            "gas_station_recall_mean": round(float(np.mean(gas_rec_vals)), 2),
            "gas_station_recall_std": round(float(np.std(gas_rec_vals, ddof=1)), 2),
            "tunnel_f1_mean": round(float(np.mean(tunn_f1_vals)), 2),
            "tunnel_f1_std": round(float(np.std(tunn_f1_vals, ddof=1)), 2),
        })

    # Paired delta aggregate row
    d_top1_vals = df_paired["top1_diff"].values
    d_macro_vals = df_paired["macro_f1_diff"].values
    d_weighted_vals = df_paired["weighted_f1_diff"].values
    agg_rows.append({
        "model_id": "DELTA (K07-CB-CMPR minus K07-CB)",
        "num_seeds": 3,
        "seeds": "[42, 123, 456]",
        "top1_accuracy_mean": round(float(np.mean(d_top1_vals)), 2),
        "top1_accuracy_std": round(float(np.std(d_top1_vals, ddof=1)), 2),
        "top1_accuracy_str": f"+{np.mean(d_top1_vals):.2f}% ± {np.std(d_top1_vals, ddof=1):.2f}%",
        "macro_f1_mean": round(float(np.mean(d_macro_vals)), 2),
        "macro_f1_std": round(float(np.std(d_macro_vals, ddof=1)), 2),
        "macro_f1_str": f"{np.mean(d_macro_vals):.2f}% ± {np.std(d_macro_vals, ddof=1):.2f}%",
        "weighted_f1_mean": round(float(np.mean(d_weighted_vals)), 2),
        "weighted_f1_std": round(float(np.std(d_weighted_vals, ddof=1)), 2),
        "weighted_f1_str": f"+{np.mean(d_weighted_vals):.2f}% ± {np.std(d_weighted_vals, ddof=1):.2f}%",
        "city_street_recall_mean": round(float(np.mean(df_paired["city_street_recall_diff"].values)), 2),
        "city_street_recall_std": round(float(np.std(df_paired["city_street_recall_diff"].values, ddof=1)), 2),
        "gas_station_recall_mean": round(float(np.mean([all_data[f'K07-CB-CMPR_seed{s}']['per_class'][4]['recall'] - all_data[f'K07-CB_seed{s}']['per_class'][4]['recall'] for s in [42, 123, 456]])), 2),
        "gas_station_recall_std": round(float(np.std([all_data[f'K07-CB-CMPR_seed{s}']['per_class'][4]['recall'] - all_data[f'K07-CB_seed{s}']['per_class'][4]['recall'] for s in [42, 123, 456]], ddof=1)), 2),
        "tunnel_f1_mean": round(float(np.mean([all_data[f'K07-CB-CMPR_seed{s}']['per_class'][5]['f1'] - all_data[f'K07-CB_seed{s}']['per_class'][5]['f1'] for s in [42, 123, 456]])), 2),
        "tunnel_f1_std": round(float(np.std([all_data[f'K07-CB-CMPR_seed{s}']['per_class'][5]['f1'] - all_data[f'K07-CB_seed{s}']['per_class'][5]['f1'] for s in [42, 123, 456]], ddof=1)), 2),
    })

    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(MULTI_SEED_DIR, "aggregate_results.csv"), index=False)

    # 7. efficiency_results.csv
    # Import verified efficiency metrics
    eff_src = os.path.join(STUDY_DIR, "efficiency_benchmark.csv")
    if os.path.isfile(eff_src):
        df_eff = pd.read_csv(eff_src)
        df_eff.to_csv(os.path.join(MULTI_SEED_DIR, "efficiency_results.csv"), index=False)

    print("Successfully generated all CSV deliverables in multi_seed_validation/!")


if __name__ == "__main__":
    main()
