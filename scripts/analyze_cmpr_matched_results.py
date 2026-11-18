"""Post-processing and Statistical Analysis Suite for CMPR Matched Experiments.

Aggregates test predictions across all completed arms and generates:
- outputs/cmpr_matched_training/per_seed_results.csv
- outputs/cmpr_matched_training/aggregate_results.csv
- outputs/cmpr_matched_training/per_class_metrics.csv
- outputs/cmpr_matched_training/margin_analysis.csv
- outputs/cmpr_matched_training/efficiency_benchmarks.csv
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from src.phase23b.bdd_dataset import BDD_CLASSES, NUM_BDD_CLASSES
from src.phase23b.models import MODEL_CONFIGS, build_model

MATCHED_DIR = os.path.join(PROJECT_ROOT, "outputs", "cmpr_matched_training")
CHECKPOINTS_DIR = os.path.join(MATCHED_DIR, "checkpoints")
CM_DIR = os.path.join(MATCHED_DIR, "confusion_matrices")
CONFIGS_DIR = os.path.join(MATCHED_DIR, "training_configurations")
LOGS_DIR = os.path.join(MATCHED_DIR, "training_logs")


def benchmark_efficiency(device: str = "cuda") -> pd.DataFrame:
    """Benchmark MACs, parameters, batch-1 latency (p50, p95, p99), and throughput."""
    dev = torch.device(device)
    dummy_input = torch.randn(1, 3, 224, 224, device=dev)
    batch_input = torch.randn(64, 3, 224, 224, device=dev)

    models = {
        "M0_DeiT_Baseline": ("deit_small_baseline", 4.60),
        "M1_EViT_Baseline": ("evit_small_k05_fuse", 2.31),
        "M2_EViT_CB_CE": ("evit_small_k05_fuse", 2.31),
        "M3_EViT_Standard_KD": ("evit_small_k05_fuse", 2.31),
        "M4_EViT_DKD": ("evit_small_k05_fuse", 2.31),
        "M5_EViT_Feature_Distill": ("evit_small_k05_fuse", 2.31),
        "M6_EViT_CMPR": ("evit_small_k05_fuse", 2.31),
    }

    rows = []
    print("[*] Benchmarking hardware efficiency (RTX 4060 GPU)...", flush=True)

    for arm_label, (arch, published_macs) in models.items():
        model = build_model(arch, num_classes=NUM_BDD_CLASSES, pretrained=False).to(dev)
        model.eval()

        # Parameter count
        params_total = sum(p.numel() for p in model.parameters()) / 1e6

        # Warmup
        with torch.no_grad():
            for _ in range(50):
                _ = model(dummy_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()

        # Batch-1 Latency (1,000 iterations)
        latencies_ms = []
        with torch.no_grad():
            for _ in range(1000):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        lat_arr = np.array(latencies_ms)
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_lat = float(np.mean(lat_arr))

        # Throughput (batch 64)
        with torch.no_grad():
            for _ in range(20):
                _ = model(batch_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()

            t_start = time.perf_counter()
            iters = 50
            for _ in range(iters):
                _ = model(batch_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()

        throughput = (iters * 64) / (t_end - t_start)

        rows.append({
            "arm_label": arm_label,
            "architecture": arch,
            "parameters_m": round(params_total, 2),
            "macs_g": round(published_macs, 2),
            "latency_p50_ms": round(p50, 3),
            "latency_p95_ms": round(p95, 3),
            "latency_p99_ms": round(p99, 3),
            "latency_mean_ms": round(mean_lat, 3),
            "throughput_samples_s": round(throughput, 1),
            "real_time_deadline_met_5ms": p95 <= 5.0,
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(MATCHED_DIR, "efficiency_benchmarks.csv")
    df.to_csv(out_csv, index=False)
    print(f"[+] Saved efficiency benchmarks to {out_csv}", flush=True)
    return df


def aggregate_experiment_results() -> None:
    """Aggregates all confusion matrices, metrics, and logs into deliverable CSVs."""
    cm_files = sorted(glob.glob(os.path.join(CM_DIR, "*_confusion_matrix.json")))
    if not cm_files:
        print("[!] No completed confusion matrix JSON files found in", CM_DIR)
        return

    per_seed_rows = []
    per_class_rows = []

    for cm_file in cm_files:
        with open(cm_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        run_name = data["run_name"]
        parts = run_name.split("_")
        arm_id = parts[0]
        seed_str = parts[1] if len(parts) > 1 else "seed42"
        seed = int(seed_str.replace("seed", "")) if "seed" in seed_str else 42

        # Check for config file to retrieve training time and checkpoint hash
        cfg_path = os.path.join(CONFIGS_DIR, f"{run_name}_config.json")
        cfg = {}
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        # Check for training history to retrieve best epoch and val f1
        hist_path = os.path.join(LOGS_DIR, f"{run_name}_history.json")
        best_epoch = 1
        best_val_f1 = 0.0
        if os.path.isfile(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            if hist:
                best_entry = max(hist, key=lambda x: x.get("val_macro_f1", 0.0))
                best_epoch = best_entry.get("epoch", 1)
                best_val_f1 = best_entry.get("val_macro_f1", 0.0)

        cm = np.array(data["confusion_matrix"])
        total_samples = int(np.sum(cm))
        correct_samples = int(np.trace(cm))
        top1_acc = (correct_samples / total_samples * 100.0) if total_samples > 0 else 0.0

        per_class = data["per_class"]
        f1_list = [pc["f1"] for pc in per_class]
        supp_list = [pc["support"] for pc in per_class]
        macro_f1 = float(np.mean(f1_list))
        weighted_f1 = float(np.average(f1_list, weights=supp_list))

        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{run_name}_best.pth")

        per_seed_rows.append({
            "arm_id": arm_id,
            "run_name": run_name,
            "seed": seed,
            "arch_config": cfg.get("arch_config", "unknown"),
            "loss_type": cfg.get("loss_type", "unknown"),
            "selected_epoch": best_epoch,
            "best_val_macro_f1": round(best_val_f1, 2),
            "test_top1_accuracy": round(top1_acc, 2),
            "test_macro_f1": round(macro_f1, 2),
            "test_weighted_f1": round(weighted_f1, 2),
            "residential_recall": round(per_class[2]["recall"], 2),
            "residential_f1": round(per_class[2]["f1"], 2),
            "parking_lot_recall": round(per_class[3]["recall"], 2),
            "parking_lot_f1": round(per_class[3]["f1"], 2),
            "highway_recall": round(per_class[1]["recall"], 2),
            "city_street_recall": round(per_class[0]["recall"], 2),
            "checkpoint_path": ckpt_path,
        })

        for pc in per_class:
            per_class_rows.append({
                "arm_id": arm_id,
                "run_name": run_name,
                "seed": seed,
                "class_id": pc["class_id"],
                "class_name": pc["class_name"],
                "support": pc["support"],
                "recall": pc["recall"],
                "precision": pc["precision"],
                "f1": pc["f1"],
            })

    # Save per_seed_results.csv
    df_seed = pd.DataFrame(per_seed_rows)
    seed_csv = os.path.join(MATCHED_DIR, "per_seed_results.csv")
    df_seed.to_csv(seed_csv, index=False)
    print(f"[+] Saved per-seed results to {seed_csv}", flush=True)

    # Save per_class_metrics.csv
    df_pc = pd.DataFrame(per_class_rows)
    pc_csv = os.path.join(MATCHED_DIR, "per_class_metrics.csv")
    df_pc.to_csv(pc_csv, index=False)
    print(f"[+] Saved per-class metrics to {pc_csv}", flush=True)

    # Compute aggregate results across seeds
    agg_rows = []
    for arm_id, group in df_seed.groupby("arm_id"):
        agg_rows.append({
            "arm_id": arm_id,
            "arch_config": group["arch_config"].iloc[0],
            "loss_type": group["loss_type"].iloc[0],
            "num_seeds": len(group),
            "mean_top1_accuracy": round(group["test_top1_accuracy"].mean(), 2),
            "std_top1_accuracy": round(group["test_top1_accuracy"].std(), 2) if len(group) > 1 else 0.0,
            "mean_macro_f1": round(group["test_macro_f1"].mean(), 2),
            "std_macro_f1": round(group["test_macro_f1"].std(), 2) if len(group) > 1 else 0.0,
            "mean_weighted_f1": round(group["test_weighted_f1"].mean(), 2),
            "std_weighted_f1": round(group["test_weighted_f1"].std(), 2) if len(group) > 1 else 0.0,
            "mean_residential_recall": round(group["residential_recall"].mean(), 2),
            "std_residential_recall": round(group["residential_recall"].std(), 2) if len(group) > 1 else 0.0,
            "mean_parking_lot_recall": round(group["parking_lot_recall"].mean(), 2),
            "std_parking_lot_recall": round(group["parking_lot_recall"].std(), 2) if len(group) > 1 else 0.0,
        })

    df_agg = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(MATCHED_DIR, "aggregate_results.csv")
    df_agg.to_csv(agg_csv, index=False)
    print(f"[+] Saved aggregate results to {agg_csv}", flush=True)


def compute_margin_analysis() -> None:
    """Computes sample-level margin distributions and flips between M0 teacher and student arms."""
    cm_dir = os.path.join(MATCHED_DIR, "confusion_matrices")
    m0_path = os.path.join(cm_dir, "M0_seed42_confusion_matrix.json")
    if not os.path.isfile(m0_path):
        return

    with open(m0_path, "r", encoding="utf-8") as f:
        m0_data = json.load(f)

    # Check if student runs have detailed predictions
    student_files = sorted(glob.glob(os.path.join(cm_dir, "M*_confusion_matrix.json")))
    margin_rows = []

    for s_file in student_files:
        if "M0_" in os.path.basename(s_file):
            continue
        with open(s_file, "r", encoding="utf-8") as f:
            s_data = json.load(f)

        s_run = s_data.get("run_name", "unknown")
        cm_s = np.array(s_data["confusion_matrix"])
        cm_0 = np.array(m0_data["confusion_matrix"])

        for c in range(NUM_BDD_CLASSES):
            c_name = BDD_CLASSES[c]
            n_tot = int(cm_0[c].sum())
            t_corr = int(cm_0[c, c])
            s_corr = int(cm_s[c, c])

            t_rec = (t_corr / n_tot * 100.0) if n_tot > 0 else 0.0
            s_rec = (s_corr / n_tot * 100.0) if n_tot > 0 else 0.0
            rec_delta = s_rec - t_rec

            # Confusion destinations for minority samples
            dest_city = int(cm_s[c, 0]) if c != 0 else 0
            dest_hwy = int(cm_s[c, 1]) if c != 1 else 0

            margin_rows.append({
                "student_run": s_run,
                "class_id": c,
                "class_name": c_name,
                "total_support": n_tot,
                "teacher_correct": t_corr,
                "student_correct": s_corr,
                "teacher_recall": round(t_rec, 2),
                "student_recall": round(s_rec, 2),
                "recall_delta_pp": round(rec_delta, 2),
                "samples_to_city_street": dest_city,
                "samples_to_highway": dest_hwy,
            })

    if margin_rows:
        df_m = pd.DataFrame(margin_rows)
        m_csv = os.path.join(MATCHED_DIR, "margin_analysis.csv")
        df_m.to_csv(m_csv, index=False)
        print(f"[+] Saved margin analysis to {m_csv}", flush=True)


def main():
    aggregate_experiment_results()
    compute_margin_analysis()
    eff_csv = os.path.join(MATCHED_DIR, "efficiency_benchmarks.csv")
    if not os.path.isfile(eff_csv):
        benchmark_efficiency()
    else:
        print(f"[*] Efficiency benchmarks already exist at {eff_csv}. Skipping benchmark.")


if __name__ == "__main__":
    main()
