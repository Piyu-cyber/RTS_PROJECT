"""Figure generation module for IEEE manuscript (Figures 2 to 6)."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_figure2_pareto(cross_results_csv: str, output_path: str) -> None:
    """Figure 2: Perception-efficiency Pareto front (Top-1 Acc & Macro-F1 vs GMACs)."""
    with open(cross_results_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Markers and color schemes
    colors = {
        "DeiT": "#2b5c8f",
        "CE": "#7f7f7f",
        "CB": "#d95f02",
        "CMPR": "#7570b3",
        "CB-CMPR": "#1b9e77",
    }

    for r in rows:
        m = r["model_id"]
        gmacs = float(r["gmacs"])
        acc = float(r["top1_accuracy"])
        mf1 = float(r["macro_f1"])

        if m == "DeiT":
            c, marker = colors["DeiT"], "s"
        elif "CB-CMPR" in m:
            c, marker = colors["CB-CMPR"], "*"
        elif "CMPR" in m:
            c, marker = colors["CMPR"], "^"
        elif "CB" in m:
            c, marker = colors["CB"], "D"
        else:
            c, marker = colors["CE"], "o"

        size = 140 if marker == "*" else 70

        ax1.scatter(gmacs, acc, color=c, marker=marker, s=size, edgecolors="k", linewidth=0.6, zorder=5)
        ax1.annotate(m, (gmacs, acc), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5)

        ax2.scatter(gmacs, mf1, color=c, marker=marker, s=size, edgecolors="k", linewidth=0.6, zorder=5)
        ax2.annotate(m, (gmacs, mf1), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5)

    ax1.set_xlabel("Computational Complexity (GMACs)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Top-1 Accuracy vs GMACs", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Computational Complexity (GMACs)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Macro-F1 Score (%)", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Macro-F1 vs GMACs", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_figure3_latency(latency_csv: str, output_path: str) -> None:
    """Figure 3: Eager PyTorch vs Static CUDA Graph Latency Distributions."""
    with open(latency_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    models = [r.get("model_name", r.get("model_id", f"M{i}")) for i, r in enumerate(rows)]
    eager_p95 = [float(r.get("eager_p95_ms", 0.0)) for r in rows]
    graph_p95 = [float(r.get("cuda_graph_p95_ms", 0.0)) for r in rows]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    rects1 = ax.bar(x - width/2, eager_p95, width, label="Eager PyTorch (p95)", color="#e7298a", alpha=0.85)
    rects2 = ax.bar(x + width/2, graph_p95, width, label="CUDA Graph Replay (p95)", color="#1b9e77", alpha=0.85)

    ax.set_ylabel("Execution Time (ms)", fontsize=10, fontweight="bold")
    ax.set_title("Inference Latency Bounds on NVIDIA RTX 4060 GPU", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5, rotation=15)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_figure4_scheduling_ontime(simulation_csv: str, output_path: str) -> None:
    """Figure 4: On-time completion ratio across 10 workloads for 7 policies."""
    with open(simulation_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    workloads = sorted(list(set(r["workload_name"] for r in rows)))
    policies = [
        "FIFO_fixed_DeiT",
        "EDF_fixed_DeiT",
        "FIFO_fixed_EViT",
        "EDF_fixed_EViT",
        "EDF_AC_fixed_DeiT",
        "EDF_Adaptive",
        "EDF_Adaptive_AC",
    ]

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)

    x = np.arange(len(workloads))
    width = 0.11

    colors = ["#999999", "#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#e41a1c", "#1b9e77"]

    for i, p in enumerate(policies):
        p_rows = {r["workload_name"]: float(r["on_time_ratio_offered"]) * 100.0 for r in rows if r["policy"] == p}
        vals = [p_rows.get(w, 0.0) for w in workloads]
        ax.bar(x + (i - 3) * width, vals, width, label=p, color=colors[i % len(colors)], alpha=0.9)

    ax.set_ylabel("On-Time Completion Ratio (%)", fontsize=10, fontweight="bold")
    ax.set_title("Real-Time QoS Across 10 Autonomous Driving Workload Traces", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    clean_labels = [w.replace("_", "\n") for w in workloads]
    ax.set_xticklabels(clean_labels, fontsize=8)
    ax.legend(frameon=True, fontsize=8, ncol=4, loc="upper right")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_figure5_stacked_outcomes(simulation_csv: str, output_path: str) -> None:
    """Figure 5: Stacked request outcomes across overload workloads."""
    with open(simulation_csv, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if "overload" in r["workload_name"]]

    policies = sorted(list(set(r["policy"] for r in rows)))
    # Pick overload_tight_deadline
    target_rows = {r["policy"]: r for r in rows if r["workload_name"] == "overload_tight_deadline"}

    on_time = [float(target_rows[p]["num_completed_on_time"]) for p in policies]
    late = [float(target_rows[p]["num_completed_late"]) for p in policies]
    expired = [float(target_rows[p]["num_expired"]) for p in policies]
    rejected = [float(target_rows[p]["num_rejected"]) for p in policies]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

    y = np.arange(len(policies))
    p1 = ax.barh(y, on_time, color="#2ca02c", label="On-Time Completed")
    p2 = ax.barh(y, late, left=on_time, color="#d62728", label="Late Completed")
    p3 = ax.barh(y, expired, left=np.array(on_time)+np.array(late), color="#ff7f0e", label="Queue Expired")
    p4 = ax.barh(y, rejected, left=np.array(on_time)+np.array(late)+np.array(expired), color="#7f7f7f", label="Admission Rejected")

    ax.set_xlabel("Number of Requests (Total Offered = 100)", fontsize=10, fontweight="bold")
    ax.set_title("Request Terminal Outcomes under Heavy Overload Stress (Overload-Tight)", fontsize=11, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(policies, fontsize=9)
    ax.legend(frameon=True, fontsize=8.5, loc="lower right")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_figure6_hardware_replay(replay_csv: str, output_path: str) -> None:
    """Figure 6: Simulation vs Hardware Replay Fidelity Comparison."""
    with open(replay_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    configs = [f"{r['workload'][:7]}-{r['policy'].split('_')[-1]}" for r in rows]
    sim_ot = [float(r["sim_ontime_pct"]) for r in rows]
    gpu_ot = [float(r["gpu_replay_ontime_pct"]) for r in rows]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.bar(x - width/2, sim_ot, width, label="Simulation Model", color="#3182bd", alpha=0.85)
    ax.bar(x + width/2, gpu_ot, width, label="Physical GPU Trace Replay", color="#31a354", alpha=0.85)

    ax.set_ylabel("On-Time Ratio (%)", fontsize=10, fontweight="bold")
    ax.set_title("Fidelity Validation: Discrete-Event Simulation vs Physical GPU Replay", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8.5, rotation=25, ha="right")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
