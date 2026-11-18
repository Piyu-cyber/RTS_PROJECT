"""Table generation module for IEEE manuscript (Tables I to VI)."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Any


def generate_table1_latex(input_csv: str, output_tex: str) -> None:
    """Generate Table I: 11-Model Cross-Keep-Rate Comparison."""
    with open(input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Perception Performance Across Pruning Keep-Rates and Objectives on BDD100K Driving Scenes}",
        r"\label{tab:cross_keep_rate}",
        r"\small",
        r"\begin{tabular}{lccccccccc}",
        r"\hline",
        r"\textbf{Model Architecture} & \textbf{Keep-Rate} & \textbf{GMACs} & \textbf{Top-1 Acc (\%)} & \textbf{95\% CI} & \textbf{Macro-F1 (\%)} & \textbf{95\% CI} & \textbf{Weighted-F1 (\%)} & \textbf{City Rec (\%)} & \textbf{Tunnel F1 (\%)} \\",
        r"\hline",
    ]

    for r in rows:
        m = r["model_id"]
        k = r["keep_rate"]
        g = r["gmacs"]
        acc = f"{float(r['top1_accuracy']):.2f}"
        acc_ci = f"[{float(r['top1_ci_low']):.1f}, {float(r['top1_ci_high']):.1f}]"
        mf1 = f"{float(r['macro_f1']):.2f}"
        mf1_ci = f"[{float(r['macro_f1_ci_low']):.1f}, {float(r['macro_f1_ci_high']):.1f}]"
        wf1 = f"{float(r['weighted_f1']):.2f}"
        city = f"{float(r['city_street_recall']):.2f}"
        tun = f"{float(r['tunnel_f1']):.2f}"

        bold_m = rf"\textbf{{{m}}}" if "CMPR" in m else m
        lines.append(f"{bold_m} & {k} & {g} & {acc} & {acc_ci} & {mf1} & {mf1_ci} & {wf1} & {city} & {tun} \\\\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    os.makedirs(os.path.dirname(os.path.abspath(output_tex)), exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_table2_latex(per_seed_csv: str, aggregate_csv: str, output_tex: str) -> None:
    """Generate Table II: Multi-Seed Reproducibility Analysis."""
    with open(per_seed_csv, "r", encoding="utf-8") as f:
        per_seed_rows = list(csv.DictReader(f))
    with open(aggregate_csv, "r", encoding="utf-8") as f:
        agg_rows = list(csv.DictReader(f))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multi-Seed Validation and Statistical Reliability ($k=0.7$)}",
        r"\label{tab:multi_seed}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Seed 42} & \textbf{Seed 123} & \textbf{Seed 456} & \textbf{Mean $\pm$ Std} \\",
        r"\hline",
        r"\multicolumn{5}{l}{\textit{K07-CB Baseline:}} \\",
    ]

    cb_seeds = {int(r["seed"]): r for r in per_seed_rows if r["model_id"] == "K07-CB"}
    cmpr_seeds = {int(r["seed"]): r for r in per_seed_rows if r["model_id"] == "K07-CB-CMPR"}
    agg_dict = {r["model_id"]: r for r in agg_rows}

    if 42 in cb_seeds and 123 in cb_seeds and 456 in cb_seeds:
        lines.append(
            f"Top-1 Acc (\\%) & {cb_seeds[42]['top1_accuracy']} & {cb_seeds[123]['top1_accuracy']} & "
            f"{cb_seeds[456]['top1_accuracy']} & {agg_dict['K07-CB']['top1_accuracy_str']} \\\\"
        )
        lines.append(
            f"Macro-F1 (\\%) & {cb_seeds[42]['macro_f1']} & {cb_seeds[123]['macro_f1']} & "
            f"{cb_seeds[456]['macro_f1']} & {agg_dict['K07-CB']['macro_f1_str']} \\\\"
        )
        lines.append(
            f"Weighted-F1 (\\%) & {cb_seeds[42]['weighted_f1']} & {cb_seeds[123]['weighted_f1']} & "
            f"{cb_seeds[456]['weighted_f1']} & {agg_dict['K07-CB']['weighted_f1_str']} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\multicolumn{5}{l}{\textit{K07-CB-CMPR (Proposed):}} \\")

    if 42 in cmpr_seeds and 123 in cmpr_seeds and 456 in cmpr_seeds:
        lines.append(
            f"Top-1 Acc (\\%) & {cmpr_seeds[42]['top1_accuracy']} & {cmpr_seeds[123]['top1_accuracy']} & "
            f"{cmpr_seeds[456]['top1_accuracy']} & {agg_dict['K07-CB-CMPR']['top1_accuracy_str']} \\\\"
        )
        lines.append(
            f"Macro-F1 (\\%) & {cmpr_seeds[42]['macro_f1']} & {cmpr_seeds[123]['macro_f1']} & "
            f"{cmpr_seeds[456]['macro_f1']} & {agg_dict['K07-CB-CMPR']['macro_f1_str']} \\\\"
        )
        lines.append(
            f"Weighted-F1 (\\%) & {cmpr_seeds[42]['weighted_f1']} & {cmpr_seeds[123]['weighted_f1']} & "
            f"{cmpr_seeds[456]['weighted_f1']} & {agg_dict['K07-CB-CMPR']['weighted_f1_str']} \\\\"
        )

    lines.append(r"\hline")
    delta_key = [k for k in agg_dict.keys() if "DELTA" in k]
    if delta_key:
        d = agg_dict[delta_key[0]]
        lines.append(
            f"\\textbf{{Paired Delta $\\Delta$}} & "
            f"{float(cmpr_seeds[42]['top1_accuracy']) - float(cb_seeds[42]['top1_accuracy']):+.2f} & "
            f"{float(cmpr_seeds[123]['top1_accuracy']) - float(cb_seeds[123]['top1_accuracy']):+.2f} & "
            f"{float(cmpr_seeds[456]['top1_accuracy']) - float(cb_seeds[456]['top1_accuracy']):+.2f} & "
            f"\\textbf{{{d['top1_accuracy_str']}}} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    os.makedirs(os.path.dirname(os.path.abspath(output_tex)), exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_table3_latex(input_csv: str, output_tex: str) -> None:
    """Generate Table III: Latency & Throughput Profiles on RTX 4060."""
    with open(input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Latency and Throughput Benchmarks on NVIDIA RTX 4060 Laptop GPU}",
        r"\label{tab:hardware_profiles}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{Tokens} & \textbf{Eager Mean} & \textbf{Eager p95} & \textbf{Graph Mean} & \textbf{Graph p95} & \textbf{FPS} \\",
        r"\hline",
    ]

    for r in rows:
        m = r.get("model_name", r.get("model_id", "Model"))
        tokens = r.get("token_count", r.get("tokens", "197"))
        e_mean = f"{float(r.get('eager_mean_ms', 0.0)):.2f}"
        e_p95 = f"{float(r.get('eager_p95_ms', 0.0)):.2f}"
        g_mean = f"{float(r.get('cuda_graph_mean_ms', 0.0)):.2f}"
        g_p95 = f"{float(r.get('cuda_graph_p95_ms', 0.0)):.2f}"
        fps = f"{float(r.get('throughput_fps', 0.0)):.1f}"
        lines.append(f"{m} & {tokens} & {e_mean} & {e_p95} & {g_mean} & {g_p95} & {fps} \\\\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    os.makedirs(os.path.dirname(os.path.abspath(output_tex)), exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_table6_latex(input_csv: str, output_tex: str) -> None:
    """Generate Table VI: Hardware Replay Comparison."""
    with open(input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Discrete-Event Simulation vs Physical GPU Hardware Trace Replay Validation}",
        r"\label{tab:hardware_replay}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"\textbf{Workload} & \textbf{Policy} & \textbf{Sim On-Time} & \textbf{GPU Replay} & \textbf{Makespan $\Delta$} & \textbf{Empirical Acc} \\",
        r"\hline",
    ]

    for r in rows:
        w = r["workload"].replace("_", r"\_")
        p = r["policy"].replace("_", r"\_")
        sim_ot = f"{float(r['sim_ontime_pct']):.1f}\\%"
        gpu_ot = f"{float(r['gpu_replay_ontime_pct']):.1f}\\%"
        delta = f"{float(r['makespan_delta_pct']):+.2f}\\%"
        acc = f"{float(r['empirical_test_accuracy']):.1f}\\%"
        lines.append(f"{w} & {p} & {sim_ot} & {gpu_ot} & {delta} & {acc} \\\\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    os.makedirs(os.path.dirname(os.path.abspath(output_tex)), exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
