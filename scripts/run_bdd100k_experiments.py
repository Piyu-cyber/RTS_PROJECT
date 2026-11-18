"""Master experiment execution for BDD100K Real-Time Scheduling.

Runs:
1. Canonical 10-workload simulation across all 7 real-time policies.
2. Physical GPU hardware trace replay on RTX 4060 with captured CUDA Graphs.
3. Conservation invariant checks and performance comparison.

Outputs saved strictly under outputs/bdd100k_migration/
"""

from __future__ import annotations

import os
import sys
import json
import csv
import time
from typing import Dict, List, Any, Tuple
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.phase24.model_profiles import load_model_profiles
from src.phase24.request import Request, RequestOutcome
from src.phase24.discrete_event_simulator import Policy, RealTimeSimulator, SimulationResult
from src.phase24.workload_generator import get_standard_workload_suite, save_workload_definitions
from src.phase24.gpu_trace_replay import GPUTraceReplayer
from src.phase23b.bdd_dataset import load_bdd_split_samples

import argparse

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration", "full_dataset_experiments")


def export_results_csv(results: List[SimulationResult], filepath: str):
    """Export summary metrics table for all policies and workloads."""
    fieldnames = [
        "policy",
        "workload_name",
        "num_offered",
        "num_admitted",
        "num_rejected",
        "num_expired",
        "num_completed",
        "num_completed_on_time",
        "num_completed_late",
        "on_time_ratio_offered",
        "deadline_miss_ratio_completed",
        "rejection_ratio",
        "expiration_ratio",
        "mean_response_time_ms",
        "median_response_time_ms",
        "p95_response_time_ms",
        "p99_response_time_ms",
        "mean_waiting_time_ms",
        "median_waiting_time_ms",
        "gpu_busy_time_ms",
        "total_makespan_ms",
        "gpu_utilization_pct",
        "mean_val_accuracy",
        "mean_test_accuracy",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            d = {k: getattr(r, k) for k in fieldnames}
            writer.writerow(d)
    print(f"Exported results to: {filepath}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run RTS experiments on BDD100K.")
    parser.add_argument("--profiles", type=str, default="", help="Path to model profiles JSON.")
    parser.add_argument("--manifest", type=str, default="", help="Path to canonical test manifest JSON.")
    parser.add_argument("--smoke-test", action="store_true", help="Explicitly run smoke test mode with 225 prototype items.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for results.")
    parser.add_argument("--replay-only", action="store_true", help="Rerun only GPU replay against canonical simulation results.")
    args = parser.parse_args()

    default_output_dir = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration", "corrected_replay") if args.replay_only else OUTPUT_DIR
    out_dir = args.output_dir if args.output_dir else default_output_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.profiles:
        profiles_path = args.profiles
    else:
        full_prof = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration", "full_dataset_model_profiles.json")
        default_prof = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration", "model_profiles.json")
        profiles_path = full_prof if os.path.isfile(full_prof) else default_prof

    profiles = load_model_profiles(profiles_path)
    print("================================================================================")
    print(" Running Full RTS Benchmark Suite on BDD100K Driving Scene Dataset")
    print(f" Loaded {len(profiles)} model profiles from: {profiles_path}")
    print(f" Destination output directory: {out_dir}")
    print("================================================================================")

    # 1. Load BDD100K test samples (Canonical full dataset or explicit smoke test)
    bdd_samples = []
    if args.smoke_test:
        test_txt = os.path.join(PROJECT_ROOT, "data", "bdd100k", "annotations", "test.txt")
        print("\n[SMOKE TEST MODE ENABLED] Using prototype 225-image annotation list.")
        if os.path.isfile(test_txt):
            raw_samples = load_bdd_split_samples(test_txt)
            bdd_samples = [{"image_name": name, "class_id": cid} for name, cid in raw_samples]
            print(f"  Loaded {len(bdd_samples)} prototype BDD100K smoke-test image references.")
    else:
        manifest_path = args.manifest if args.manifest else os.path.join(PROJECT_ROOT, "data", "bdd100k", "annotations", "test_full.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                raw_manifest = json.load(f)
            bdd_samples = [{"image_name": it["image_name"], "class_id": int(it["class_id"])} for it in raw_manifest]
            print(f"\n[CANONICAL FULL-DATASET MODE] Loaded {len(bdd_samples)} test image references from: {manifest_path}")
        else:
            print(f"Warning: Canonical manifest not found at {manifest_path}. Falling back to test.txt.")
            test_txt = os.path.join(PROJECT_ROOT, "data", "bdd100k", "annotations", "test.txt")
            if os.path.isfile(test_txt):
                raw_samples = load_bdd_split_samples(test_txt)
                bdd_samples = [{"image_name": name, "class_id": cid} for name, cid in raw_samples]

    # 2. Generate canonical workloads
    print("\n--- Generating Canonical 10-Workload Suite ---")
    workload_suite = get_standard_workload_suite(bdd_test_samples=bdd_samples)
    workload_json_path = os.path.join(out_dir, "workload_definitions.json")
    save_workload_definitions(workload_suite, workload_json_path)

    # 3. Discrete-Event Simulation across all 7 policies x 10 workloads = 70 runs
    policies = [
        Policy.FIFO_FIXED_DEIT,
        Policy.EDF_FIXED_DEIT,
        Policy.FIFO_FIXED_EVIT,
        Policy.EDF_FIXED_EVIT,
        Policy.EDF_ADMISSION_FIXED_DEIT,
        Policy.EDF_ADAPTIVE,
        Policy.PROPOSED_ADAPTIVE_AC,
    ]

    sim_results: List[SimulationResult] = []
    canonical_sim_csv = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration", "full_dataset_experiments", "simulation_results.csv")

    if args.replay_only and os.path.isfile(canonical_sim_csv):
        print(f"\n[REPLAY-ONLY MODE] Loading canonical simulation results from: {canonical_sim_csv}")
        with open(canonical_sim_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                sim_results.append(
                    SimulationResult(
                        policy=r["policy"],
                        workload_name=r["workload_name"],
                        num_offered=int(r["num_offered"]),
                        num_admitted=int(r["num_admitted"]),
                        num_rejected=int(r["num_rejected"]),
                        num_expired=int(r["num_expired"]),
                        num_completed=int(r["num_completed"]),
                        num_completed_on_time=int(r["num_completed_on_time"]),
                        num_completed_late=int(r["num_completed_late"]),
                        on_time_ratio_offered=float(r["on_time_ratio_offered"]),
                        deadline_miss_ratio_completed=float(r["deadline_miss_ratio_completed"]),
                        rejection_ratio=float(r["rejection_ratio"]),
                        expiration_ratio=float(r["expiration_ratio"]),
                        mean_response_time_ms=float(r["mean_response_time_ms"]),
                        median_response_time_ms=float(r["median_response_time_ms"]),
                        p95_response_time_ms=float(r["p95_response_time_ms"]),
                        p99_response_time_ms=float(r["p99_response_time_ms"]),
                        mean_waiting_time_ms=float(r["mean_waiting_time_ms"]),
                        median_waiting_time_ms=float(r["median_waiting_time_ms"]),
                        gpu_busy_time_ms=float(r["gpu_busy_time_ms"]),
                        total_makespan_ms=float(r["total_makespan_ms"]),
                        gpu_utilization_pct=float(r["gpu_utilization_pct"]),
                        mean_val_accuracy=float(r["mean_val_accuracy"]),
                        mean_test_accuracy=float(r["mean_test_accuracy"]),
                        model_selection_counts={},
                        request_records=[],
                    )
                )
    else:
        simulator = RealTimeSimulator(
            profiles=profiles,
            latency_metric="p95",
            backend="cuda_graph",
            quality_threshold=45.0,
            firm_deadlines=True,
        )

        print("\n--- Running Discrete-Event Simulation (70 configurations) ---")
        for wl_name, wl_requests in workload_suite.items():
            print(f"  Simulating workload: {wl_name} ({len(wl_requests)} requests)...", flush=True)
            for pol in policies:
                res = simulator.run(pol, wl_requests, workload_name=wl_name)
                sim_results.append(res)

        sim_csv_path = os.path.join(out_dir, "simulation_results.csv")
        export_results_csv(sim_results, sim_csv_path)

    # 4. Physical GPU Replay on RTX 4060
    print("\n--- Running Physical GPU Replay on RTX 4060 with Canonical Images ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    replayer = GPUTraceReplayer(
        profiles=profiles,
        device=device,
        quality_threshold=45.0,
        firm_deadlines=True,
    )

    replay_targets = ["periodic_mod_mod", "bursty_high_tight", "overload_tight_deadline"]
    replay_policies = [Policy.FIFO_FIXED_DEIT, Policy.EDF_FIXED_DEIT, Policy.PROPOSED_ADAPTIVE_AC]
    replay_results: List[SimulationResult] = []

    for wl_name in replay_targets:
        wl_requests = workload_suite[wl_name]
        print(f"  GPU Replaying workload: {wl_name} on actual RTX 4060...", flush=True)
        for pol in replay_policies:
            res = replayer.replay_trace(pol, wl_requests, workload_name=f"gpu_{wl_name}")
            replay_results.append(res)

    # Audit image usage
    usage_stats = replayer.get_image_usage_stats()
    print("\n--- Image Resolution & Audit Summary ---")
    print(f"  Total unique images loaded from disk: {usage_stats['unique_images_loaded']}")
    print(f"  Total disk image load operations:    {usage_stats['disk_image_loads']}")
    print(f"  Total synthetic fallback loads:      {usage_stats['synthetic_image_loads']}")
    for sp in usage_stats["sample_image_paths"]:
        print(f"    Sample: {sp}")

    audit_json_path = os.path.join(out_dir, "replay_image_audit.json")
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(usage_stats, f, indent=2)
    print(f"  Audit record saved to: {audit_json_path}")

    # Comparison CSV between simulation and real GPU replay
    hw_rows = []
    for rep in replay_results:
        orig_wl = rep.workload_name.replace("gpu_", "")
        sim_match = next(r for r in sim_results if r.workload_name == orig_wl and r.policy == rep.policy)

        hw_rows.append({
            "workload": orig_wl,
            "policy": rep.policy,
            "sim_ontime_pct": round(sim_match.on_time_ratio_offered * 100.0, 2),
            "gpu_replay_ontime_pct": round(rep.on_time_ratio_offered * 100.0, 2),
            "sim_makespan_ms": sim_match.total_makespan_ms,
            "gpu_makespan_ms": rep.total_makespan_ms,
            "makespan_delta_pct": round(((rep.total_makespan_ms - sim_match.total_makespan_ms) / sim_match.total_makespan_ms) * 100.0, 2),
            "sim_mean_resp_ms": sim_match.mean_response_time_ms,
            "gpu_mean_resp_ms": rep.mean_response_time_ms,
            "empirical_test_accuracy": rep.mean_test_accuracy,
        })

    hw_csv_path = os.path.join(out_dir, "hardware_replay_comparison.csv")
    with open(hw_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(hw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(hw_rows)
    print(f"Exported hardware comparison to: {hw_csv_path}", flush=True)

    print("\n================================================================================")
    print(" BDD100K Real-Time Scheduling Benchmark Suite Completed Successfully!")
    print("================================================================================")


if __name__ == "__main__":
    main()
