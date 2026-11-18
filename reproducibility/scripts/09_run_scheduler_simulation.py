"""Script 09: Discrete-event real-time scheduling simulation (70 configurations)."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reproducibility.src.scheduler.admission_control import load_model_profiles
from reproducibility.src.scheduler.workload_generator import get_standard_workload_suite
from reproducibility.src.scheduler.simulator import (
    Policy,
    RealTimeSimulator,
    export_simulation_results_csv,
    verify_conservation_laws,
)


def main():
    parser = argparse.ArgumentParser(description="Run discrete-event real-time simulation study.")
    parser.add_argument(
        "--profiles-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration", "full_dataset_model_profiles.json"),
        help="Path to validated model profiles JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "reproducibility", "artifacts", "scheduler", "simulation_results.csv"),
        help="Path to output simulation results CSV.",
    )
    args = parser.parse_args()

    # Fallback to model_profiles.json if full_dataset_model_profiles.json is absent
    if not os.path.isfile(args.profiles_path):
        args.profiles_path = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration", "model_profiles.json")

    print("=" * 80)
    print("STEP 09: DISCRETE-EVENT REAL-TIME SCHEDULING SIMULATION")
    print(f"Model profiles: {args.profiles_path}")
    print(f"Output CSV:     {args.output_csv}")
    print("=" * 80)

    profiles = load_model_profiles(args.profiles_path)
    simulator = RealTimeSimulator(
        profiles=profiles,
        latency_metric="p95",
        backend="cuda_graph",
        quality_threshold=45.0,
        firm_deadlines=True,
    )

    workload_suite = get_standard_workload_suite()
    policies = list(Policy)

    total_runs = len(workload_suite) * len(policies)
    print(f"Executing {total_runs} simulation configurations ({len(workload_suite)} workloads x {len(policies)} policies)...")

    results = []
    conservation_errors = 0

    for w_name, trace in workload_suite.items():
        for p in policies:
            res = simulator.run(policy=p, workload=trace, workload_name=w_name)
            try:
                verify_conservation_laws(res)
            except AssertionError as e:
                print(f"[ERROR] Conservation law violation: {e}")
                conservation_errors += 1
            results.append(res)

    print(f"\nCompleted {len(results)} simulation runs.")
    if conservation_errors == 0:
        print("[PASS] All 70 configurations strictly satisfy all 4 conservation equations (0 errors)!")
    else:
        print(f"[FAIL] Found {conservation_errors} accounting errors.")
        sys.exit(1)

    export_simulation_results_csv(results, args.output_csv)
    print(f"Exported simulation results table to: {args.output_csv}")

    # Summary preview of proposed adaptive AC policy vs baselines on stress workloads
    print("\n[Preview: On-Time Completion Ratio (%) on Stress Workloads]")
    stress_w = ["bursty_high_tight", "overload_tight_deadline"]
    print(f"{'Workload':<25} {'FIFO-DeiT':<12} {'EDF-DeiT':<12} {'EDF-Adaptive-AC':<15}")
    print("-" * 65)
    for w in stress_w:
        fifo_r = next(r for r in results if r.workload_name == w and r.policy == Policy.FIFO_FIXED_DEIT.value)
        edf_r = next(r for r in results if r.workload_name == w and r.policy == Policy.EDF_FIXED_DEIT.value)
        prop_r = next(r for r in results if r.workload_name == w and r.policy == Policy.PROPOSED_ADAPTIVE_AC.value)
        print(f"{w:<25} {fifo_r.on_time_ratio_offered * 100:>8.1f}%    {edf_r.on_time_ratio_offered * 100:>8.1f}%    {prop_r.on_time_ratio_offered * 100:>11.1f}%")
    print("-" * 65)

    print("\nSimulation study completed successfully (exit code 0).\n")


if __name__ == "__main__":
    main()
