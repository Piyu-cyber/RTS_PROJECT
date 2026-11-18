"""Verify the real-time systems conservation equations across all simulation results."""

import csv
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
path = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration", "full_dataset_experiments", "simulation_results.csv")

if not os.path.isfile(path):
    print(f"Error: {path} not found.")
    sys.exit(1)

with open(path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Validating {len(rows)} simulation result rows from: {path}")
errors = 0

for r in rows:
    offered = int(r["num_offered"])
    admitted = int(r["num_admitted"])
    rejected = int(r["num_rejected"])
    expired = int(r["num_expired"])
    completed = int(r["num_completed"])
    on_time = int(r["num_completed_on_time"])
    late = int(r["num_completed_late"])

    # Invariant 1: offered == completed_on_time + completed_late + rejected + expired
    if offered != on_time + late + rejected + expired:
        print(f"Invariant 1 violated in {r['workload_name']} / {r['policy']}: {offered} != {on_time + late + rejected + expired}")
        errors += 1

    # Invariant 2: completed == on_time + late
    if completed != on_time + late:
        print(f"Invariant 2 violated in {r['workload_name']} / {r['policy']}: {completed} != {on_time + late}")
        errors += 1

    # Invariant 3: admitted == completed + expired
    if admitted != completed + expired:
        print(f"Invariant 3 violated in {r['workload_name']} / {r['policy']}: {admitted} != {completed + expired}")
        errors += 1

    # Invariant 4: offered == admitted + rejected
    if offered != admitted + rejected:
        print(f"Invariant 4 violated in {r['workload_name']} / {r['policy']}: {offered} != {admitted + rejected}")
        errors += 1

if errors == 0:
    print(f"PASS: All {len(rows)} configurations strictly satisfy all 4 conservation equations (0 errors)!")
else:
    print(f"FAIL: Found {errors} accounting violations!")
    sys.exit(1)
