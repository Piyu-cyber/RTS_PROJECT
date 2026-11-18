"""Tests for real-time systems conservation equations across simulation outputs."""

import csv
import os
import pytest
from reproducibility.src.scheduler.simulator import verify_conservation_laws, SimulationResult

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIMULATION_CSV = os.path.join(PROJECT_ROOT, "reproducibility", "artifacts", "scheduler", "simulation_results.csv")


def test_conservation_laws_on_dummy():
    """Verify conservation laws pass on mathematically valid simulation result."""
    res = SimulationResult(
        policy="EDF_Adaptive_AC",
        workload_name="test_workload",
        num_offered=100,
        num_admitted=85,
        num_rejected=15,
        num_expired=5,
        num_completed=80,
        num_completed_on_time=75,
        num_completed_late=5,
        on_time_ratio_offered=0.75,
        deadline_miss_ratio_completed=0.0625,
        rejection_ratio=0.15,
        expiration_ratio=0.05,
        mean_response_time_ms=3.5,
        median_response_time_ms=3.2,
        p95_response_time_ms=4.1,
        p99_response_time_ms=4.5,
        mean_waiting_time_ms=0.5,
        median_waiting_time_ms=0.2,
        gpu_busy_time_ms=250.0,
        total_makespan_ms=350.0,
        gpu_utilization_pct=71.4,
        mean_val_accuracy=77.5,
        mean_test_accuracy=77.1,
        model_selection_counts={},
        request_records=[],
    )

    assert verify_conservation_laws(res) is True


def test_conservation_laws_violation_detection():
    """Verify conservation laws raise AssertionError when invariant is violated."""
    res = SimulationResult(
        policy="EDF_Adaptive_AC",
        workload_name="bad_workload",
        num_offered=100,
        num_admitted=85,
        num_rejected=10,  # 85 + 10 = 95 != 100!
        num_expired=5,
        num_completed=80,
        num_completed_on_time=75,
        num_completed_late=5,
        on_time_ratio_offered=0.75,
        deadline_miss_ratio_completed=0.0625,
        rejection_ratio=0.10,
        expiration_ratio=0.05,
        mean_response_time_ms=3.5,
        median_response_time_ms=3.2,
        p95_response_time_ms=4.1,
        p99_response_time_ms=4.5,
        mean_waiting_time_ms=0.5,
        median_waiting_time_ms=0.2,
        gpu_busy_time_ms=250.0,
        total_makespan_ms=350.0,
        gpu_utilization_pct=71.4,
        mean_val_accuracy=77.5,
        mean_test_accuracy=77.1,
        model_selection_counts={},
        request_records=[],
    )

    with pytest.raises(AssertionError):
        verify_conservation_laws(res)


def test_conservation_laws_across_all_simulation_results():
    """Verify all rows in the generated simulation_results.csv satisfy all 4 equations."""
    if not os.path.isfile(SIMULATION_CSV):
        pytest.skip(f"Simulation CSV not yet generated at {SIMULATION_CSV}")

    with open(SIMULATION_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 70, f"Expected 70 simulation configurations, found {len(rows)}"

    for r in rows:
        offered = int(r["num_offered"])
        admitted = int(r["num_admitted"])
        rejected = int(r["num_rejected"])
        expired = int(r["num_expired"])
        completed = int(r["num_completed"])
        on_time = int(r["num_completed_on_time"])
        late = int(r["num_completed_late"])

        # Law 1: offered == on_time + late + rejected + expired
        assert offered == on_time + late + rejected + expired, f"Law 1 failed in {r['workload_name']} / {r['policy']}"
        # Law 2: completed == on_time + late
        assert completed == on_time + late, f"Law 2 failed in {r['workload_name']} / {r['policy']}"
        # Law 3: admitted == completed + expired
        assert admitted == completed + expired, f"Law 3 failed in {r['workload_name']} / {r['policy']}"
        # Law 4: offered == admitted + rejected
        assert offered == admitted + rejected, f"Law 4 failed in {r['workload_name']} / {r['policy']}"
