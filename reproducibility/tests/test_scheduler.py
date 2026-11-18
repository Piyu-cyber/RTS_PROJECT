"""Tests for priority queueing, admission control, and adaptive dispatch."""

import pytest
from reproducibility.src.scheduler.request import Request, RequestState, RequestOutcome
from reproducibility.src.scheduler.policies import FIFOQueue, EDFQueue
from reproducibility.src.scheduler.admission_control import ModelProfile, AdmissionController
from reproducibility.src.scheduler.adaptive_selector import AdaptiveController


def test_fifo_queue_ordering():
    """Verify FIFO ordering with monotonic insertion counter tie-breaker."""
    q = FIFOQueue()
    r1 = Request("r1", arrival_time_ms=10.0, relative_deadline_ms=20.0)
    r2 = Request("r2", arrival_time_ms=5.0, relative_deadline_ms=30.0)
    r3 = Request("r3", arrival_time_ms=10.0, relative_deadline_ms=15.0)

    q.push(r1)
    q.push(r2)
    q.push(r3)

    assert len(q) == 3
    # r2 arrived first (5.0)
    assert q.pop().request_id == "r2"
    # r1 and r3 arrived at 10.0, but r1 was pushed first
    assert q.pop().request_id == "r1"
    assert q.pop().request_id == "r3"


def test_edf_queue_ordering():
    """Verify EDF ordering with earliest absolute deadline first."""
    q = EDFQueue()
    # r1: arr 0, rel 10 -> abs 10
    r1 = Request("r1", arrival_time_ms=0.0, relative_deadline_ms=10.0)
    # r2: arr 2, rel 5 -> abs 7
    r2 = Request("r2", arrival_time_ms=2.0, relative_deadline_ms=5.0)
    # r3: arr 1, rel 15 -> abs 16
    r3 = Request("r3", arrival_time_ms=1.0, relative_deadline_ms=15.0)

    q.push(r1)
    q.push(r2)
    q.push(r3)

    assert q.pop().request_id == "r2"  # abs deadline 7.0
    assert q.pop().request_id == "r1"  # abs deadline 10.0
    assert q.pop().request_id == "r3"  # abs deadline 16.0


def test_admission_control_heuristic():
    """Verify admission controller rejects requests that cannot meet deadlines."""
    dummy_profiles = {
        "fast": ModelProfile(
            config_id="fast",
            display_name="Fast",
            keep_rate=0.5,
            fuse_token=True,
            parameters_m=22.0,
            macs_g=2.31,
            token_sequence=[28],
            val_top1_accuracy=77.0,
            test_top1_accuracy=76.8,
            test_macro_f1=57.0,
            cuda_graph_latency_ms={"p95": 2.5},
            eager_latency_ms={"p95": 5.0},
            throughput_fps=600.0,
        )
    }

    ac = AdmissionController(profiles=dummy_profiles, latency_metric="p95", backend="cuda_graph")

    # Feasible request: arrival 0.0, rel deadline 5.0, exec time 2.5 <= 5.0
    req_feasible = Request("rf", arrival_time_ms=0.0, relative_deadline_ms=5.0)
    assert ac.is_feasible(req_feasible, current_time_ms=0.0, gpu_busy_until_ms=0.0, queued_requests=[]) is True

    # Infeasible request: arrival 0.0, rel deadline 1.0, exec time 2.5 > 1.0
    req_infeasible = Request("ri", arrival_time_ms=0.0, relative_deadline_ms=1.0)
    assert ac.is_feasible(req_infeasible, current_time_ms=0.0, gpu_busy_until_ms=0.0, queued_requests=[]) is False
