"""Phase 24B Canonical Experiment Engine and Robustness Suite.

Implements rigorous discrete-event simulation, multi-seed workload generation,
sensitivity analysis, and physical GPU replay with strict conservation checks.
"""

from __future__ import annotations

import os
import json
import random
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.phase24.model_profiles import ModelProfile, load_model_profiles
from src.phase24.request import Request, RequestOutcome, RequestState
from src.phase24.priority_queue import FIFOQueue, EDFQueue
from src.phase24.admission_control import AdmissionController
from src.phase24.adaptive_controller import AdaptiveController
from src.phase24.discrete_event_simulator import Policy, EventType


@dataclass
class CanonicalResult:
    """Canonical aggregated metrics for a policy/workload run with strict accounting."""
    policy: str
    workload_name: str
    workload_category: str
    load_level: str
    deadline_tightness: str
    seed: int
    experiment_type: str            # SIMULATOR_P95, SIMULATOR_MEDIAN, SIMULATOR_MEAN, GPU_REPLAY
    
    # Conservation counts
    num_offered: int
    num_admitted: int
    num_rejected: int
    num_expired: int
    num_completed: int
    num_completed_on_time: int
    num_completed_late: int
    
    # Ratios (strictly defined)
    on_time_ratio_offered: float    # on_time / offered
    on_time_ratio_accepted: float   # on_time / admitted
    deadline_miss_ratio_completed: float # late / completed
    rejection_ratio: float          # rejected / offered
    expiration_ratio: float         # expired / offered
    
    # Response and waiting time distributions (ms)
    mean_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    mean_waiting_time_ms: float
    median_waiting_time_ms: float
    mean_queue_length: float
    max_queue_length: int
    
    # Execution & system utilization
    total_gpu_busy_time_ms: float
    total_makespan_ms: float
    gpu_utilization_pct: float
    
    # Accuracy quantities (strictly separated)
    mean_val_accuracy: float                 # Gating/validation metric
    synthetic_expected_test_accuracy: float  # Model-weighted dataset expectation over completed requests
    empirical_test_accuracy: Optional[float] # Actual top-1 accuracy on real test images (GPU replay only)
    
    # Model configuration dispatches
    count_deit_small: int
    count_evit_k09: int
    count_evit_k07: int
    count_evit_k05: int
    
    # Detailed records
    request_records: List[Request] = field(default_factory=list, repr=False)

    def verify_conservation(self):
        """Enforce fundamental conservation identities."""
        if self.num_admitted + self.num_rejected != self.num_offered:
            raise ValueError(
                f"Conservation violated: admitted ({self.num_admitted}) + rejected ({self.num_rejected}) "
                f"!= offered ({self.num_offered}) for {self.policy} on {self.workload_name}"
            )
        if self.num_completed + self.num_expired != self.num_admitted:
            raise ValueError(
                f"Conservation violated: completed ({self.num_completed}) + expired ({self.num_expired}) "
                f"!= admitted ({self.num_admitted}) for {self.policy} on {self.workload_name}"
            )
        if self.num_completed_on_time + self.num_completed_late != self.num_completed:
            raise ValueError(
                f"Conservation violated: on-time ({self.num_completed_on_time}) + late ({self.num_completed_late}) "
                f"!= completed ({self.num_completed}) for {self.policy} on {self.workload_name}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "workload_name": self.workload_name,
            "workload_category": self.workload_category,
            "load_level": self.load_level,
            "deadline_tightness": self.deadline_tightness,
            "seed": self.seed,
            "experiment_type": self.experiment_type,
            "num_offered": self.num_offered,
            "num_admitted": self.num_admitted,
            "num_rejected": self.num_rejected,
            "num_expired": self.num_expired,
            "num_completed": self.num_completed,
            "num_completed_on_time": self.num_completed_on_time,
            "num_completed_late": self.num_completed_late,
            "on_time_ratio_offered": self.on_time_ratio_offered,
            "on_time_ratio_accepted": self.on_time_ratio_accepted,
            "deadline_miss_ratio_completed": self.deadline_miss_ratio_completed,
            "rejection_ratio": self.rejection_ratio,
            "expiration_ratio": self.expiration_ratio,
            "mean_response_time_ms": self.mean_response_time_ms,
            "median_response_time_ms": self.median_response_time_ms,
            "p95_response_time_ms": self.p95_response_time_ms,
            "p99_response_time_ms": self.p99_response_time_ms,
            "mean_waiting_time_ms": self.mean_waiting_time_ms,
            "median_waiting_time_ms": self.median_waiting_time_ms,
            "mean_queue_length": self.mean_queue_length,
            "max_queue_length": self.max_queue_length,
            "total_gpu_busy_time_ms": self.total_gpu_busy_time_ms,
            "total_makespan_ms": self.total_makespan_ms,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "mean_val_accuracy": self.mean_val_accuracy,
            "synthetic_expected_test_accuracy": self.synthetic_expected_test_accuracy,
            "empirical_test_accuracy": self.empirical_test_accuracy if self.empirical_test_accuracy is not None else "N/A",
            "count_deit_small": self.count_deit_small,
            "count_evit_k09": self.count_evit_k09,
            "count_evit_k07": self.count_evit_k07,
            "count_evit_k05": self.count_evit_k05,
        }


class CanonicalSimulator:
    """Rigorous non-preemptive discrete-event simulator with exact queue length tracking."""

    def __init__(
        self,
        profiles: Dict[str, ModelProfile],
        latency_metric: str = "p95",
        backend: str = "cuda_graph",
        firm_deadlines: bool = True,
        quality_threshold: float = 92.0,
    ):
        self.profiles = profiles
        self.latency_metric = latency_metric
        self.backend = backend
        self.firm_deadlines = firm_deadlines
        self.quality_threshold = quality_threshold

        self.admission_controller = AdmissionController(
            profiles=profiles,
            latency_metric=latency_metric,
            backend=backend,
            quality_threshold=quality_threshold,
        )
        self.adaptive_controller = AdaptiveController(
            profiles=profiles,
            latency_metric=latency_metric,
            backend=backend,
            quality_threshold=quality_threshold,
            firm_deadlines=firm_deadlines,
        )
        self.fixed_evit_model = "evit_small_k07_fuse"

    def run(
        self,
        policy: Policy,
        workload: List[Request],
        workload_name: str,
        workload_category: str = "unknown",
        load_level: str = "unknown",
        deadline_tightness: str = "unknown",
        seed: int = 42,
    ) -> CanonicalResult:
        """Execute discrete-event simulation across workload trace."""
        is_edf = policy in (
            Policy.EDF_FIXED_DEIT,
            Policy.EDF_FIXED_EVIT,
            Policy.EDF_ADMISSION_FIXED_DEIT,
            Policy.EDF_ADAPTIVE,
            Policy.PROPOSED_ADAPTIVE_AC,
        )
        queue = EDFQueue() if is_edf else FIFOQueue()

        use_admission = policy in (Policy.EDF_ADMISSION_FIXED_DEIT, Policy.PROPOSED_ADAPTIVE_AC)
        is_adaptive = policy in (Policy.EDF_ADAPTIVE, Policy.PROPOSED_ADAPTIVE_AC)

        # Event queue: heap of (timestamp_ms, event_order, event_type, payload)
        event_heap: List[Tuple[float, int, EventType, Any]] = []
        event_counter = 0

        # Enqueue all arrival events
        for req in workload:
            heapq.heappush(event_heap, (req.arrival_time_ms, event_counter, EventType.ARRIVAL, req))
            event_counter += 1

        current_time_ms = 0.0
        gpu_busy = False
        gpu_busy_until_ms = 0.0
        total_gpu_busy_time = 0.0

        all_requests: List[Request] = []
        completed_requests: List[Request] = []
        model_counts = {"deit_small_baseline": 0, "evit_small_k09_fuse": 0, "evit_small_k07_fuse": 0, "evit_small_k05_fuse": 0}

        # Queue tracking over time
        last_event_time = 0.0
        time_weighted_queue_area = 0.0
        max_q_len = 0

        def update_queue_tracking(now: float):
            nonlocal last_event_time, time_weighted_queue_area, max_q_len
            dt = max(0.0, now - last_event_time)
            q_size = len(queue)
            time_weighted_queue_area += q_size * dt
            if q_size > max_q_len:
                max_q_len = q_size
            last_event_time = now

        def dispatch_next():
            nonlocal gpu_busy, gpu_busy_until_ms, total_gpu_busy_time, event_counter
            while not queue.is_empty():
                head = queue.pop()
                update_queue_tracking(current_time_ms)

                # Check if expired while queued
                if current_time_ms >= head.absolute_deadline_ms - 1e-6:
                    head.mark_expired(current_time_ms, reason="Expired while queued")
                    all_requests.append(head)
                    continue

                if is_adaptive:
                    model_id, exec_time, slack = self.adaptive_controller.select_model(head, current_time_ms)
                    if model_id is None:
                        head.mark_expired(current_time_ms, reason="Infeasible at dispatch (firm deadline)")
                        all_requests.append(head)
                        continue
                else:
                    if policy in (Policy.FIFO_FIXED_DEIT, Policy.EDF_FIXED_DEIT, Policy.EDF_ADMISSION_FIXED_DEIT):
                        model_id = "deit_small_baseline"
                    else:
                        model_id = self.fixed_evit_model

                    exec_time = self.profiles[model_id].get_latency_ms(self.latency_metric, self.backend)

                    if self.firm_deadlines and (current_time_ms + exec_time > head.absolute_deadline_ms + 1e-6):
                        if use_admission:
                            head.mark_expired(current_time_ms, reason="Infeasible at dispatch (fixed model)")
                            all_requests.append(head)
                            continue

                # Dispatch onto GPU
                gpu_busy = True
                gpu_busy_until_ms = current_time_ms + exec_time
                total_gpu_busy_time += exec_time
                model_counts[model_id] += 1

                head.mark_dispatched(current_time_ms, model_id, exec_time)
                heapq.heappush(event_heap, (gpu_busy_until_ms, event_counter, EventType.COMPLETION, head))
                event_counter += 1
                break

        # Main Event Loop
        while event_heap:
            timestamp, _, ev_type, payload = heapq.heappop(event_heap)
            update_queue_tracking(timestamp)
            current_time_ms = timestamp

            if ev_type == EventType.ARRIVAL:
                req: Request = payload

                if use_admission:
                    fixed_model = "deit_small_baseline" if policy == Policy.EDF_ADMISSION_FIXED_DEIT else None
                    feasible = self.admission_controller.is_feasible(
                        request=req,
                        current_time_ms=current_time_ms,
                        gpu_busy_until_ms=gpu_busy_until_ms,
                        queued_requests=queue.list_requests(),
                        fixed_model_id=fixed_model,
                    )
                    if not feasible:
                        req.mark_rejected(current_time_ms, reason="Admission test: insufficient slack")
                        all_requests.append(req)
                        continue

                queue.push(req)
                update_queue_tracking(current_time_ms)

                has_concurrent_arrival = bool(
                    event_heap
                    and abs(event_heap[0][0] - timestamp) < 1e-9
                    and event_heap[0][2] == EventType.ARRIVAL
                )
                if not gpu_busy and not has_concurrent_arrival:
                    dispatch_next()

            elif ev_type == EventType.COMPLETION:
                req: Request = payload
                actual_exec = req.estimated_exec_time_ms
                req.mark_completed(current_time_ms, actual_exec)
                all_requests.append(req)
                completed_requests.append(req)

                gpu_busy = False
                dispatch_next()

        # Compute metrics
        num_offered = len(workload)
        num_rejected = sum(1 for r in all_requests if r.outcome == RequestOutcome.REJECTED)
        num_admitted = num_offered - num_rejected
        num_expired = sum(1 for r in all_requests if r.outcome == RequestOutcome.EXPIRED)
        num_completed = len(completed_requests)
        num_on_time = sum(1 for r in completed_requests if r.outcome == RequestOutcome.ON_TIME)
        num_late = sum(1 for r in completed_requests if r.outcome == RequestOutcome.LATE)

        on_time_ratio_offered = round(num_on_time / num_offered, 4) if num_offered > 0 else 0.0
        on_time_ratio_accepted = round(num_on_time / num_admitted, 4) if num_admitted > 0 else 0.0
        deadline_miss_ratio_completed = round(num_late / num_completed, 4) if num_completed > 0 else 0.0
        rejection_ratio = round(num_rejected / num_offered, 4) if num_offered > 0 else 0.0
        expiration_ratio = round(num_expired / num_offered, 4) if num_offered > 0 else 0.0

        resp_times = [r.response_time_ms for r in completed_requests if r.response_time_ms is not None]
        wait_times = [r.waiting_time_ms for r in completed_requests if r.waiting_time_ms is not None]

        mean_resp = round(float(np.mean(resp_times)), 3) if resp_times else 0.0
        median_resp = round(float(np.median(resp_times)), 3) if resp_times else 0.0
        p95_resp = round(float(np.percentile(resp_times, 95)), 3) if resp_times else 0.0
        p99_resp = round(float(np.percentile(resp_times, 99)), 3) if resp_times else 0.0

        mean_wait = round(float(np.mean(wait_times)), 3) if wait_times else 0.0
        median_wait = round(float(np.median(wait_times)), 3) if wait_times else 0.0

        makespan = max(current_time_ms, 1e-6)
        gpu_util = round((total_gpu_busy_time / makespan) * 100.0, 2)
        mean_q_len = round(time_weighted_queue_area / makespan, 3)

        # Accuracy calculations (strictly separated)
        if completed_requests:
            val_accs = [self.profiles[r.selected_model].val_top1_accuracy for r in completed_requests]
            test_accs = [self.profiles[r.selected_model].test_top1_accuracy for r in completed_requests]
            mean_val_acc = round(float(np.mean(val_accs)), 2)
            synthetic_test_acc = round(float(np.mean(test_accs)), 2)
        else:
            mean_val_acc = 0.0
            synthetic_test_acc = 0.0

        res = CanonicalResult(
            policy=policy.value,
            workload_name=workload_name,
            workload_category=workload_category,
            load_level=load_level,
            deadline_tightness=deadline_tightness,
            seed=seed,
            experiment_type=f"SIMULATOR_{self.latency_metric.upper()}",
            num_offered=num_offered,
            num_admitted=num_admitted,
            num_rejected=num_rejected,
            num_expired=num_expired,
            num_completed=num_completed,
            num_completed_on_time=num_on_time,
            num_completed_late=num_late,
            on_time_ratio_offered=on_time_ratio_offered,
            on_time_ratio_accepted=on_time_ratio_accepted,
            deadline_miss_ratio_completed=deadline_miss_ratio_completed,
            rejection_ratio=rejection_ratio,
            expiration_ratio=expiration_ratio,
            mean_response_time_ms=mean_resp,
            median_response_time_ms=median_resp,
            p95_response_time_ms=p95_resp,
            p99_response_time_ms=p99_resp,
            mean_waiting_time_ms=mean_wait,
            median_waiting_time_ms=median_wait,
            mean_queue_length=mean_q_len,
            max_queue_length=max_q_len,
            total_gpu_busy_time_ms=round(total_gpu_busy_time, 3),
            total_makespan_ms=round(makespan, 3),
            gpu_utilization_pct=gpu_util,
            mean_val_accuracy=mean_val_acc,
            synthetic_expected_test_accuracy=synthetic_test_acc,
            empirical_test_accuracy=None,
            count_deit_small=model_counts["deit_small_baseline"],
            count_evit_k09=model_counts["evit_small_k09_fuse"],
            count_evit_k07=model_counts["evit_small_k07_fuse"],
            count_evit_k05=model_counts["evit_small_k05_fuse"],
            request_records=all_requests,
        )

        res.verify_conservation()
        return res
