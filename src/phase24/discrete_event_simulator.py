"""Discrete-event real-time simulator for single-GPU inference scheduling."""

from __future__ import annotations

import copy
import enum
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from .request import Request, RequestOutcome, RequestState
from .model_profiles import ModelProfile
from .priority_queue import FIFOQueue, EDFQueue, PriorityQueue
from .admission_control import AdmissionController
from .adaptive_controller import AdaptiveController


class Policy(str, enum.Enum):
    """The 7 scheduling policies evaluated in Phase 24."""
    FIFO_FIXED_DEIT = "FIFO_fixed_DeiT"
    EDF_FIXED_DEIT = "EDF_fixed_DeiT"
    FIFO_FIXED_EVIT = "FIFO_fixed_EViT"
    EDF_FIXED_EVIT = "EDF_fixed_EViT"
    EDF_ADMISSION_FIXED_DEIT = "EDF_AC_fixed_DeiT"
    EDF_ADAPTIVE = "EDF_Adaptive"
    PROPOSED_ADAPTIVE_AC = "EDF_Adaptive_AC"


@dataclass
class SimulationResult:
    """Aggregated evaluation metrics for a completed simulation run."""
    policy: str
    workload_name: str
    num_offered: int
    num_admitted: int
    num_rejected: int
    num_expired: int
    num_completed: int
    num_completed_on_time: int
    num_completed_late: int

    on_time_ratio_offered: float        # on_time / offered (Primary QoS metric)
    deadline_miss_ratio_completed: float # late / completed
    rejection_ratio: float              # rejected / offered
    expiration_ratio: float             # expired / offered

    mean_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float

    mean_waiting_time_ms: float
    median_waiting_time_ms: float

    gpu_busy_time_ms: float
    total_makespan_ms: float
    gpu_utilization_pct: float

    mean_val_accuracy: float
    mean_test_accuracy: float

    model_selection_counts: Dict[str, int]
    request_records: List[Request] = field(repr=False)


class EventType(str, enum.Enum):
    ARRIVAL = "ARRIVAL"
    COMPLETION = "COMPLETION"


class RealTimeSimulator:
    """Discrete-event simulator modeling non-preemptive GPU execution."""

    def __init__(
        self,
        profiles: Dict[str, ModelProfile],
        latency_metric: str = "p95",
        backend: str = "cuda_graph",
        quality_threshold: float = 45.0,
        firm_deadlines: bool = True,
        fixed_evit_model: str = "evit_small_k07_fuse",
    ):
        self.profiles = profiles
        self.latency_metric = latency_metric
        self.backend = backend
        self.quality_threshold = quality_threshold
        self.firm_deadlines = firm_deadlines
        self.fixed_evit_model = fixed_evit_model

        self.admission_controller = AdmissionController(
            profiles=self.profiles,
            latency_metric=self.latency_metric,
            backend=self.backend,
            quality_threshold=self.quality_threshold,
        )
        self.adaptive_controller = AdaptiveController(
            profiles=self.profiles,
            latency_metric=self.latency_metric,
            backend=self.backend,
            quality_threshold=self.quality_threshold,
            firm_deadlines=self.firm_deadlines,
        )

    def run(self, policy: Policy, workload: List[Request], workload_name: str = "workload") -> SimulationResult:
        """Execute simulation over a workload trace under the specified policy."""
        # Deepcopy workload so simulation runs leave input unmodified
        trace = [copy.deepcopy(r) for r in workload]

        # Select queue data structure
        if policy in (Policy.FIFO_FIXED_DEIT, Policy.FIFO_FIXED_EVIT):
            queue: PriorityQueue = FIFOQueue()
        else:
            queue = EDFQueue()

        use_admission = policy in (Policy.EDF_ADMISSION_FIXED_DEIT, Policy.PROPOSED_ADAPTIVE_AC)
        is_adaptive = policy in (Policy.EDF_ADAPTIVE, Policy.PROPOSED_ADAPTIVE_AC)

        # Event queue: entries are (timestamp, event_counter, event_type, payload)
        event_heap: List[Tuple[float, int, EventType, Any]] = []
        event_counter = 0

        for req in trace:
            heapq.heappush(event_heap, (req.arrival_time_ms, event_counter, EventType.ARRIVAL, req))
            event_counter += 1

        current_time_ms = 0.0
        gpu_busy = False
        gpu_busy_until_ms = 0.0
        current_request: Optional[Request] = None
        total_gpu_busy_time = 0.0

        all_requests: List[Request] = []
        completed_requests: List[Request] = []
        model_counts = {cfg: 0 for cfg in self.profiles.keys()}

        def dispatch_next():
            nonlocal gpu_busy, gpu_busy_until_ms, current_request, event_counter, total_gpu_busy_time

            while not queue.is_empty() and not gpu_busy:
                req = queue.pop()

                # 1. Check if deadline has already passed while in queue
                if current_time_ms >= req.absolute_deadline_ms - 1e-6:
                    req.mark_expired(current_time_ms, reason="Expired while queued")
                    all_requests.append(req)
                    continue

                # 2. Determine model and execution time
                if is_adaptive:
                    model_id, exec_time, slack = self.adaptive_controller.select_model(req, current_time_ms)
                    if model_id is None:
                        # Even fastest model cannot meet deadline under firm semantics
                        req.mark_expired(current_time_ms, reason="Infeasible at dispatch (firm deadline)")
                        all_requests.append(req)
                        continue
                else:
                    if policy in (Policy.FIFO_FIXED_DEIT, Policy.EDF_FIXED_DEIT, Policy.EDF_ADMISSION_FIXED_DEIT):
                        model_id = "deit_small_baseline"
                    else:
                        model_id = self.fixed_evit_model

                    exec_time = self.profiles[model_id].get_latency_ms(self.latency_metric, self.backend)

                    # Under firm deadlines for fixed policy: if execution cannot meet deadline, expire early
                    if self.firm_deadlines and (current_time_ms + exec_time > req.absolute_deadline_ms + 1e-6):
                        if use_admission:
                            # Already filtered at admission, but in case queue delay pushed it past deadline
                            req.mark_expired(current_time_ms, reason="Infeasible at dispatch (fixed model)")
                            all_requests.append(req)
                            continue

                # 3. Dispatch onto GPU
                gpu_busy = True
                gpu_busy_until_ms = current_time_ms + exec_time
                total_gpu_busy_time += exec_time
                current_request = req
                model_counts[model_id] += 1

                req.mark_dispatched(current_time_ms, model_id, exec_time)
                heapq.heappush(event_heap, (gpu_busy_until_ms, event_counter, EventType.COMPLETION, req))
                event_counter += 1
                break

        # Main Simulation Event Loop
        while event_heap:
            timestamp, _, ev_type, payload = heapq.heappop(event_heap)
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
                # Process all concurrent arrivals at the identical timestamp before dispatching
                has_concurrent_arrival = bool(
                    event_heap
                    and abs(event_heap[0][0] - timestamp) < 1e-9
                    and event_heap[0][2] == EventType.ARRIVAL
                )
                if not gpu_busy and not has_concurrent_arrival:
                    dispatch_next()

            elif ev_type == EventType.COMPLETION:
                req: Request = payload
                actual_exec = req.estimated_exec_time_ms  # In simulation, actual matches estimated profile
                req.mark_completed(current_time_ms, actual_exec_ms=actual_exec)
                all_requests.append(req)
                completed_requests.append(req)

                gpu_busy = False
                current_request = None
                dispatch_next()

        # Handle any requests remaining in queue after simulation events exhaust (e.g. if any)
        while not queue.is_empty():
            req = queue.pop()
            req.mark_expired(current_time_ms, reason="Simulation ended while queued")
            all_requests.append(req)

        # Aggregate Metrics & Compute Accounting
        num_offered = len(trace)
        num_rejected = sum(1 for r in all_requests if r.outcome == RequestOutcome.REJECTED)
        num_admitted = num_offered - num_rejected
        num_expired = sum(1 for r in all_requests if r.outcome == RequestOutcome.EXPIRED)
        num_completed = len(completed_requests)
        num_on_time = sum(1 for r in completed_requests if r.outcome == RequestOutcome.ON_TIME)
        num_late = sum(1 for r in completed_requests if r.outcome == RequestOutcome.LATE)

        # Invariant verification: no double counting or loss of requests
        assert num_offered == num_admitted + num_rejected, "Offered != Admitted + Rejected"
        assert num_admitted == num_completed + num_expired, "Admitted != Completed + Expired"
        assert num_completed == num_on_time + num_late, "Completed != OnTime + Late"

        on_time_ratio_offered = round(num_on_time / num_offered, 4) if num_offered > 0 else 0.0
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

        # Accuracy of completed predictions
        if completed_requests:
            val_accs = [self.profiles[r.selected_model].val_top1_accuracy for r in completed_requests]
            test_accs = [self.profiles[r.selected_model].test_top1_accuracy for r in completed_requests]
            mean_val_acc = round(float(np.mean(val_accs)), 2)
            mean_test_acc = round(float(np.mean(test_accs)), 2)
        else:
            mean_val_acc = 0.0
            mean_test_acc = 0.0

        return SimulationResult(
            policy=policy.value,
            workload_name=workload_name,
            num_offered=num_offered,
            num_admitted=num_admitted,
            num_rejected=num_rejected,
            num_expired=num_expired,
            num_completed=num_completed,
            num_completed_on_time=num_on_time,
            num_completed_late=num_late,
            on_time_ratio_offered=on_time_ratio_offered,
            deadline_miss_ratio_completed=deadline_miss_ratio_completed,
            rejection_ratio=rejection_ratio,
            expiration_ratio=expiration_ratio,
            mean_response_time_ms=mean_resp,
            median_response_time_ms=median_resp,
            p95_response_time_ms=p95_resp,
            p99_response_time_ms=p99_resp,
            mean_waiting_time_ms=mean_wait,
            median_waiting_time_ms=median_wait,
            gpu_busy_time_ms=round(total_gpu_busy_time, 3),
            total_makespan_ms=round(makespan, 3),
            gpu_utilization_pct=gpu_util,
            mean_val_accuracy=mean_val_acc,
            mean_test_accuracy=mean_test_acc,
            model_selection_counts=model_counts,
            request_records=all_requests,
        )
