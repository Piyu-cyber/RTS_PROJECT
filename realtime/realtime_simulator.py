"""Real-Time Discrete-Event Simulator with Priority Queue and Admission Control.

Phase 18 Implementation:
- Models real-time image segmentation inference on a single non-preemptive GPU.
- Explicitly tracks absolute deadlines D_i = r_i + d_i, remaining budgets B_i(t) = D_i - t,
  queueing delays, execution intervals, lateness = C_i - D_i, and request states.
- Priority Queue implementation using heapq with deterministic tie-breaking.
- Supports policies:
    1. FIFO (First In, First Out)
    2. EDF (Earliest Deadline First)
    3. EDF with quality-aware candidate selection (tau threshold)
    4. FIFO with deadline-aware admission control
    5. Combined Proposed Policy: EDF + quality selection + deadline-aware admission control
    6. Always Dispatch (Fixed model)
    7. Empirical Upper Bound (EUB = 32.33 ms)
    8. Formal Bound-Aware (enforces Outcome C safeguard: formal WCET unavailable)
"""

from __future__ import annotations

import enum
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class RequestStatus(str, enum.Enum):
    PENDING = "PENDING"
    ADMITTED = "ADMITTED"
    RUNNING = "RUNNING"
    COMPLETED_ON_TIME = "COMPLETED_ON_TIME"
    DEADLINE_MISSED = "DEADLINE_MISSED"
    REJECTED_INFEASIBLE = "REJECTED_INFEASIBLE"


class PolicyType(str, enum.Enum):
    ALWAYS_FULL_ATTENTION = "always_full_attention"
    ALWAYS_MANUAL_DWCONV = "always_manual_dwconv"
    FIFO = "fifo"
    EDF = "edf"
    EDF_QUALITY = "edf_quality"
    ADMISSION_CONTROL_FIFO = "admission_control_fifo"
    COMBINED_PROPOSED = "combined_proposed"
    EMPIRICAL_UPPER_BOUND = "empirical_upper_bound"
    FORMAL_BOUND_AWARE = "formal_bound_aware"


class SimulatorError(Exception):
    """Base exception for simulator errors."""


class FormalBoundUnavailableError(SimulatorError):
    """Raised when formal WCET bound is requested on a platform where it is not established."""


@dataclass
class CandidateProfile:
    """Model candidate with measured validation mIoU and latency statistics."""

    name: str
    display_name: str
    validation_miou: float
    median_ms: float
    mean_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    empirical_upper_bound_ms: float
    formal_wcet_ms: float | None = None
    is_formally_justified: bool = False

    def get_latency_estimate(self, metric: str = "p95") -> float:
        if metric == "median":
            return self.median_ms
        elif metric == "mean":
            return self.mean_ms
        elif metric == "p95":
            return self.p95_ms
        elif metric == "p99":
            return self.p99_ms
        elif metric == "max":
            return self.max_ms
        elif metric in ("eub", "empirical_upper_bound"):
            return self.empirical_upper_bound_ms
        elif metric == "formal_wcet":
            if not self.is_formally_justified or self.formal_wcet_ms is None:
                raise FormalBoundUnavailableError(
                    f"Candidate {self.name!r} lacks verified formal WCET bound. "
                    "Phase 16 established Outcome C: formal WCET cannot be guaranteed on RTX 4060/Windows 11."
                )
            return self.formal_wcet_ms
        else:
            raise SimulatorError(f"Unsupported latency metric: {metric!r}")


def get_default_candidates() -> dict[str, CandidateProfile]:
    """Return the audited Phase 12B/16 candidate profiles measured on the RTX 4060 Laptop GPU."""
    return {
        "full_attention": CandidateProfile(
            name="full_attention",
            display_name="Full Attention",
            validation_miou=0.819709,
            median_ms=23.655,
            mean_ms=23.534,
            p95_ms=28.575,
            p99_ms=29.972,
            max_ms=30.737,
            empirical_upper_bound_ms=32.329,
            formal_wcet_ms=None,
            is_formally_justified=False,
        ),
        "sigma_selected": CandidateProfile(
            name="sigma_selected",
            display_name="Sigma DWConv",
            validation_miou=0.758961,
            median_ms=26.411,
            mean_ms=25.654,
            p95_ms=30.520,
            p99_ms=31.518,
            max_ms=31.592,
            empirical_upper_bound_ms=32.953,
            formal_wcet_ms=None,
            is_formally_justified=False,
        ),
        "manual_interleaved": CandidateProfile(
            name="manual_interleaved",
            display_name="Manual DWConv",
            validation_miou=0.744658,
            median_ms=25.740,
            mean_ms=25.364,
            p95_ms=29.889,
            p99_ms=30.788,
            max_ms=30.881,
            empirical_upper_bound_ms=33.644,
            formal_wcet_ms=None,
            is_formally_justified=False,
        ),
    }


@dataclass
class Request:
    """Explicit real-time representation of an image segmentation inference request."""

    request_id: str
    arrival_time: float  # r_i in ms
    relative_deadline: float  # d_i in ms
    min_miou_threshold: float = 0.75  # tau_i
    actual_duration_ms: float | None = None  # Ground-truth physical execution latency

    # Computed real-time state
    absolute_deadline: float = field(init=False)  # D_i = r_i + d_i in ms
    status: RequestStatus = RequestStatus.PENDING
    selected_configuration: str | None = None
    estimated_execution_time: float | None = None  # C_c in ms
    admitted_time: float | None = None  # ms
    start_time: float | None = None  # S_i in ms
    completion_time: float | None = None  # C_i in ms
    lateness: float | None = None  # lateness_i = C_i - D_i in ms
    queue_waiting_time: float | None = None  # S_i - r_i in ms
    response_time: float | None = None  # C_i - r_i in ms
    rejection_reason: str | None = None
    miss_type: str | None = None  # 'ON_GPU' or 'IN_QUEUE'
    eligible_candidates: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.absolute_deadline = round(self.arrival_time + self.relative_deadline, 4)

    def remaining_budget(self, current_time: float) -> float:
        """Calculate dynamic remaining budget B_i(t) = D_i - t in ms."""
        return self.absolute_deadline - current_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "arrival_time_ms": round(self.arrival_time, 2),
            "relative_deadline_ms": round(self.relative_deadline, 2),
            "absolute_deadline_ms": round(self.absolute_deadline, 2),
            "min_miou_threshold": round(self.min_miou_threshold, 4),
            "status": self.status.value,
            "selected_configuration": self.selected_configuration,
            "estimated_execution_time_ms": round(self.estimated_execution_time, 2) if self.estimated_execution_time is not None else None,
            "actual_duration_ms": round(self.actual_duration_ms, 2) if self.actual_duration_ms is not None else None,
            "start_time_ms": round(self.start_time, 2) if self.start_time is not None else None,
            "completion_time_ms": round(self.completion_time, 2) if self.completion_time is not None else None,
            "lateness_ms": round(self.lateness, 2) if self.lateness is not None else None,
            "queue_waiting_time_ms": round(self.queue_waiting_time, 2) if self.queue_waiting_time is not None else None,
            "response_time_ms": round(self.response_time, 2) if self.response_time is not None else None,
            "rejection_reason": self.rejection_reason,
            "eligible_candidates": self.eligible_candidates,
        }


@dataclass(order=True)
class QueueItem:
    """Priority queue item with deterministic ordering."""

    priority_key: float  # Absolute deadline (EDF) or arrival time (FIFO)
    secondary_key: float  # Arrival time (EDF) or absolute deadline (FIFO)
    request_id: str
    request: Request = field(compare=False)


class RealTimePriorityQueue:
    """Priority queue for pending requests with deterministic tie-breaking."""

    def __init__(self, use_edf: bool = True):
        self.use_edf = use_edf
        self._heap: list[QueueItem] = []

    def push(self, request: Request) -> None:
        if self.use_edf:
            # Order primarily by earliest absolute deadline, then arrival time, then request_id
            item = QueueItem(
                priority_key=request.absolute_deadline,
                secondary_key=request.arrival_time,
                request_id=request.request_id,
                request=request,
            )
        else:
            # FIFO: Order primarily by earliest arrival time, then deadline, then request_id
            item = QueueItem(
                priority_key=request.arrival_time,
                secondary_key=request.absolute_deadline,
                request_id=request.request_id,
                request=request,
            )
        heapq.heappush(self._heap, item)

    def pop(self) -> Request:
        if not self._heap:
            raise SimulatorError("Cannot pop from empty priority queue")
        return heapq.heappop(self._heap).request

    def peek(self) -> Request | None:
        if not self._heap:
            return None
        return self._heap[0].request

    def __len__(self) -> int:
        return len(self._heap)

    def get_ordered_requests(self) -> list[Request]:
        """Return a copy of all requests in priority order without modifying the heap."""
        items_copy = list(self._heap)
        sorted_items = sorted(items_copy)
        return [item.request for item in sorted_items]

    def total_estimated_duration(self, default_estimate: float = 23.66) -> float:
        """Sum of estimated execution times for all requests currently in queue."""
        total = 0.0
        for item in self._heap:
            req = item.request
            est = req.estimated_execution_time if req.estimated_execution_time is not None else default_estimate
            total += est
        return total


@dataclass
class ScheduleEvent:
    """Log record of a discrete scheduling or execution event."""

    timestamp_ms: float
    event_type: str  # ARRIVAL, ADMISSION, REJECTION, DISPATCH, COMPLETION, TIMEOUT
    request_id: str
    message: str
    queue_length: int
    gpu_busy: bool
    current_gpu_task: str | None = None
    remaining_budget_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


class RealTimeSimulator:
    """Discrete-event simulator modeling non-preemptive GPU inference under explicit deadlines."""

    def __init__(
        self,
        policy: PolicyType | str = PolicyType.COMBINED_PROPOSED,
        candidates: dict[str, CandidateProfile] | None = None,
        latency_metric: str = "p95",
        use_admission_control: bool | None = None,
        drop_expired_in_queue: bool = True,
        raw_latencies: dict[str, list[float]] | None = None,
    ):
        self.policy = PolicyType(policy)
        self.candidates = candidates or get_default_candidates()
        self.latency_metric = latency_metric
        self.drop_expired_in_queue = drop_expired_in_queue
        self.raw_latencies = raw_latencies
        self.dispatch_counter: int = 0

        # Determine policy features
        if self.policy == PolicyType.FORMAL_BOUND_AWARE:
            # Check if any candidate has verified formal WCET
            any_formal = any(c.is_formally_justified for c in self.candidates.values())
            if not any_formal:
                # Document Outcome C safeguard
                self.is_formal_available = False
            else:
                self.is_formal_available = True
        else:
            self.is_formal_available = True

        if self.policy in (
            PolicyType.ALWAYS_FULL_ATTENTION,
            PolicyType.ALWAYS_MANUAL_DWCONV,
            PolicyType.FIFO,
            PolicyType.ADMISSION_CONTROL_FIFO,
        ):
            self.use_edf = False
        else:
            self.use_edf = True

        if use_admission_control is not None:
            self.use_admission_control = use_admission_control
        else:
            self.use_admission_control = self.policy in (
                PolicyType.ADMISSION_CONTROL_FIFO,
                PolicyType.COMBINED_PROPOSED,
                PolicyType.EMPIRICAL_UPPER_BOUND,
            )

        # Simulator runtime state
        self.current_time_ms: float = 0.0
        self.queue = RealTimePriorityQueue(use_edf=self.use_edf)
        self.current_running_request: Request | None = None
        self.current_task_end_time_ms: float = 0.0

        # Output collections
        self.events: list[ScheduleEvent] = []
        self.processed_requests: list[Request] = []
        self.total_gpu_busy_time_ms: float = 0.0
        self.total_gpu_idle_time_ms: float = 0.0
        self.last_idle_start_ms: float = 0.0
        self.max_queue_length: int = 0

    def get_remaining_running_time(self, current_time: float) -> float:
        """Return remaining execution time of currently running GPU task in ms."""
        if self.current_running_request is None:
            return 0.0
        return max(0.0, self.current_task_end_time_ms - current_time)

    def get_queued_work_ahead_of(self, request: Request) -> float:
        """Calculate the sum of estimated execution times for requests queued ahead of this request."""
        if not self.use_edf:
            # FIFO: All currently queued requests arrived before this request, so all execute ahead
            return self.queue.total_estimated_duration()
        else:
            # EDF: Only requests with earlier absolute deadline (or tie-break ahead) execute before this request
            total = 0.0
            for item in self.queue._heap:
                q_req = item.request
                # EDF order: earlier absolute deadline, or equal deadline with earlier arrival
                if (q_req.absolute_deadline < request.absolute_deadline) or (
                    q_req.absolute_deadline == request.absolute_deadline and q_req.arrival_time <= request.arrival_time
                ):
                    est = q_req.estimated_execution_time if q_req.estimated_execution_time is not None else 23.66
                    total += est
            return total

    def select_configuration_and_check_feasibility(
        self,
        request: Request,
        current_time: float,
    ) -> tuple[bool, str | None, float | None, list[str], str]:
        """Perform quality filtering, latency estimation, and queue-aware feasibility check."""
        # 1. Quality Eligibility: C_Q = {c | validation_miou >= min_miou_threshold}
        if self.policy == PolicyType.ALWAYS_FULL_ATTENTION:
            eligible_configs = ["full_attention"]
        elif self.policy == PolicyType.ALWAYS_MANUAL_DWCONV:
            eligible_configs = ["manual_interleaved"]
        else:
            eligible_configs = [
                c.name for c in self.candidates.values()
                if c.validation_miou >= request.min_miou_threshold
            ]

        request.eligible_candidates = sorted(eligible_configs)

        if not eligible_configs:
            return (
                False,
                None,
                None,
                [],
                f"Quality floor violated: tau={request.min_miou_threshold:.4f} excludes all models.",
            )

        # 2. Timing Estimates for eligible configurations
        metric_to_use = "eub" if self.policy == PolicyType.EMPIRICAL_UPPER_BOUND else self.latency_metric
        estimates: dict[str, float] = {}
        for c_name in eligible_configs:
            estimates[c_name] = self.candidates[c_name].get_latency_estimate(metric_to_use)

        # 3. Calculate earliest completion time accounting for running GPU task and queued work ahead
        remaining_running = self.get_remaining_running_time(current_time)
        queued_work_ahead = self.get_queued_work_ahead_of(request)

        # Best candidate selection:
        # Highest validation mIoU, tie-break by lowest latency estimate
        sorted_candidates = sorted(
            eligible_configs,
            key=lambda name: (-self.candidates[name].validation_miou, estimates[name], name),
        )

        best_cand = sorted_candidates[0]
        cand_est = estimates[best_cand]

        estimated_completion_ms = current_time + remaining_running + queued_work_ahead + cand_est
        is_feasible = estimated_completion_ms <= request.absolute_deadline

        # Also check if admitting this request in EDF would push any existing lower-priority request past its deadline
        if is_feasible and self.use_edf and self.use_admission_control:
            # Check existing requests in queue with later deadlines
            accumulated_time = estimated_completion_ms
            for item in sorted(self.queue._heap):
                later_req = item.request
                if later_req.absolute_deadline > request.absolute_deadline:
                    later_est = later_req.estimated_execution_time if later_req.estimated_execution_time is not None else 23.66
                    accumulated_time += later_est
                    if accumulated_time > later_req.absolute_deadline:
                        is_feasible = False
                        overshoot = accumulated_time - later_req.absolute_deadline
                        return (
                            False,
                            best_cand,
                            cand_est,
                            eligible_configs,
                            f"Infeasible: admitting {request.request_id} would delay existing queued request {later_req.request_id} past its deadline (overshoot {overshoot:+.2f} ms).",
                        )

        if not self.use_admission_control:
            # Policy does not reject: always admit
            return True, best_cand, cand_est, eligible_configs, "Admitted without admission control."

        if is_feasible:
            margin = request.absolute_deadline - estimated_completion_ms
            return (
                True,
                best_cand,
                cand_est,
                eligible_configs,
                f"Feasible: estimated completion {estimated_completion_ms:.2f} ms <= deadline {request.absolute_deadline:.2f} ms (margin {margin:+.2f} ms).",
            )
        else:
            overshoot = estimated_completion_ms - request.absolute_deadline
            return (
                False,
                best_cand,
                cand_est,
                eligible_configs,
                f"Infeasible: estimated completion {estimated_completion_ms:.2f} ms > deadline {request.absolute_deadline:.2f} ms (overshoot {overshoot:+.2f} ms).",
            )

    def _log_event(
        self,
        event_type: str,
        request_id: str,
        message: str,
        remaining_budget_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        event = ScheduleEvent(
            timestamp_ms=round(self.current_time_ms, 2),
            event_type=event_type,
            request_id=request_id,
            message=message,
            queue_length=len(self.queue),
            gpu_busy=(self.current_running_request is not None),
            current_gpu_task=(
                self.current_running_request.request_id
                if self.current_running_request is not None
                else None
            ),
            remaining_budget_ms=(
                round(remaining_budget_ms, 2)
                if remaining_budget_ms is not None
                else None
            ),
            details=details or {},
        )
        self.events.append(event)

    def _dispatch_next_to_gpu(self) -> None:
        """Pop the highest-priority waiting request and begin non-preemptive GPU inference."""
        if self.current_running_request is not None:
            return  # GPU is currently busy; cannot dispatch

        while len(self.queue) > 0:
            candidate_req = self.queue.pop()

            # Check if request has already expired while waiting in queue
            if self.drop_expired_in_queue and candidate_req.remaining_budget(self.current_time_ms) < 0:
                # Deadline missed while waiting in queue
                candidate_req.status = RequestStatus.DEADLINE_MISSED
                candidate_req.miss_type = "IN_QUEUE"
                candidate_req.start_time = None
                candidate_req.completion_time = self.current_time_ms
                candidate_req.queue_waiting_time = self.current_time_ms - candidate_req.arrival_time
                candidate_req.response_time = self.current_time_ms - candidate_req.arrival_time
                candidate_req.lateness = self.current_time_ms - candidate_req.absolute_deadline
                candidate_req.rejection_reason = "Deadline expired while waiting in queue before GPU dispatch."

                self._log_event(
                    event_type="TIMEOUT",
                    request_id=candidate_req.request_id,
                    message=f"Request {candidate_req.request_id} missed deadline while waiting in queue (lateness: {candidate_req.lateness:+.2f} ms).",
                    remaining_budget_ms=candidate_req.remaining_budget(self.current_time_ms),
                )
                self.processed_requests.append(candidate_req)
                continue

            # Found valid waiting request to execute
            self.current_running_request = candidate_req
            candidate_req.status = RequestStatus.RUNNING
            candidate_req.start_time = self.current_time_ms
            candidate_req.queue_waiting_time = self.current_time_ms - candidate_req.arrival_time

            # Determine actual execution duration
            if self.raw_latencies and candidate_req.selected_configuration in self.raw_latencies:
                l_list = self.raw_latencies[candidate_req.selected_configuration]
                duration = l_list[self.dispatch_counter % len(l_list)]
                self.dispatch_counter += 1
                candidate_req.actual_duration_ms = duration
            elif candidate_req.actual_duration_ms is not None:
                duration = candidate_req.actual_duration_ms
            elif candidate_req.estimated_execution_time is not None:
                duration = candidate_req.estimated_execution_time
            else:
                duration = 23.66  # Default full attention median

            self.current_task_end_time_ms = self.current_time_ms + duration

            self._log_event(
                event_type="DISPATCH",
                request_id=candidate_req.request_id,
                message=(
                    f"Dispatched {candidate_req.request_id} ({candidate_req.selected_configuration}) to GPU. "
                    f"Duration: {duration:.2f} ms, estimated end: {self.current_task_end_time_ms:.2f} ms."
                ),
                remaining_budget_ms=candidate_req.remaining_budget(self.current_time_ms),
                details={"duration_ms": duration, "configuration": candidate_req.selected_configuration},
            )
            return

        # Queue is empty and GPU is idle
        self.last_idle_start_ms = self.current_time_ms

    def _complete_current_task(self) -> None:
        """Finish the currently running GPU task and record outcomes."""
        req = self.current_running_request
        if req is None:
            return

        duration = self.current_task_end_time_ms - req.start_time
        self.total_gpu_busy_time_ms += duration

        req.completion_time = self.current_task_end_time_ms
        req.response_time = req.completion_time - req.arrival_time
        req.lateness = round(req.completion_time - req.absolute_deadline, 4)

        if req.completion_time <= req.absolute_deadline:
            req.status = RequestStatus.COMPLETED_ON_TIME
            outcome_msg = f"Completed ON TIME (lateness: {req.lateness:+.2f} ms <= 0)."
        else:
            req.status = RequestStatus.DEADLINE_MISSED
            req.miss_type = "ON_GPU"
            outcome_msg = f"DEADLINE MISSED on GPU (lateness: {req.lateness:+.2f} ms > 0)."

        self._log_event(
            event_type="COMPLETION",
            request_id=req.request_id,
            message=f"Request {req.request_id} finished execution. {outcome_msg}",
            remaining_budget_ms=req.remaining_budget(self.current_time_ms),
            details={"lateness_ms": req.lateness, "status": req.status.value},
        )

        self.processed_requests.append(req)
        self.current_running_request = None

        # GPU is now free; dispatch next pending task
        self._dispatch_next_to_gpu()

    def run_workload(self, requests: list[Request]) -> list[Request]:
        """Execute the discrete-event simulation across an input stream of requests."""
        if self.policy == PolicyType.FORMAL_BOUND_AWARE and not self.is_formal_available:
            raise FormalBoundUnavailableError(
                "Formal bound-aware scheduling is UNAVAILABLE because formal WCET has not been established on this platform (Outcome C). "
                "The simulator refused to execute with unverified bounds to protect safety guarantees."
            )

        # Reset simulator state
        self.current_time_ms = 0.0
        self.queue = RealTimePriorityQueue(use_edf=self.use_edf)
        self.current_running_request = None
        self.current_task_end_time_ms = 0.0
        self.events = []
        self.processed_requests = []
        self.total_gpu_busy_time_ms = 0.0
        self.total_gpu_idle_time_ms = 0.0
        self.last_idle_start_ms = 0.0
        self.max_queue_length = 0

        # Sort incoming requests chronologically by arrival time
        pending_arrivals = sorted(requests, key=lambda r: (r.arrival_time, r.request_id))

        arrival_idx = 0
        num_arrivals = len(pending_arrivals)

        while arrival_idx < num_arrivals or self.current_running_request is not None or len(self.queue) > 0:
            # Determine the next discrete event timestamp:
            # 1. Next request arrival time
            # 2. Next GPU task completion time
            next_arrival_time = pending_arrivals[arrival_idx].arrival_time if arrival_idx < num_arrivals else float("inf")
            next_completion_time = self.current_task_end_time_ms if self.current_running_request is not None else float("inf")

            if next_arrival_time <= next_completion_time and next_arrival_time < float("inf"):
                # Handle Arrival Event
                time_advance = next_arrival_time - self.current_time_ms
                if self.current_running_request is None and time_advance > 0:
                    self.total_gpu_idle_time_ms += time_advance

                self.current_time_ms = next_arrival_time
                req = pending_arrivals[arrival_idx]
                arrival_idx += 1

                self._log_event(
                    event_type="ARRIVAL",
                    request_id=req.request_id,
                    message=f"Request {req.request_id} arrived at {self.current_time_ms:.2f} ms (relative deadline {req.relative_deadline:.2f} ms, abs deadline {req.absolute_deadline:.2f} ms).",
                    remaining_budget_ms=req.remaining_budget(self.current_time_ms),
                )

                # Evaluate admission control and configuration selection
                is_admitted, chosen_config, est_lat, eligible, reason = (
                    self.select_configuration_and_check_feasibility(req, self.current_time_ms)
                )

                req.selected_configuration = chosen_config
                req.estimated_execution_time = est_lat

                if is_admitted:
                    req.status = RequestStatus.ADMITTED
                    req.admitted_time = self.current_time_ms
                    self._log_event(
                        event_type="ADMISSION",
                        request_id=req.request_id,
                        message=f"Admitted {req.request_id} with config {chosen_config}. Reason: {reason}",
                        remaining_budget_ms=req.remaining_budget(self.current_time_ms),
                    )

                    # If GPU is currently idle, dispatch immediately; otherwise push to queue
                    if self.current_running_request is None:
                        self.queue.push(req)
                        self._dispatch_next_to_gpu()
                    else:
                        self.queue.push(req)
                else:
                    req.status = RequestStatus.REJECTED_INFEASIBLE
                    req.rejection_reason = reason
                    req.completion_time = None
                    req.lateness = None
                    self._log_event(
                        event_type="REJECTION",
                        request_id=req.request_id,
                        message=f"Rejected {req.request_id}. Reason: {reason}",
                        remaining_budget_ms=req.remaining_budget(self.current_time_ms),
                    )
                    self.processed_requests.append(req)

                self.max_queue_length = max(self.max_queue_length, len(self.queue))

            elif next_completion_time < float("inf"):
                # Handle Task Completion Event
                self.current_time_ms = next_completion_time
                self._complete_current_task()
                self.max_queue_length = max(self.max_queue_length, len(self.queue))
            else:
                break

        # Re-sort processed requests to original arrival order
        id_to_order = {r.request_id: idx for idx, r in enumerate(requests)}
        self.processed_requests.sort(key=lambda r: id_to_order.get(r.request_id, 0))
        return self.processed_requests

    def compute_summary_metrics(self) -> dict[str, Any]:
        """Compute comprehensive and mathematically verified evaluation metrics."""
        total = len(self.processed_requests)
        if total == 0:
            return {}

        admitted = [r for r in self.processed_requests if r.status != RequestStatus.REJECTED_INFEASIBLE]
        rejected = [r for r in self.processed_requests if r.status == RequestStatus.REJECTED_INFEASIBLE]
        on_time = [r for r in self.processed_requests if r.status == RequestStatus.COMPLETED_ON_TIME]
        missed = [r for r in self.processed_requests if r.status == RequestStatus.DEADLINE_MISSED]
        misses_on_gpu = [r for r in missed if r.start_time is not None]
        misses_in_queue = [r for r in missed if r.start_time is None]
        pending_requests = len(self.queue)

        dispatched = [r for r in self.processed_requests if r.start_time is not None]

        total_hit_rate = (len(on_time) / total) * 100.0
        disp_hit_rate = (len(on_time) / len(dispatched) * 100.0) if dispatched else 0.0
        rejection_rate = (len(rejected) / total) * 100.0

        # Response times and queue waiting times for completed requests
        executed = [r for r in self.processed_requests if r.completion_time is not None]
        response_times = [r.response_time for r in executed if r.response_time is not None]
        wait_times = [r.queue_waiting_time for r in executed if r.queue_waiting_time is not None]

        mean_resp = sum(response_times) / len(response_times) if response_times else 0.0
        p95_resp = (
            sorted(response_times)[int(math.ceil(0.95 * len(response_times))) - 1]
            if response_times
            else 0.0
        )
        mean_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0

        # Mean validation mIoU of dispatched models
        miou_vals = []
        for r in dispatched:
            if r.selected_configuration and r.selected_configuration in self.candidates:
                miou_vals.append(self.candidates[r.selected_configuration].validation_miou)
        mean_miou = sum(miou_vals) / len(miou_vals) if miou_vals else 0.0

        total_sim_time = self.total_gpu_busy_time_ms + self.total_gpu_idle_time_ms
        utilization = (self.total_gpu_busy_time_ms / total_sim_time * 100.0) if total_sim_time > 0 else 0.0

        reconciled_sum = len(on_time) + len(misses_on_gpu) + len(misses_in_queue) + len(rejected) + pending_requests
        reconciliation_valid = (reconciled_sum == total)

        return {
            "policy": self.policy.value,
            "total_requests": total,
            "admitted_requests": len(admitted),
            "rejected_requests": len(rejected),
            "dispatched_requests": len(dispatched),
            "on_time_completions": len(on_time),
            "deadline_misses": len(missed),
            "misses_on_gpu": len(misses_on_gpu),
            "misses_in_queue": len(misses_in_queue),
            "pending_requests": pending_requests,
            "reconciled_sum": reconciled_sum,
            "reconciliation_valid": reconciliation_valid,
            "total_arrival_hit_rate_pct": round(total_hit_rate, 2),
            "dispatched_hit_rate_pct": round(disp_hit_rate, 2),
            "rejection_rate_pct": round(rejection_rate, 2),
            "mean_response_time_ms": round(mean_resp, 2),
            "p95_response_time_ms": round(p95_resp, 2),
            "mean_queue_waiting_time_ms": round(mean_wait, 2),
            "max_queue_length": self.max_queue_length,
            "gpu_busy_time_ms": round(self.total_gpu_busy_time_ms, 2),
            "gpu_busy_time_s": round(self.total_gpu_busy_time_ms / 1000.0, 4),
            "gpu_idle_time_ms": round(self.total_gpu_idle_time_ms, 2),
            "gpu_utilization_pct": round(utilization, 2),
            "mean_selected_miou": round(mean_miou, 6),
        }
