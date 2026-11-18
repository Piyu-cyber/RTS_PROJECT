"""Request data model and outcome lifecycle for real-time inference."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Any, Dict


class RequestOutcome(str, enum.Enum):
    """Possible terminal outcomes for a real-time request."""
    ON_TIME = "ON_TIME"             # Dispatched and completed before or at absolute deadline
    LATE = "LATE"                   # Dispatched and completed after absolute deadline
    EXPIRED = "EXPIRED"             # Expired while waiting in queue before dispatch
    REJECTED = "REJECTED"           # Rejected by admission control upon arrival


class RequestState(str, enum.Enum):
    """Lifecycle states of a request."""
    ARRIVED = "ARRIVED"
    QUEUED = "QUEUED"
    DISPATCHED = "DISPATCHED"
    TERMINATED = "TERMINATED"


@dataclass
class Request:
    """Represents an incoming vision inference real-time request."""

    request_id: str
    arrival_time_ms: float
    relative_deadline_ms: float
    absolute_deadline_ms: float = field(init=False)

    # Optional payload metadata for dataset/GPU execution
    image_idx: Optional[int] = None
    image_name: Optional[str] = None
    ground_truth_label: Optional[int] = None

    # Scheduling and dispatch attributes
    selected_model: Optional[str] = None
    estimated_exec_time_ms: Optional[float] = None
    actual_exec_time_ms: Optional[float] = None
    dispatch_time_ms: Optional[float] = None
    completion_time_ms: Optional[float] = None

    # Lifecycle state and outcome
    state: RequestState = RequestState.ARRIVED
    outcome: Optional[RequestOutcome] = None
    rejection_reason: Optional[str] = None

    # Prediction outputs
    predicted_label: Optional[int] = None
    is_correct: Optional[bool] = None

    def __post_init__(self):
        self.absolute_deadline_ms = round(self.arrival_time_ms + self.relative_deadline_ms, 4)

    @property
    def waiting_time_ms(self) -> Optional[float]:
        """Time spent waiting in queue before execution."""
        if self.dispatch_time_ms is not None:
            return round(self.dispatch_time_ms - self.arrival_time_ms, 4)
        return None

    @property
    def response_time_ms(self) -> Optional[float]:
        """Total turnaround time from arrival to completion."""
        if self.completion_time_ms is not None:
            return round(self.completion_time_ms - self.arrival_time_ms, 4)
        return None

    @property
    def lateness_ms(self) -> Optional[float]:
        """Lateness: Completion - Absolute Deadline (positive means late)."""
        if self.completion_time_ms is not None:
            return round(self.completion_time_ms - self.absolute_deadline_ms, 4)
        return None

    @property
    def remaining_slack_ms(self) -> Optional[float]:
        """Slack at arrival assuming default estimated execution time."""
        if self.estimated_exec_time_ms is not None:
            return round(self.relative_deadline_ms - self.estimated_exec_time_ms, 4)
        return None

    def mark_rejected(self, timestamp_ms: float, reason: str = "Infeasible at arrival"):
        """Mark request as rejected by admission control."""
        self.state = RequestState.TERMINATED
        self.outcome = RequestOutcome.REJECTED
        self.rejection_reason = reason

    def mark_expired(self, timestamp_ms: float, reason: str = "Deadline expired while queued"):
        """Mark request as expired in queue before dispatch."""
        self.state = RequestState.TERMINATED
        self.outcome = RequestOutcome.EXPIRED
        self.rejection_reason = reason

    def mark_dispatched(self, timestamp_ms: float, model_id: str, estimated_ms: float):
        """Mark request as dispatched onto the GPU."""
        self.state = RequestState.DISPATCHED
        self.dispatch_time_ms = timestamp_ms
        self.selected_model = model_id
        self.estimated_exec_time_ms = estimated_ms

    def mark_completed(
        self,
        timestamp_ms: float,
        actual_exec_ms: float,
        predicted_label: Optional[int] = None,
    ):
        """Mark request as completed and record outcome."""
        self.state = RequestState.TERMINATED
        self.completion_time_ms = timestamp_ms
        self.actual_exec_time_ms = actual_exec_ms
        self.predicted_label = predicted_label

        if self.ground_truth_label is not None and predicted_label is not None:
            self.is_correct = (predicted_label == self.ground_truth_label)

        if self.completion_time_ms <= self.absolute_deadline_ms + 1e-6:
            self.outcome = RequestOutcome.ON_TIME
        else:
            self.outcome = RequestOutcome.LATE

    def to_dict(self) -> Dict[str, Any]:
        """Convert request record to dictionary for logging and CSV export."""
        return {
            "request_id": self.request_id,
            "arrival_time_ms": self.arrival_time_ms,
            "relative_deadline_ms": self.relative_deadline_ms,
            "absolute_deadline_ms": self.absolute_deadline_ms,
            "selected_model": self.selected_model or "None",
            "estimated_exec_time_ms": self.estimated_exec_time_ms,
            "actual_exec_time_ms": self.actual_exec_time_ms,
            "dispatch_time_ms": self.dispatch_time_ms,
            "completion_time_ms": self.completion_time_ms,
            "waiting_time_ms": self.waiting_time_ms,
            "response_time_ms": self.response_time_ms,
            "lateness_ms": self.lateness_ms,
            "outcome": self.outcome.value if self.outcome else "None",
            "rejection_reason": self.rejection_reason or "",
            "ground_truth_label": self.ground_truth_label,
            "predicted_label": self.predicted_label,
            "is_correct": self.is_correct,
        }
