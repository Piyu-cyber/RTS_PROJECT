"""Bound-aware and empirical upper-bound inference scheduler for Real-Time Vision Transformers.

Extends the Phase 10C/12C accuracy scheduler with explicit bound-aware semantics:
  1. Multiple Scheduling Modes:
     - 'empirical_latency': Uses empirical percentile statistics (p90, p95, p99, median).
     - 'empirical_upper_bound': Uses documented conservative empirical upper bounds
       (max_observed + 3*round_std), clearly designated as an empirical estimate, NOT WCET.
     - 'bound_aware': Strictly requires a formally verified WCET bound. Since Phase 16
       established Outcome C (no formal WCET on consumer GPU/WDDM), invoking this mode
       without verified formal bounds raises FormalBoundUnavailableError.
  2. Timing Scopes:
     - 'model_forward': Model forward execution latency only.
     - 'end_to_end': Complete request response time (forward + host/transfer offset).
  3. Absolute Deadlines and Budget Calculation:
     - Supports B_i(t) = D_i - t, accounting for queue waiting delay W_i.
     - Feasibility check: W_i + C_c <= B_i(t).
  4. Explicit Quality Gating and Distinction:
     - Distinguishes quality floor violations (C_quality == empty) from deadline infeasibility.
     - Distinguishes predicted infeasibility (empirical) from proven infeasibility (formal).
  5. Configurable Fallbacks:
     - 'reject': Proactive load shedding without GPU dispatch.
     - 'highest_quality_miss': Maximizes task utility despite timing miss.
     - 'fastest_miss': Minimizes deadline overshoot.
"""

from __future__ import annotations

import enum
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ControllerError(Exception):
    """Base exception for controller and scheduler configuration errors."""


class FormalBoundUnavailableError(ControllerError):
    """Raised when a formal bound-aware mode is requested without verified formal WCET metadata."""


class SchedulingMode(str, enum.Enum):
    EMPIRICAL_LATENCY = "empirical_latency"
    EMPIRICAL_UPPER_BOUND = "empirical_upper_bound"
    BOUND_AWARE = "bound_aware"


class TimingScope(str, enum.Enum):
    MODEL_FORWARD = "model_forward"
    END_TO_END = "end_to_end"


SUPPORTED_FALLBACK_POLICIES = {
    "highest_quality_miss",
    "fastest_miss",
    "reject",
}

SUPPORTED_METRICS = {
    "p90",
    "p95",
    "p99",
    "median",
    "max_observed",
}


@dataclass(frozen=True)
class BoundCandidateProfile:
    """Complete candidate profile containing accuracy, empirical percentiles, and bounds."""

    name: str
    checkpoint_path: str
    validation_miou: float
    input_shape: tuple[int, ...]
    model_forward_latencies: dict[str, float]
    end_to_end_latencies: dict[str, float]
    empirical_upper_bound_forward_ms: float
    empirical_upper_bound_e2e_ms: float
    formal_wcet_forward_ms: float | None = None
    formal_wcet_e2e_ms: float | None = None
    is_formally_justified: bool = False
    operating_assumptions: list[str] = field(default_factory=list)
    source: str = ""
    blocks: list[int] = field(default_factory=list)

    def timing_value(
        self,
        mode: SchedulingMode | str,
        scope: TimingScope | str = TimingScope.MODEL_FORWARD,
        metric: str = "p95",
    ) -> float:
        """Extract the timing value based on scheduling mode and scope."""
        mode_enum = SchedulingMode(mode)
        scope_enum = TimingScope(scope)

        if mode_enum == SchedulingMode.BOUND_AWARE:
            if not self.is_formally_justified:
                raise FormalBoundUnavailableError(
                    f"Candidate {self.name!r} lacks formally justified WCET metadata. "
                    "Phase 16 audit established Outcome C: formal WCET cannot be guaranteed on this platform."
                )
            val = (
                self.formal_wcet_forward_ms
                if scope_enum == TimingScope.MODEL_FORWARD
                else self.formal_wcet_e2e_ms
            )
            if val is None:
                raise FormalBoundUnavailableError(
                    f"Formal WCET bound is None for candidate {self.name!r} under scope {scope_enum.value}"
                )
            return val

        if mode_enum == SchedulingMode.EMPIRICAL_UPPER_BOUND:
            return (
                self.empirical_upper_bound_forward_ms
                if scope_enum == TimingScope.MODEL_FORWARD
                else self.empirical_upper_bound_e2e_ms
            )

        if mode_enum == SchedulingMode.EMPIRICAL_LATENCY:
            lat_dict = (
                self.model_forward_latencies
                if scope_enum == TimingScope.MODEL_FORWARD
                else self.end_to_end_latencies
            )
            if metric not in lat_dict:
                raise ControllerError(
                    f"Unsupported or missing metric {metric!r} in {scope_enum.value} latencies for {self.name!r}. "
                    f"Available: {sorted(lat_dict.keys())}"
                )
            return lat_dict[metric]

        raise ControllerError(f"Unhandled scheduling mode: {mode}")


@dataclass(frozen=True)
class BoundAwareSelectionResult:
    """Structured decision returned by the bound-aware scheduler."""

    request_id: str
    scheduling_mode: str
    timing_scope: str
    arrival_time: float | None
    current_time: float | None
    absolute_deadline: float | None
    remaining_budget_ms: float
    queue_delay_ms: float
    min_miou_threshold: float
    latency_metric: str
    fallback_policy: str
    quality_eligible_candidates: list[str]
    deadline_eligible_candidates: list[str]
    jointly_feasible_candidates: list[str]
    selected_candidate: str | None
    selected_miou: float | None
    selected_timing_value_ms: float | None
    is_infeasible: bool
    is_quality_floor_violated: bool
    is_rejected: bool
    is_fallback: bool
    predicted_infeasible: bool
    proven_infeasible: bool
    reason: str
    all_candidate_timing_values: dict[str, float]
    all_candidate_mious: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "scheduling_mode": self.scheduling_mode,
            "timing_scope": self.timing_scope,
            "arrival_time": self.arrival_time,
            "current_time": self.current_time,
            "absolute_deadline": self.absolute_deadline,
            "remaining_budget_ms": self.remaining_budget_ms,
            "queue_delay_ms": self.queue_delay_ms,
            "min_miou_threshold": self.min_miou_threshold,
            "latency_metric": self.latency_metric,
            "fallback_policy": self.fallback_policy,
            "quality_eligible_candidates": self.quality_eligible_candidates,
            "deadline_eligible_candidates": self.deadline_eligible_candidates,
            "jointly_feasible_candidates": self.jointly_feasible_candidates,
            "selected_candidate": self.selected_candidate,
            "selected_miou": self.selected_miou,
            "selected_timing_value_ms": self.selected_timing_value_ms,
            "is_infeasible": self.is_infeasible,
            "is_quality_floor_violated": self.is_quality_floor_violated,
            "is_rejected": self.is_rejected,
            "is_fallback": self.is_fallback,
            "predicted_infeasible": self.predicted_infeasible,
            "proven_infeasible": self.proven_infeasible,
            "reason": self.reason,
            "all_candidate_timing_values": self.all_candidate_timing_values,
            "all_candidate_mious": self.all_candidate_mious,
        }


def load_bound_candidates(profiles_path: Path | str) -> list[BoundCandidateProfile]:
    """Load bound-aware candidate profiles from the Phase 16 JSON artifact."""
    path = Path(profiles_path)
    if not path.is_file():
        raise ControllerError(f"Profiles file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ControllerError(f"Failed to read profiles file {path}: {e}") from e

    candidates_dict = data.get("candidates")
    if not isinstance(candidates_dict, dict) or not candidates_dict:
        raise ControllerError(f"Profiles file must contain a non-empty 'candidates' dict: {path}")

    profiles: list[BoundCandidateProfile] = []
    for name, c_info in candidates_dict.items():
        if not isinstance(c_info, dict):
            raise ControllerError(f"Candidate entry for {name!r} must be a dict")

        miou = c_info.get("validation_miou")
        if miou is None or not isinstance(miou, (int, float)) or math.isnan(miou):
            raise ControllerError(f"Invalid validation_miou for candidate {name!r}")

        fwd_dict = c_info.get("model_forward", {})
        e2e_dict = c_info.get("end_to_end", {})

        eub_fwd = fwd_dict.get("empirical_upper_bound_ms")
        eub_e2e = e2e_dict.get("empirical_upper_bound_ms")
        if eub_fwd is None or eub_e2e is None:
            raise ControllerError(f"Missing empirical_upper_bound_ms in candidate {name!r}")

        formal_fwd = c_info.get("formal_wcet_forward_ms")
        formal_e2e = c_info.get("formal_wcet_e2e_ms")
        is_formal = bool(c_info.get("is_formally_justified", False))

        profiles.append(
            BoundCandidateProfile(
                name=name,
                checkpoint_path=str(c_info.get("checkpoint_path", "")),
                validation_miou=float(miou),
                input_shape=tuple(data.get("metadata", {}).get("input_resolution", [224, 224])),
                model_forward_latencies={
                    k.replace("_ms", ""): float(v)
                    for k, v in fwd_dict.items()
                    if isinstance(v, (int, float)) and not math.isnan(v)
                },
                end_to_end_latencies={
                    k.replace("_ms", ""): float(v)
                    for k, v in e2e_dict.items()
                    if isinstance(v, (int, float)) and not math.isnan(v)
                },
                empirical_upper_bound_forward_ms=float(eub_fwd),
                empirical_upper_bound_e2e_ms=float(eub_e2e),
                formal_wcet_forward_ms=float(formal_fwd) if formal_fwd is not None else None,
                formal_wcet_e2e_ms=float(formal_e2e) if formal_e2e is not None else None,
                is_formally_justified=is_formal,
                operating_assumptions=list(c_info.get("operating_assumptions", [])),
                source=str(c_info.get("source", "")),
                blocks=list(c_info.get("blocks", [])),
            )
        )

    return profiles


def select_bound_aware_candidate(
    candidates: list[BoundCandidateProfile],
    min_miou_threshold: float,
    remaining_budget_ms: float | None = None,
    absolute_deadline: float | None = None,
    current_time: float | None = None,
    arrival_time: float | None = None,
    queue_delay_ms: float = 0.0,
    scheduling_mode: SchedulingMode | str = SchedulingMode.EMPIRICAL_LATENCY,
    timing_scope: TimingScope | str = TimingScope.MODEL_FORWARD,
    latency_metric: str = "p95",
    fallback_policy: str = "reject",
    request_id: str = "req-001",
) -> BoundAwareSelectionResult:
    """Schedule an inference request subject to quality and deadline feasibility constraints."""
    if not candidates:
        raise ControllerError("candidates list must not be empty")

    if fallback_policy not in SUPPORTED_FALLBACK_POLICIES:
        raise ControllerError(
            f"Unsupported fallback_policy: {fallback_policy!r}. "
            f"Supported: {sorted(SUPPORTED_FALLBACK_POLICIES)}"
        )

    mode_enum = SchedulingMode(scheduling_mode)
    scope_enum = TimingScope(timing_scope)

    if not (0.0 <= min_miou_threshold <= 1.0):
        raise ControllerError(f"min_miou_threshold must be in range [0.0, 1.0], got {min_miou_threshold}")

    if queue_delay_ms < 0:
        raise ControllerError(f"queue_delay_ms cannot be negative, got {queue_delay_ms}")

    # Determine remaining budget B_i(t)
    if remaining_budget_ms is not None:
        if remaining_budget_ms < 0:
            raise ControllerError(f"remaining_budget_ms cannot be negative, got {remaining_budget_ms}")
        budget_ms = float(remaining_budget_ms)
    elif absolute_deadline is not None and current_time is not None:
        if current_time > absolute_deadline:
            budget_ms = 0.0
        else:
            budget_ms = (absolute_deadline - current_time) * 1000.0  # seconds to ms
    else:
        raise ControllerError(
            "Must provide either remaining_budget_ms or both (absolute_deadline, current_time)"
        )

    # 1. Quality Eligibility: C_Q(i) = {c in C | mIoU(c) >= Q_i}
    quality_eligible = [c for c in candidates if c.validation_miou >= min_miou_threshold]
    is_quality_floor_violated = len(quality_eligible) == 0

    # 2. Timing Estimates & Feasibility Check: W_i + C_c <= B_i(t)
    candidate_timings: dict[str, float] = {}
    all_mious: dict[str, float] = {}
    for c in candidates:
        candidate_timings[c.name] = c.timing_value(mode_enum, scope_enum, latency_metric)
        all_mious[c.name] = c.validation_miou

    deadline_eligible = [
        c for c in candidates if (queue_delay_ms + candidate_timings[c.name]) <= budget_ms
    ]

    # 3. Jointly Feasible Set: F = C_Q(i) intersect C_deadline
    jointly_feasible = [c for c in candidates if c in quality_eligible and c in deadline_eligible]

    quality_names = sorted(c.name for c in quality_eligible)
    deadline_names = sorted(c.name for c in deadline_eligible)
    feasible_names = sorted(c.name for c in jointly_feasible)

    is_proven = (mode_enum == SchedulingMode.BOUND_AWARE)

    if jointly_feasible:
        # Feasible: select candidate with highest validation mIoU, tie-break by lowest timing value
        best = min(
            jointly_feasible,
            key=lambda c: (-c.validation_miou, candidate_timings[c.name], c.name),
        )
        return BoundAwareSelectionResult(
            request_id=request_id,
            scheduling_mode=mode_enum.value,
            timing_scope=scope_enum.value,
            arrival_time=arrival_time,
            current_time=current_time,
            absolute_deadline=absolute_deadline,
            remaining_budget_ms=budget_ms,
            queue_delay_ms=queue_delay_ms,
            min_miou_threshold=min_miou_threshold,
            latency_metric=latency_metric,
            fallback_policy=fallback_policy,
            quality_eligible_candidates=quality_names,
            deadline_eligible_candidates=deadline_names,
            jointly_feasible_candidates=feasible_names,
            selected_candidate=best.name,
            selected_miou=best.validation_miou,
            selected_timing_value_ms=candidate_timings[best.name],
            is_infeasible=False,
            is_quality_floor_violated=False,
            is_rejected=False,
            is_fallback=False,
            predicted_infeasible=False,
            proven_infeasible=False,
            reason=(
                f"Selected {best.name} (mIoU={best.validation_miou:.4f} >= {min_miou_threshold:.4f}, "
                f"timing={candidate_timings[best.name]:.2f}ms + queue={queue_delay_ms:.2f}ms <= {budget_ms:.2f}ms)"
            ),
            all_candidate_timing_values=candidate_timings,
            all_candidate_mious=all_mious,
        )

    # Infeasible: Execute configured fallback
    pred_infeasible = not is_proven
    prov_infeasible = is_proven

    if fallback_policy == "reject":
        if is_quality_floor_violated:
            reason_str = (
                f"Infeasible request rejected: Quality floor violation. No candidate meets "
                f"tau_mIoU={min_miou_threshold:.4f} (best={max(all_mious.values()):.4f})."
            )
        else:
            min_timing = min(candidate_timings.values())
            reason_str = (
                f"Infeasible request rejected: Deadline infeasibility under {mode_enum.value}. "
                f"Minimum required timing ({min_timing:.2f}ms + queue={queue_delay_ms:.2f}ms) "
                f"exceeds remaining budget ({budget_ms:.2f}ms)."
            )

        return BoundAwareSelectionResult(
            request_id=request_id,
            scheduling_mode=mode_enum.value,
            timing_scope=scope_enum.value,
            arrival_time=arrival_time,
            current_time=current_time,
            absolute_deadline=absolute_deadline,
            remaining_budget_ms=budget_ms,
            queue_delay_ms=queue_delay_ms,
            min_miou_threshold=min_miou_threshold,
            latency_metric=latency_metric,
            fallback_policy=fallback_policy,
            quality_eligible_candidates=quality_names,
            deadline_eligible_candidates=deadline_names,
            jointly_feasible_candidates=[],
            selected_candidate=None,
            selected_miou=None,
            selected_timing_value_ms=None,
            is_infeasible=True,
            is_quality_floor_violated=is_quality_floor_violated,
            is_rejected=True,
            is_fallback=True,
            predicted_infeasible=pred_infeasible,
            proven_infeasible=prov_infeasible,
            reason=reason_str,
            all_candidate_timing_values=candidate_timings,
            all_candidate_mious=all_mious,
        )

    if fallback_policy == "highest_quality_miss":
        pool = quality_eligible if quality_eligible else candidates
        fallback_cand = min(
            pool,
            key=lambda c: (-c.validation_miou, candidate_timings[c.name], c.name),
        )
        return BoundAwareSelectionResult(
            request_id=request_id,
            scheduling_mode=mode_enum.value,
            timing_scope=scope_enum.value,
            arrival_time=arrival_time,
            current_time=current_time,
            absolute_deadline=absolute_deadline,
            remaining_budget_ms=budget_ms,
            queue_delay_ms=queue_delay_ms,
            min_miou_threshold=min_miou_threshold,
            latency_metric=latency_metric,
            fallback_policy=fallback_policy,
            quality_eligible_candidates=quality_names,
            deadline_eligible_candidates=deadline_names,
            jointly_feasible_candidates=[],
            selected_candidate=fallback_cand.name,
            selected_miou=fallback_cand.validation_miou,
            selected_timing_value_ms=candidate_timings[fallback_cand.name],
            is_infeasible=True,
            is_quality_floor_violated=is_quality_floor_violated,
            is_rejected=False,
            is_fallback=True,
            predicted_infeasible=pred_infeasible,
            proven_infeasible=prov_infeasible,
            reason=(
                f"Infeasible fallback (highest_quality_miss): dispatched {fallback_cand.name} "
                f"(mIoU={fallback_cand.validation_miou:.4f}) despite timing miss "
                f"({candidate_timings[fallback_cand.name]:.2f}ms > {budget_ms:.2f}ms)"
            ),
            all_candidate_timing_values=candidate_timings,
            all_candidate_mious=all_mious,
        )

    if fallback_policy == "fastest_miss":
        fallback_cand = min(
            candidates,
            key=lambda c: (candidate_timings[c.name], -c.validation_miou, c.name),
        )
        return BoundAwareSelectionResult(
            request_id=request_id,
            scheduling_mode=mode_enum.value,
            timing_scope=scope_enum.value,
            arrival_time=arrival_time,
            current_time=current_time,
            absolute_deadline=absolute_deadline,
            remaining_budget_ms=budget_ms,
            queue_delay_ms=queue_delay_ms,
            min_miou_threshold=min_miou_threshold,
            latency_metric=latency_metric,
            fallback_policy=fallback_policy,
            quality_eligible_candidates=quality_names,
            deadline_eligible_candidates=deadline_names,
            jointly_feasible_candidates=[],
            selected_candidate=fallback_cand.name,
            selected_miou=fallback_cand.validation_miou,
            selected_timing_value_ms=candidate_timings[fallback_cand.name],
            is_infeasible=True,
            is_quality_floor_violated=is_quality_floor_violated,
            is_rejected=False,
            is_fallback=True,
            predicted_infeasible=pred_infeasible,
            proven_infeasible=prov_infeasible,
            reason=(
                f"Infeasible fallback (fastest_miss): dispatched fastest candidate {fallback_cand.name} "
                f"({candidate_timings[fallback_cand.name]:.2f}ms) to minimize overshoot"
            ),
            all_candidate_timing_values=candidate_timings,
            all_candidate_mious=all_mious,
        )

    raise ControllerError(f"Unhandled fallback policy: {fallback_policy}")
