"""Accuracy-constrained deadline-aware inference controller for Vision Transformers.

Extends the Phase 7 deadline controller with an explicit minimum accuracy constraint:
  - Quality constraint: candidate mIoU >= min_miou_threshold (tau_miou)
  - Deadline constraint: candidate latency_bound <= remaining_deadline_ms
  - Selection criterion: among feasible candidates, select highest validation mIoU.
    Tie-breaking: lowest empirical latency bound -> lexicographical name.
  - Fallback policy: configurable when no candidate is both deadline-feasible and
    quality-feasible. Supported fallback modes:
      * 'highest_quality_miss': Pick highest mIoU regardless of deadline miss.
      * 'fastest_miss': Pick fastest candidate to minimize deadline overshoot.
      * 'reject': Explicitly mark request as infeasible/rejected without execution.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from realtime.controller import (
    SUPPORTED_METRICS,
    ControllerError,
    ModeProfile,
    _validate_finite_nonneg,
)

# Supported fallback policies
SUPPORTED_FALLBACK_POLICIES = {
    "highest_quality_miss",
    "fastest_miss",
    "reject",
}


@dataclass(frozen=True)
class AccuracyCandidateProfile:
    """Latency, accuracy, and metadata profile for an inference candidate."""

    name: str
    validation_miou: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_observed_ms: float
    median_ms: float
    blocks: list[int] = field(default_factory=list)

    def latency_for_metric(self, metric: str) -> float:
        """Return the latency value for the named metric."""
        if metric == "p90":
            return self.p90_ms
        if metric == "median":
            return self.median_ms
        field_name = SUPPORTED_METRICS.get(metric)
        if field_name is None:
            raise ControllerError(
                f"Unsupported latency metric: {metric!r}. "
                f"Supported: {sorted(list(SUPPORTED_METRICS.keys()) + ['p90', 'median'])}"
            )
        return getattr(self, field_name)


@dataclass(frozen=True)
class AccuracySelectionResult:
    """Structured decision returned by the accuracy-constrained scheduler."""

    selected_candidate: str | None
    selected_miou: float | None
    selected_latency_bound_ms: float | None
    remaining_deadline_ms: float
    min_miou_threshold: float
    latency_metric: str
    fallback_policy: str
    quality_eligible_candidates: list[str]
    deadline_eligible_candidates: list[str]
    jointly_feasible_candidates: list[str]
    is_infeasible: bool
    is_fallback: bool
    is_rejected: bool
    reason: str
    all_candidate_latencies: dict[str, float]
    all_candidate_mious: dict[str, float]


def load_accuracy_candidates(benchmark_path: Path | str) -> list[AccuracyCandidateProfile]:
    """Load validated candidate profiles from a benchmark JSON file."""
    path = Path(benchmark_path)
    if not path.is_file():
        raise ControllerError(f"Benchmark file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ControllerError(f"Failed to read benchmark file {path}: {e}") from e

    modes_data = data.get("modes")
    if not isinstance(modes_data, dict) or not modes_data:
        raise ControllerError(f"Benchmark file must contain a non-empty 'modes' dict: {path}")

    profiles: list[AccuracyCandidateProfile] = []
    for mode_name, mode_info in modes_data.items():
        if not isinstance(mode_info, dict):
            raise ControllerError(f"Entry for mode {mode_name!r} must be a dict")

        miou = mode_info.get("validation_miou")
        if miou is None or not isinstance(miou, (int, float)) or math.isnan(miou):
            raise ControllerError(f"Missing or invalid validation_miou for {mode_name!r}")

        # Latency statistics
        p90 = mode_info.get("p90_ms")
        p95 = mode_info.get("p95_ms")
        p99 = mode_info.get("p99_ms")
        max_obs = mode_info.get("max_observed_ms")
        median = mode_info.get("median_ms")

        for val, label in [
            (p90, "p90_ms"),
            (p95, "p95_ms"),
            (p99, "p99_ms"),
            (max_obs, "max_observed_ms"),
            (median, "median_ms"),
        ]:
            if val is None or not isinstance(val, (int, float)) or math.isnan(val) or val < 0:
                raise ControllerError(f"Missing or invalid {label} for mode {mode_name!r}")

        blocks = mode_info.get("blocks", [])
        profiles.append(
            AccuracyCandidateProfile(
                name=mode_name,
                validation_miou=float(miou),
                p90_ms=float(p90),
                p95_ms=float(p95),
                p99_ms=float(p99),
                max_observed_ms=float(max_obs),
                median_ms=float(median),
                blocks=list(blocks),
            )
        )

    return profiles


def select_accuracy_constrained_mode(
    remaining_deadline_ms: float,
    candidates: list[AccuracyCandidateProfile],
    min_miou_threshold: float,
    latency_metric: str = "p95",
    fallback_policy: str = "highest_quality_miss",
) -> AccuracySelectionResult:
    """Select the best candidate subject to both accuracy and deadline feasibility constraints.

    Rules:
      1. Quality eligibility: candidate.validation_miou >= min_miou_threshold
      2. Deadline eligibility: candidate.latency_for_metric(latency_metric) <= remaining_deadline_ms
      3. Feasible set: candidates satisfying BOTH (1) and (2).
      4. If feasible set is non-empty:
         - Select candidate maximizing validation_miou.
         - Tie-break: lowest latency bound, then lexicographically by name.
      5. If feasible set is empty:
         - is_infeasible = True, is_fallback = True
         - Follow configured fallback_policy:
           * 'reject': selected_candidate = None, is_rejected = True.
           * 'highest_quality_miss': Pick candidate with highest validation_miou (regardless of deadline).
           * 'fastest_miss': Pick candidate with lowest latency bound (regardless of quality).
    """
    _validate_finite_nonneg(remaining_deadline_ms, "remaining_deadline_ms")
    _validate_finite_nonneg(min_miou_threshold, "min_miou_threshold")

    if not (0.0 <= min_miou_threshold <= 1.0):
        raise ControllerError(
            f"min_miou_threshold must be in range [0.0, 1.0], got {min_miou_threshold}"
        )

    if not candidates:
        raise ControllerError("candidates list must not be empty")

    if fallback_policy not in SUPPORTED_FALLBACK_POLICIES:
        raise ControllerError(
            f"Unsupported fallback_policy: {fallback_policy!r}. "
            f"Supported: {sorted(SUPPORTED_FALLBACK_POLICIES)}"
        )

    all_latencies: dict[str, float] = {}
    all_mious: dict[str, float] = {}
    for c in candidates:
        all_latencies[c.name] = c.latency_for_metric(latency_metric)
        all_mious[c.name] = c.validation_miou

    quality_eligible = [
        c for c in candidates if c.validation_miou >= min_miou_threshold
    ]
    deadline_eligible = [
        c for c in candidates if all_latencies[c.name] <= remaining_deadline_ms
    ]
    jointly_feasible = [
        c for c in candidates if c in quality_eligible and c in deadline_eligible
    ]

    quality_names = sorted(c.name for c in quality_eligible)
    deadline_names = sorted(c.name for c in deadline_eligible)
    feasible_names = sorted(c.name for c in jointly_feasible)

    if jointly_feasible:
        # Feasible candidates available.
        # Maximize mIoU -> minimize latency bound -> sort by name
        best = min(
            jointly_feasible,
            key=lambda c: (-c.validation_miou, all_latencies[c.name], c.name),
        )
        return AccuracySelectionResult(
            selected_candidate=best.name,
            selected_miou=best.validation_miou,
            selected_latency_bound_ms=all_latencies[best.name],
            remaining_deadline_ms=remaining_deadline_ms,
            min_miou_threshold=min_miou_threshold,
            latency_metric=latency_metric,
            fallback_policy=fallback_policy,
            quality_eligible_candidates=quality_names,
            deadline_eligible_candidates=deadline_names,
            jointly_feasible_candidates=feasible_names,
            is_infeasible=False,
            is_fallback=False,
            is_rejected=False,
            reason=(
                f"Selected {best.name} (mIoU={best.validation_miou:.4f} >= {min_miou_threshold:.4f}, "
                f"latency={all_latencies[best.name]:.2f}ms <= {remaining_deadline_ms:.2f}ms)"
            ),
            all_candidate_latencies=all_latencies,
            all_candidate_mious=all_mious,
        )

    # Infeasible: apply fallback policy
    if fallback_policy == "reject":
        return AccuracySelectionResult(
            selected_candidate=None,
            selected_miou=None,
            selected_latency_bound_ms=None,
            remaining_deadline_ms=remaining_deadline_ms,
            min_miou_threshold=min_miou_threshold,
            latency_metric=latency_metric,
            fallback_policy=fallback_policy,
            quality_eligible_candidates=quality_names,
            deadline_eligible_candidates=deadline_names,
            jointly_feasible_candidates=[],
            is_infeasible=True,
            is_fallback=True,
            is_rejected=True,
            reason=(
                f"Infeasible request rejected: no candidate meets both quality threshold "
                f"(tau={min_miou_threshold:.4f}) and deadline ({remaining_deadline_ms:.2f}ms)"
            ),
            all_candidate_latencies=all_latencies,
            all_candidate_mious=all_mious,
        )

    if fallback_policy == "highest_quality_miss":
        # Prefer candidates that meet quality threshold if any exist; otherwise highest mIoU overall
        pool = quality_eligible if quality_eligible else candidates
        fallback_cand = min(
            pool,
            key=lambda c: (-c.validation_miou, all_latencies[c.name], c.name),
        )
        return AccuracySelectionResult(
            selected_candidate=fallback_cand.name,
            selected_miou=fallback_cand.validation_miou,
            selected_latency_bound_ms=all_latencies[fallback_cand.name],
            remaining_deadline_ms=remaining_deadline_ms,
            min_miou_threshold=min_miou_threshold,
            latency_metric=latency_metric,
            fallback_policy=fallback_policy,
            quality_eligible_candidates=quality_names,
            deadline_eligible_candidates=deadline_names,
            jointly_feasible_candidates=[],
            is_infeasible=True,
            is_fallback=True,
            is_rejected=False,
            reason=(
                f"Infeasible fallback (highest_quality_miss): dispatched {fallback_cand.name} "
                f"(mIoU={fallback_cand.validation_miou:.4f}) despite deadline overshoot "
                f"({all_latencies[fallback_cand.name]:.2f}ms > {remaining_deadline_ms:.2f}ms)"
            ),
            all_candidate_latencies=all_latencies,
            all_candidate_mious=all_mious,
        )

    if fallback_policy == "fastest_miss":
        # Pick the fastest candidate to minimize deadline miss
        fallback_cand = min(
            candidates,
            key=lambda c: (all_latencies[c.name], -c.validation_miou, c.name),
        )
        return AccuracySelectionResult(
            selected_candidate=fallback_cand.name,
            selected_miou=fallback_cand.validation_miou,
            selected_latency_bound_ms=all_latencies[fallback_cand.name],
            remaining_deadline_ms=remaining_deadline_ms,
            min_miou_threshold=min_miou_threshold,
            latency_metric=latency_metric,
            fallback_policy=fallback_policy,
            quality_eligible_candidates=quality_names,
            deadline_eligible_candidates=deadline_names,
            jointly_feasible_candidates=[],
            is_infeasible=True,
            is_fallback=True,
            is_rejected=False,
            reason=(
                f"Infeasible fallback (fastest_miss): dispatched fastest candidate {fallback_cand.name} "
                f"({all_latencies[fallback_cand.name]:.2f}ms) to minimize deadline overshoot"
            ),
            all_candidate_latencies=all_latencies,
            all_candidate_mious=all_mious,
        )

    raise ControllerError(f"Unhandled fallback policy: {fallback_policy}")
