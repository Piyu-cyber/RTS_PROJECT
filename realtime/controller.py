"""Deadline-aware adaptive inference-mode controller.

Selects the best inference mode (full_attention, moderate_dwconv,
aggressive_dwconv) for a given remaining deadline using empirical latency
bounds and validation mIoU from the Phase 6B benchmark.

This module is a deterministic selector.  It does not load neural-network
checkpoints or execute inference.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Supported latency metrics and their JSON field names in the benchmark file.
# ---------------------------------------------------------------------------
SUPPORTED_METRICS: dict[str, str] = {
    "p95": "p95_ms",
    "p99": "p99_ms",
    "max_observed": "max_observed_ms",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModeProfile:
    """Latency and accuracy profile for a single inference mode."""

    name: str
    validation_miou: float
    p95_ms: float
    p99_ms: float
    max_observed_ms: float

    def latency_for_metric(self, metric: str) -> float:
        """Return the latency value for the named metric."""
        field_name = SUPPORTED_METRICS.get(metric)
        if field_name is None:
            raise ControllerError(
                f"Unsupported latency metric: {metric!r}. "
                f"Supported: {sorted(SUPPORTED_METRICS)}"
            )
        return getattr(self, field_name)


@dataclass(frozen=True)
class SelectionResult:
    """Structured result of the deadline-aware mode selection."""

    selected_mode: str
    selected_miou: float
    remaining_deadline_ms: float
    latency_metric: str
    selected_empirical_latency_bound_ms: float
    feasible_modes: list[str]
    likely_deadline_miss: bool
    all_mode_bounds: dict[str, float]
    reason: str


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ControllerError(Exception):
    """Raised for invalid controller inputs or malformed benchmark data."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_finite_nonneg(value: Any, label: str) -> None:
    """Validate that *value* is numeric, finite, and non-negative."""
    if not isinstance(value, (int, float)):
        raise ControllerError(
            f"{label} must be numeric, got {type(value).__name__}"
        )
    if math.isnan(value):
        raise ControllerError(f"{label} must not be NaN")
    if math.isinf(value):
        raise ControllerError(
            f"{label} must be finite, got {'inf' if value > 0 else '-inf'}"
        )
    if value < 0:
        raise ControllerError(f"{label} must be non-negative, got {value}")


def validate_inputs(
    remaining_deadline_ms: float,
    modes: list[ModeProfile],
    latency_metric: str,
) -> None:
    """Validate all controller inputs.  Raises *ControllerError*."""

    # --- Deadline ----------------------------------------------------------
    if not isinstance(remaining_deadline_ms, (int, float)):
        raise ControllerError(
            f"remaining_deadline_ms must be numeric, "
            f"got {type(remaining_deadline_ms).__name__}"
        )
    if math.isnan(remaining_deadline_ms):
        raise ControllerError("remaining_deadline_ms must not be NaN")
    if math.isinf(remaining_deadline_ms):
        raise ControllerError(
            f"remaining_deadline_ms must be finite, got {remaining_deadline_ms}"
        )
    if remaining_deadline_ms < 0:
        raise ControllerError(
            f"remaining_deadline_ms must be non-negative, "
            f"got {remaining_deadline_ms}"
        )

    # --- Metric ------------------------------------------------------------
    if latency_metric not in SUPPORTED_METRICS:
        raise ControllerError(
            f"Unsupported latency metric: {latency_metric!r}. "
            f"Supported: {sorted(SUPPORTED_METRICS)}"
        )

    # --- Modes -------------------------------------------------------------
    if not modes:
        raise ControllerError("Mode collection must not be empty")

    seen_names: set[str] = set()
    for i, mode in enumerate(modes):
        # Name
        if not isinstance(mode.name, str) or not mode.name:
            raise ControllerError(
                f"Mode at index {i}: name must be a non-empty string"
            )
        if mode.name in seen_names:
            raise ControllerError(f"Duplicate mode name: {mode.name!r}")
        seen_names.add(mode.name)

        # Validation mIoU
        if not isinstance(mode.validation_miou, (int, float)):
            raise ControllerError(
                f"Mode {mode.name!r}: validation_miou must be numeric, "
                f"got {type(mode.validation_miou).__name__}"
            )
        if math.isnan(mode.validation_miou):
            raise ControllerError(
                f"Mode {mode.name!r}: validation_miou must not be NaN"
            )
        if math.isinf(mode.validation_miou):
            raise ControllerError(
                f"Mode {mode.name!r}: validation_miou must be finite"
            )
        if not (0.0 <= mode.validation_miou <= 1.0):
            raise ControllerError(
                f"Mode {mode.name!r}: validation_miou must be in [0, 1], "
                f"got {mode.validation_miou}"
            )

        # Latency values (all three must be valid regardless of metric)
        for metric_key, field_name in SUPPORTED_METRICS.items():
            value = getattr(mode, field_name)
            _validate_finite_nonneg(
                value, f"Mode {mode.name!r}: {field_name}"
            )


# ---------------------------------------------------------------------------
# Selection algorithm
# ---------------------------------------------------------------------------
def select_mode(
    remaining_deadline_ms: float,
    modes: list[ModeProfile],
    latency_metric: str,
) -> SelectionResult:
    """Select the best inference mode for the given deadline.

    Algorithm
    ---------
    1. A mode is *feasible* when its empirical latency bound for the
       selected metric is <= the remaining deadline (equality included).
    2. If feasible modes exist, choose the one with highest validation
       mIoU, breaking ties by lower latency bound, then lexicographic name.
    3. If no mode is feasible, fall back to the mode with the lowest
       latency bound, breaking ties by higher mIoU, then lexicographic name.
       Set ``likely_deadline_miss = True``.
    """
    validate_inputs(remaining_deadline_ms, modes, latency_metric)

    # Compute bounds
    all_bounds: dict[str, float] = {
        m.name: m.latency_for_metric(latency_metric) for m in modes
    }

    # Step A — feasibility
    feasible = [
        m
        for m in modes
        if m.latency_for_metric(latency_metric) <= remaining_deadline_ms
    ]
    feasible_names = sorted(m.name for m in feasible)

    if feasible:
        # Step B — best feasible
        selected = min(
            feasible,
            key=lambda m: (
                -m.validation_miou,
                m.latency_for_metric(latency_metric),
                m.name,
            ),
        )
        likely_miss = False
        reason = (
            f"Selected '{selected.name}' as the highest-accuracy feasible "
            f"mode (mIoU={selected.validation_miou:.6f}, "
            f"{latency_metric}={selected.latency_for_metric(latency_metric):.3f} ms "
            f"<= deadline={remaining_deadline_ms:.3f} ms). "
            f"{len(feasible)}/{len(modes)} modes were feasible."
        )
    else:
        # Step C — fallback
        selected = min(
            modes,
            key=lambda m: (
                m.latency_for_metric(latency_metric),
                -m.validation_miou,
                m.name,
            ),
        )
        likely_miss = True
        reason = (
            f"No mode has {latency_metric} <= {remaining_deadline_ms:.3f} ms. "
            f"Falling back to '{selected.name}' with the lowest empirical "
            f"bound ({latency_metric}="
            f"{selected.latency_for_metric(latency_metric):.3f} ms, "
            f"mIoU={selected.validation_miou:.6f}). "
            f"likely_deadline_miss=true."
        )

    return SelectionResult(
        selected_mode=selected.name,
        selected_miou=selected.validation_miou,
        remaining_deadline_ms=remaining_deadline_ms,
        latency_metric=latency_metric,
        selected_empirical_latency_bound_ms=selected.latency_for_metric(
            latency_metric
        ),
        feasible_modes=feasible_names,
        likely_deadline_miss=likely_miss,
        all_mode_bounds=all_bounds,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Benchmark JSON loader
# ---------------------------------------------------------------------------
def load_benchmark_modes(path: Path) -> list[ModeProfile]:
    """Load inference-mode profiles from a Phase 6B benchmark JSON.

    The expected JSON structure (field names taken from the actual file)::

        {
          "modes": {
            "<mode_name>": {
              "accuracy": { "mean_iou": <float> },
              "latency_ms": {
                "p95_ms": <float>,
                "p99_ms": <float>,
                "max_observed_ms": <float>,
                ...
              },
              ...
            },
            ...
          }
        }

    Raises *ControllerError* if the file is missing, malformed, or if
    any required field is absent or has an invalid value.
    """
    if not path.is_file():
        raise ControllerError(f"Benchmark file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ControllerError(f"Cannot read benchmark file {path}: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ControllerError(
            f"Malformed JSON in {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ControllerError(
            f"Benchmark JSON root must be an object, "
            f"got {type(data).__name__}"
        )

    modes_dict = data.get("modes")
    if not isinstance(modes_dict, dict) or not modes_dict:
        raise ControllerError(
            "Benchmark JSON must contain a non-empty 'modes' object"
        )

    profiles: list[ModeProfile] = []
    for mode_name, mode_data in modes_dict.items():
        if not isinstance(mode_data, dict):
            raise ControllerError(
                f"Mode {mode_name!r}: expected an object, "
                f"got {type(mode_data).__name__}"
            )

        # --- accuracy -------------------------------------------------------
        accuracy = mode_data.get("accuracy")
        if not isinstance(accuracy, dict):
            raise ControllerError(
                f"Mode {mode_name!r}: missing or invalid 'accuracy' object"
            )
        miou = accuracy.get("mean_iou")
        if miou is None:
            raise ControllerError(
                f"Mode {mode_name!r}: missing 'accuracy.mean_iou'"
            )
        if not isinstance(miou, (int, float)):
            raise ControllerError(
                f"Mode {mode_name!r}: 'accuracy.mean_iou' must be numeric, "
                f"got {type(miou).__name__}"
            )

        # --- latency_ms -----------------------------------------------------
        latency = mode_data.get("latency_ms")
        if not isinstance(latency, dict):
            raise ControllerError(
                f"Mode {mode_name!r}: missing or invalid 'latency_ms' object"
            )

        latency_values: dict[str, float] = {}
        for metric_key, json_field in SUPPORTED_METRICS.items():
            value = latency.get(json_field)
            if value is None:
                raise ControllerError(
                    f"Mode {mode_name!r}: missing 'latency_ms.{json_field}'"
                )
            if not isinstance(value, (int, float)):
                raise ControllerError(
                    f"Mode {mode_name!r}: 'latency_ms.{json_field}' must be "
                    f"numeric, got {type(value).__name__}"
                )
            latency_values[json_field] = value

        profiles.append(
            ModeProfile(
                name=mode_name,
                validation_miou=miou,
                **latency_values,
            )
        )

    return profiles
