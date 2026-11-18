"""Command-line interface for the deadline-aware inference-mode controller.

Usage examples (from project root)::

    .\.venv\Scripts\python.exe -m realtime.run_controller --deadline-ms 25 --metric p95
    .\.venv\Scripts\python.exe -m realtime.run_controller --deadline-ms 15 --metric p99
    .\.venv\Scripts\python.exe -m realtime.run_controller --deadline-ms 10 --metric max_observed
    .\.venv\Scripts\python.exe -m realtime.run_controller --deadline-ms 25 --benchmark custom.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from realtime.controller import (
    ControllerError,
    SelectionResult,
    load_benchmark_modes,
    select_mode,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "outputs" / "inference_benchmark" / "phase6_finetuned_results.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deadline-aware adaptive inference-mode controller. "
            "Selects the best DINOv2 inference mode for a given "
            "remaining deadline using empirical latency bounds."
        ),
    )
    parser.add_argument(
        "--deadline-ms",
        type=float,
        required=True,
        help="Remaining deadline in milliseconds (non-negative, finite).",
    )
    parser.add_argument(
        "--metric",
        choices=["p95", "p99", "max_observed"],
        default="p95",
        help="Empirical latency metric to use as the feasibility bound (default: p95).",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help=f"Path to the Phase 6B benchmark JSON (default: {DEFAULT_BENCHMARK}).",
    )
    return parser


def format_result(result: SelectionResult) -> str:
    """Format a SelectionResult for human-readable CLI output."""
    lines: list[str] = [
        "=" * 70,
        "  Deadline-Aware Inference-Mode Controller — Selection Report",
        "=" * 70,
        "",
        f"  Remaining deadline:     {result.remaining_deadline_ms:.3f} ms",
        f"  Latency metric:         {result.latency_metric}",
        "",
        "  Mode latency bounds ({metric}):".format(metric=result.latency_metric),
    ]
    for name, bound in sorted(result.all_mode_bounds.items()):
        tag = ""
        if name == result.selected_mode:
            tag = "  <-- SELECTED"
        feasible_tag = "feasible" if name in result.feasible_modes else "EXCEEDS deadline"
        lines.append(
            f"    {name:25s}  {bound:10.3f} ms  ({feasible_tag}){tag}"
        )
    lines.extend(
        [
            "",
            f"  Feasible modes:         {result.feasible_modes if result.feasible_modes else '(none)'}",
            f"  Selected mode:          {result.selected_mode}",
            f"  Selected mIoU:          {result.selected_miou:.6f}",
            f"  Empirical latency bound:{result.selected_empirical_latency_bound_ms:10.3f} ms",
            f"  Likely deadline miss:   {result.likely_deadline_miss}",
            "",
        ]
    )
    if result.likely_deadline_miss:
        lines.append(
            "  WARNING: No mode's empirical bound fits the deadline."
        )
        lines.append(
            "  The controller selected the fastest available mode as a fallback."
        )
        lines.append(
            "  An actual deadline miss is likely but not certain — empirical"
        )
        lines.append(
            "  bounds are statistical, not formal WCET guarantees."
        )
    else:
        lines.append(
            "  The selected mode's empirical bound fits the deadline."
        )
        lines.append(
            "  This is not a hard real-time guarantee — empirical bounds are"
        )
        lines.append(
            "  statistical observations, not formal WCET."
        )
    lines.extend(
        [
            "",
            f"  Reason: {result.reason}",
            "",
            "=" * 70,
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns 0 on success, 1 on error."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        modes = load_benchmark_modes(args.benchmark)
        result = select_mode(args.deadline_ms, modes, args.metric)
    except ControllerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
