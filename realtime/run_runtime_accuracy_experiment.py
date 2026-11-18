"""Runtime prototype evaluation for Phase 10C accuracy-constrained adaptive inference.

Executes real requests through the accuracy-constrained scheduler on the GPU,
comparing:
  - always_full_attention
  - always_fastest
  - deadline_only_adaptive
  - accuracy_constrained (tau_miou = 0.65, fallback = highest_quality_miss)
  - accuracy_constrained (tau_miou = 0.70, fallback = highest_quality_miss)
  - accuracy_constrained (tau_miou = 0.70, fallback = reject)

Measures component-wise timings on actual hardware:
  - Controller dispatch latency
  - Preprocessing latency
  - Host-to-device transfer latency
  - GPU model forward execution (host monotonic and CUDA event)
  - Postprocessing latency
  - Total end-to-end latency
  - Realized deadline hits and quality compliance
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ["XFORMERS_DISABLED"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REFERENCE_ROOT = PROJECT_ROOT / "reference"
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

import numpy as np
import torch
from PIL import Image

from realtime.accuracy_scheduler import (
    AccuracyCandidateProfile,
    AccuracySelectionResult,
    load_accuracy_candidates,
    select_accuracy_constrained_mode,
)
from realtime.benchmark_candidates_phase10c import (
    compute_summary_stats,
    load_candidate_models,
)
from realtime.runtime_inference import execute_model_forward, preprocess_image
from realtime.simulate_strategies import generate_deadline_stream

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase10c_adaptive_scheduler"
BENCHMARK_PATH = OUTPUT_DIR / "benchmark_results.json"
PET_DATASET_DIR = PROJECT_ROOT / "datasets" / "oxford-iiit-pet" / "images"


@dataclass(frozen=True)
class RuntimeAccuracyResult:
    request_id: str
    policy: str
    tau_miou: float
    fallback_policy: str
    deadline_ms: float
    workload_category: str
    selected_candidate: str
    selected_miou: float
    configured_latency_bound_ms: float
    controller_latency_ms: float
    preprocessing_latency_ms: float
    transfer_latency_ms: float
    model_execution_ms: float
    model_cuda_event_ms: float
    postprocessing_latency_ms: float
    total_request_latency_ms: float
    is_infeasible: bool
    is_fallback: bool
    is_rejected: bool
    quality_compliant: bool
    actual_deadline_hit: bool
    reason: str


def execute_runtime_policy(
    policy_name: str,
    stream: list[Any],
    models: dict[str, tuple[torch.nn.Module, Any]],
    candidates: list[AccuracyCandidateProfile],
    device: torch.device,
    image_paths: list[Path],
    tau_miou: float = 0.0,
    fallback_policy: str = "highest_quality_miss",
) -> list[RuntimeAccuracyResult]:
    results: list[RuntimeAccuracyResult] = []

    for idx, req in enumerate(stream):
        img_path = image_paths[idx % len(image_paths)]
        t_req_0 = time.perf_counter()

        # Controller selection
        t_ctrl_0 = time.perf_counter()
        if policy_name == "always_full_attention":
            cand = next(c for c in candidates if c.name == "full_attention")
            sel_cand = cand.name
            sel_miou = cand.validation_miou
            bound_ms = cand.latency_for_metric("p95")
            infeasible = bound_ms > req.deadline_ms
            fallback = infeasible
            rejected = False
            reason = "Always full attention policy"
        elif policy_name == "always_fastest":
            cand = min(candidates, key=lambda c: (c.latency_for_metric("p95"), -c.validation_miou, c.name))
            sel_cand = cand.name
            sel_miou = cand.validation_miou
            bound_ms = cand.latency_for_metric("p95")
            infeasible = bound_ms > req.deadline_ms
            fallback = infeasible
            rejected = False
            reason = f"Always fastest policy ({cand.name})"
        elif policy_name == "deadline_only_adaptive":
            res = select_accuracy_constrained_mode(
                remaining_deadline_ms=req.deadline_ms,
                candidates=candidates,
                min_miou_threshold=0.0,
                latency_metric="p95",
                fallback_policy="fastest_miss",
            )
            sel_cand = res.selected_candidate
            sel_miou = res.selected_miou
            bound_ms = res.selected_latency_bound_ms
            infeasible = res.is_infeasible
            fallback = res.is_fallback
            rejected = res.is_rejected
            reason = res.reason
        elif policy_name == "accuracy_constrained":
            res = select_accuracy_constrained_mode(
                remaining_deadline_ms=req.deadline_ms,
                candidates=candidates,
                min_miou_threshold=tau_miou,
                latency_metric="p95",
                fallback_policy=fallback_policy,
            )
            sel_cand = res.selected_candidate if res.selected_candidate is not None else "REJECTED"
            sel_miou = res.selected_miou if res.selected_miou is not None else 0.0
            bound_ms = res.selected_latency_bound_ms if res.selected_latency_bound_ms is not None else 0.0
            infeasible = res.is_infeasible
            fallback = res.is_fallback
            rejected = res.is_rejected
            reason = res.reason
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        t_ctrl_1 = time.perf_counter()
        ctrl_ms = (t_ctrl_1 - t_ctrl_0) * 1000.0

        if rejected:
            # Short-circuit on rejected requests
            t_req_1 = time.perf_counter()
            e2e_ms = (t_req_1 - t_req_0) * 1000.0
            results.append(
                RuntimeAccuracyResult(
                    request_id=req.request_id,
                    policy=policy_name,
                    tau_miou=tau_miou,
                    fallback_policy=fallback_policy,
                    deadline_ms=req.deadline_ms,
                    workload_category=req.workload_category,
                    selected_candidate="REJECTED",
                    selected_miou=0.0,
                    configured_latency_bound_ms=0.0,
                    controller_latency_ms=round(ctrl_ms, 4),
                    preprocessing_latency_ms=0.0,
                    transfer_latency_ms=0.0,
                    model_execution_ms=0.0,
                    model_cuda_event_ms=0.0,
                    postprocessing_latency_ms=0.0,
                    total_request_latency_ms=round(e2e_ms, 4),
                    is_infeasible=True,
                    is_fallback=True,
                    is_rejected=True,
                    quality_compliant=False,
                    actual_deadline_hit=False,
                    reason=reason,
                )
            )
            continue

        # Execute request
        t_prep_0 = time.perf_counter()
        img_tensor = preprocess_image(img_path)
        t_prep_1 = time.perf_counter()
        prep_ms = (t_prep_1 - t_prep_0) * 1000.0

        t_trans_0 = time.perf_counter()
        img_cuda = img_tensor.to(device, non_blocking=False)
        torch.cuda.synchronize(device)
        t_trans_1 = time.perf_counter()
        trans_ms = (t_trans_1 - t_trans_0) * 1000.0

        bb, hd = models[sel_cand]
        preds_cpu, model_host_ms, model_cuda_ms = execute_model_forward(bb, hd, img_cuda)

        t_post_0 = time.perf_counter()
        _ = preds_cpu.byte().numpy()
        t_post_1 = time.perf_counter()
        post_ms = (t_post_1 - t_post_0) * 1000.0

        t_req_1 = time.perf_counter()
        e2e_ms = (t_req_1 - t_req_0) * 1000.0

        actual_hit = e2e_ms <= req.deadline_ms
        qual_comp = sel_miou >= tau_miou

        results.append(
            RuntimeAccuracyResult(
                request_id=req.request_id,
                policy=policy_name,
                tau_miou=tau_miou,
                fallback_policy=fallback_policy,
                deadline_ms=req.deadline_ms,
                workload_category=req.workload_category,
                selected_candidate=sel_cand,
                selected_miou=round(sel_miou, 4),
                configured_latency_bound_ms=round(bound_ms, 4),
                controller_latency_ms=round(ctrl_ms, 4),
                preprocessing_latency_ms=round(prep_ms, 4),
                transfer_latency_ms=round(trans_ms, 4),
                model_execution_ms=round(model_host_ms, 4),
                model_cuda_event_ms=round(model_cuda_ms, 4),
                postprocessing_latency_ms=round(post_ms, 4),
                total_request_latency_ms=round(e2e_ms, 4),
                is_infeasible=infeasible,
                is_fallback=fallback,
                is_rejected=rejected,
                quality_compliant=qual_comp,
                actual_deadline_hit=actual_hit,
                reason=reason,
            )
        )

    return results


def summarize_runtime_results(results: list[RuntimeAccuracyResult]) -> dict[str, Any]:
    total = len(results)
    hits = sum(1 for r in results if r.actual_deadline_hit)
    compliant = sum(1 for r in results if r.quality_compliant)
    rejected = sum(1 for r in results if r.is_rejected)
    fallbacks = sum(1 for r in results if r.is_fallback and not r.is_rejected)

    executed = [r for r in results if not r.is_rejected]
    e2e_times = [r.total_request_latency_ms for r in executed] if executed else [0.0]
    model_times = [r.model_execution_ms for r in executed] if executed else [0.0]
    ctrl_times = [r.controller_latency_ms for r in results]
    mious = [r.selected_miou for r in executed] if executed else [0.0]

    cand_counts: dict[str, int] = {}
    for r in results:
        cand_counts[r.selected_candidate] = cand_counts.get(r.selected_candidate, 0) + 1

    return {
        "total_requests": total,
        "actual_deadline_hits": hits,
        "actual_deadline_hit_rate_pct": round(hits / total * 100.0, 2),
        "quality_compliant_count": compliant,
        "quality_compliance_rate_pct": round(compliant / total * 100.0, 2),
        "rejected_count": rejected,
        "fallback_count": fallbacks,
        "mean_selected_miou": round(float(np.mean(mious)), 4),
        "candidate_counts": cand_counts,
        "latency_stats": {
            "e2e_ms": compute_summary_stats(e2e_times),
            "model_ms": compute_summary_stats(model_times),
            "controller_ms": compute_summary_stats(ctrl_times),
        },
    }


def main() -> None:
    print("=" * 70)
    print("Phase 10C: Runtime Accuracy-Constrained Prototype Evaluation")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    candidates = load_accuracy_candidates(BENCHMARK_PATH)
    models = load_candidate_models(device)
    print("Candidate models preloaded on GPU.")

    images = sorted(PET_DATASET_DIR.glob("*.jpg"))[:32]
    stream = generate_deadline_stream(100, seed=42, workload_name="default_mixture")
    print(f"Loaded {len(images)} images, generated stream of {len(stream)} requests.")

    configurations = [
        ("always_full_attention", 0.0, "highest_quality_miss"),
        ("always_fastest", 0.0, "fastest_miss"),
        ("deadline_only_adaptive", 0.0, "fastest_miss"),
        ("accuracy_constrained", 0.65, "highest_quality_miss"),
        ("accuracy_constrained", 0.70, "highest_quality_miss"),
        ("accuracy_constrained", 0.70, "reject"),
    ]

    all_traces: list[RuntimeAccuracyResult] = []
    summaries: dict[str, Any] = {}

    for pol, tau, fb in configurations:
        key = f"{pol}_tau_{tau:.2f}_{fb}" if pol == "accuracy_constrained" else pol
        print(f"\n--- Running: {key} ---")
        t0 = time.time()
        res = execute_runtime_policy(
            policy_name=pol,
            stream=stream,
            models=models,
            candidates=candidates,
            device=device,
            image_paths=images,
            tau_miou=tau,
            fallback_policy=fb,
        )
        t1 = time.time()
        print(f"Finished in {t1 - t0:.2f}s.")
        all_traces.extend(res)
        summaries[key] = summarize_runtime_results(res)

    # Save outputs
    summary_path = OUTPUT_DIR / "runtime_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved runtime summary to: {summary_path}")

    trace_path = OUTPUT_DIR / "runtime_trace.csv"
    with open(trace_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in RuntimeAccuracyResult.__dataclass_fields__.values()])
        writer.writeheader()
        for r in all_traces:
            writer.writerow(asdict(r))
    print(f"Saved runtime trace to: {trace_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("RUNTIME PROTOTYPE SUMMARY (RTX 4060 Laptop GPU):")
    print(f"{'Configuration':<38} | {'Hit Rate':<10} | {'Qual Comp':<10} | {'Avg mIoU':<8} | {'Median e2e'}")
    print("-" * 70)
    for key, summ in summaries.items():
        hit = f"{summ['actual_deadline_hit_rate_pct']:.1f}%"
        qc = f"{summ['quality_compliance_rate_pct']:.1f}%"
        miou = f"{summ['mean_selected_miou']:.4f}"
        med = f"{summ['latency_stats']['e2e_ms']['median']:.2f} ms"
        print(f"{key:<38} | {hit:<10} | {qc:<10} | {miou:<8} | {med}")
    print("=" * 70)


if __name__ == "__main__":
    main()
