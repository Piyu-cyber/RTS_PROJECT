"""Admission control and feasibility assessment for real-time requests."""

from __future__ import annotations

from typing import List, Dict, Optional
from .request import Request, RequestOutcome
from .model_profiles import ModelProfile


class AdmissionController:
    """Queue-aware admission controller verifying feasibility before admission."""

    def __init__(
        self,
        profiles: Dict[str, ModelProfile],
        latency_metric: str = "p95",
        backend: str = "cuda_graph",
        quality_threshold: float = 45.0,
    ):
        self.profiles = profiles
        self.latency_metric = latency_metric
        self.backend = backend
        self.quality_threshold = quality_threshold

        # Fast lookup of eligible models
        self.eligible_profiles = {
            k: p for k, p in self.profiles.items() if p.val_top1_accuracy >= self.quality_threshold
        }
        if not self.eligible_profiles:
            raise ValueError(f"No model profiles satisfy quality threshold >= {self.quality_threshold}%")

        self.min_exec_time_ms = min(
            p.get_latency_ms(self.latency_metric, self.backend) for p in self.eligible_profiles.values()
        )

    def is_feasible(
        self,
        request: Request,
        current_time_ms: float,
        gpu_busy_until_ms: float,
        queued_requests: List[Request],
        fixed_model_id: Optional[str] = None,
    ) -> bool:
        """Check if request can complete before its absolute deadline given queue state.

        Args:
            request: The candidate request arriving.
            current_time_ms: Current clock time.
            gpu_busy_until_ms: Timestamp when ongoing GPU execution finishes.
            queued_requests: Currently admitted requests waiting in queue.
            fixed_model_id: If evaluating a fixed-model policy, specifies the fixed model.
        """
        start_time = max(current_time_ms, gpu_busy_until_ms)

        # Sum expected execution time of all requests currently queued ahead
        accumulated_queue_work = 0.0
        for q in queued_requests:
            if q.selected_model and q.selected_model in self.profiles:
                work = self.profiles[q.selected_model].get_latency_ms(self.latency_metric, self.backend)
            elif fixed_model_id and fixed_model_id in self.profiles:
                work = self.profiles[fixed_model_id].get_latency_ms(self.latency_metric, self.backend)
            else:
                work = self.min_exec_time_ms
            accumulated_queue_work += work

        # Candidate execution time
        if fixed_model_id and fixed_model_id in self.profiles:
            candidate_work = self.profiles[fixed_model_id].get_latency_ms(self.latency_metric, self.backend)
        else:
            # Under adaptive selection, check if at least the fastest eligible model can finish on time
            candidate_work = self.min_exec_time_ms

        predicted_finish = start_time + accumulated_queue_work + candidate_work
        return predicted_finish <= request.absolute_deadline_ms + 1e-6
