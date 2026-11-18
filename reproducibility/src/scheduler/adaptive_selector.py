"""Adaptive model selection based on remaining deadline slack and validation quality."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List
from .request import Request
from .admission_control import ModelProfile


class AdaptiveController:
    """Slack-aware model selector enforcing validation accuracy eligibility."""

    def __init__(
        self,
        profiles: Dict[str, ModelProfile],
        latency_metric: str = "p95",
        backend: str = "cuda_graph",
        quality_threshold: float = 45.0,
        firm_deadlines: bool = True,
    ):
        self.profiles = profiles
        self.latency_metric = latency_metric
        self.backend = backend
        self.quality_threshold = quality_threshold
        self.firm_deadlines = firm_deadlines

        # Filter eligible models by validation accuracy threshold
        self.eligible_models: List[str] = [
            cfg_id for cfg_id, p in self.profiles.items()
            if p.val_top1_accuracy >= self.quality_threshold
        ]
        if not self.eligible_models:
            raise ValueError(f"No models meet quality threshold >= {self.quality_threshold}%")

        # Sort eligible models: primary by val_accuracy DESC, secondary by latency ASC
        self.eligible_models.sort(
            key=lambda m: (
                self.profiles[m].val_top1_accuracy,
                -self.profiles[m].get_latency_ms(self.latency_metric, self.backend),
            ),
            reverse=True,
        )

        # Fastest eligible model
        self.fastest_model_id = min(
            self.eligible_models,
            key=lambda m: self.profiles[m].get_latency_ms(self.latency_metric, self.backend),
        )

    def select_model(
        self,
        request: Request,
        current_time_ms: float,
    ) -> Tuple[Optional[str], float, float]:
        """Select the highest-quality eligible model predicted to finish on time.

        Args:
            request: The request ready for dispatch.
            current_time_ms: Current simulation clock time.

        Returns:
            Tuple of (selected_model_id or None, estimated_exec_ms, remaining_slack_ms).
        """
        best_candidate: Optional[str] = None
        best_val_acc: float = -1.0
        best_slack: float = -float("inf")
        best_exec_time: float = 0.0

        for model_id in self.eligible_models:
            profile = self.profiles[model_id]
            exec_time = profile.get_latency_ms(self.latency_metric, self.backend)
            predicted_completion = current_time_ms + exec_time
            slack = request.absolute_deadline_ms - predicted_completion

            if slack >= -1e-6:
                if profile.val_top1_accuracy > best_val_acc or (
                    profile.val_top1_accuracy == best_val_acc and slack > best_slack
                ):
                    best_candidate = model_id
                    best_val_acc = profile.val_top1_accuracy
                    best_slack = slack
                    best_exec_time = exec_time

        if best_candidate is not None:
            return best_candidate, best_exec_time, best_slack

        fastest_profile = self.profiles[self.fastest_model_id]
        fastest_exec = fastest_profile.get_latency_ms(self.latency_metric, self.backend)
        fastest_slack = request.absolute_deadline_ms - (current_time_ms + fastest_exec)

        if self.firm_deadlines:
            return None, 0.0, fastest_slack
        else:
            return self.fastest_model_id, fastest_exec, fastest_slack
