"""Admission control and model profiling for real-time inference."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .request import Request, RequestOutcome


@dataclass
class ModelProfile:
    """Represents a validated DeiT or EViT model configuration profile."""

    config_id: str
    display_name: str
    keep_rate: float
    fuse_token: bool
    parameters_m: float
    macs_g: float
    token_sequence: List[int]
    val_top1_accuracy: float
    test_top1_accuracy: float
    test_macro_f1: float
    cuda_graph_latency_ms: Dict[str, float]
    eager_latency_ms: Dict[str, float]
    throughput_fps: float
    checkpoint_path: str = ""
    official_checkpoint: str = ""

    def get_latency_ms(self, metric: str = "p95", backend: str = "cuda_graph") -> float:
        """Return the estimated latency for the specified metric and execution backend."""
        table = self.cuda_graph_latency_ms if backend == "cuda_graph" else self.eager_latency_ms
        if metric in table:
            return float(table[metric])
        raise ValueError(f"Unknown latency metric '{metric}'. Available: {list(table.keys())}")


def load_model_profiles(json_path: str) -> Dict[str, ModelProfile]:
    """Load model profiles from a validated JSON file."""
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Model profiles JSON not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profiles = {}
    items = data.get("profiles", data) if isinstance(data, dict) else data
    for cfg_id, p in items.items():
        if isinstance(p, dict):
            profiles[cfg_id] = ModelProfile(
                config_id=p.get("config_id", cfg_id),
                display_name=p.get("display_name", cfg_id),
                keep_rate=p.get("keep_rate", 1.0),
                fuse_token=p.get("fuse_token", True),
                parameters_m=p.get("parameters_m", 22.0),
                macs_g=p.get("macs_g", 4.6),
                token_sequence=p.get("token_sequence", [197]),
                val_top1_accuracy=p.get("val_top1_accuracy", 78.0),
                test_top1_accuracy=p.get("test_top1_accuracy", 77.5),
                test_macro_f1=p.get("test_macro_f1", 54.5),
                cuda_graph_latency_ms=p.get("cuda_graph_latency_ms", {"p95": 3.68}),
                eager_latency_ms=p.get("eager_latency_ms", {"p95": 4.5}),
                throughput_fps=p.get("throughput_fps", 300.0),
                checkpoint_path=p.get("checkpoint_path", ""),
                official_checkpoint=p.get("official_checkpoint", ""),
            )
    return profiles


class AdmissionController:
    """Queue-aware admission controller verifying feasibility before queue admission."""

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
            request: The candidate arriving request.
            current_time_ms: Current clock time.
            gpu_busy_until_ms: Timestamp when ongoing GPU execution finishes.
            queued_requests: Currently admitted requests waiting in queue.
            fixed_model_id: If evaluating a fixed-model policy, specifies the fixed model.
        """
        start_time = max(current_time_ms, gpu_busy_until_ms)

        accumulated_queue_work = 0.0
        for q in queued_requests:
            if q.selected_model and q.selected_model in self.profiles:
                work = self.profiles[q.selected_model].get_latency_ms(self.latency_metric, self.backend)
            elif fixed_model_id and fixed_model_id in self.profiles:
                work = self.profiles[fixed_model_id].get_latency_ms(self.latency_metric, self.backend)
            else:
                work = self.min_exec_time_ms
            accumulated_queue_work += work

        if fixed_model_id and fixed_model_id in self.profiles:
            candidate_work = self.profiles[fixed_model_id].get_latency_ms(self.latency_metric, self.backend)
        else:
            candidate_work = self.min_exec_time_ms

        expected_completion_ms = start_time + accumulated_queue_work + candidate_work
        return expected_completion_ms <= request.absolute_deadline_ms + 1e-6
