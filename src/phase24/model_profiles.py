"""Model profiles and latency distributions for Phase 24 real-time scheduling."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

DEFAULT_PROFILES_JSON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "bdd100k_migration", "model_profiles.json")
)


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
    checkpoint_path: str
    official_checkpoint: str

    def get_latency_ms(self, metric: str = "p95", backend: str = "cuda_graph") -> float:
        """Return the estimated latency for the specified metric and execution backend.

        Args:
            metric: One of 'mean', 'median', 'p95', 'p99', 'min', 'max'.
            backend: One of 'cuda_graph' (default) or 'eager'.
        """
        table = self.cuda_graph_latency_ms if backend == "cuda_graph" else self.eager_latency_ms
        if metric in table:
            return float(table[metric])
        raise ValueError(f"Unknown latency metric '{metric}'. Available: {list(table.keys())}")


def load_model_profiles(json_path: Optional[str] = None) -> Dict[str, ModelProfile]:
    """Load model profiles from the Phase 24 JSON file."""
    path = json_path or DEFAULT_PROFILES_JSON
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model profiles JSON not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profiles = {}
    for cfg_id, p in data.get("profiles", {}).items():
        profiles[cfg_id] = ModelProfile(
            config_id=p["config_id"],
            display_name=p["display_name"],
            keep_rate=float(p["keep_rate"]),
            fuse_token=bool(p["fuse_token"]),
            parameters_m=float(p["parameters_m"]),
            macs_g=float(p["macs_g"]),
            token_sequence=list(p["token_sequence"]),
            val_top1_accuracy=float(p["val_top1_accuracy"]),
            test_top1_accuracy=float(p["test_top1_accuracy"]),
            test_macro_f1=float(p["test_macro_f1"]),
            cuda_graph_latency_ms=p["cuda_graph_latency_ms"],
            eager_latency_ms=p["eager_latency_ms"],
            throughput_fps=float(p["throughput_fps"]),
            checkpoint_path=p.get("checkpoint_path", ""),
            official_checkpoint=p.get("official_checkpoint", ""),
        )

    return profiles
