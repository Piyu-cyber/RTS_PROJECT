"""Controlled workload trace generator for real-time inference evaluation."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .request import Request


@dataclass
class WorkloadConfig:
    """Configuration for synthetic workload generation."""
    name: str
    workload_type: str               # periodic, jittered, bursty, overload
    load_level: str                  # low, moderate, high, overload
    deadline_tightness: str          # loose, moderate, tight
    num_requests: int = 100
    nominal_service_time_ms: float = 3.28   # DeiT Baseline CUDA Graph mean
    seed: int = 42


DEADLINE_RANGES = {
    "tight": (3.5, 6.0),       # Just enough for EViT (2.36ms) or single DeiT (3.28ms)
    "moderate": (6.0, 12.0),   # Room for 2-3 queued inferences
    "loose": (15.0, 25.0),     # Ample slack for 5+ inferences
}


def generate_trace(config: WorkloadConfig, bdd_test_samples: Optional[List[Dict[str, Any]]] = None) -> List[Request]:
    """Generate a reproducible sequence of Request objects."""
    rng = random.Random(config.seed)
    d_min, d_max = DEADLINE_RANGES[config.deadline_tightness]

    requests: List[Request] = []
    current_time = 0.0

    # Determine base inter-arrival time according to load level
    # load = service_time / inter_arrival -> inter_arrival = service_time / load
    if config.load_level == "low":
        target_load = 0.50
    elif config.load_level == "moderate":
        target_load = 0.85
    elif config.load_level == "high":
        target_load = 1.15
    elif config.load_level == "overload":
        target_load = 1.50
    else:
        target_load = 0.85

    base_inter_arrival = config.nominal_service_time_ms / target_load

    num_bdd_samples = len(bdd_test_samples) if bdd_test_samples else 0

    for i in range(config.num_requests):
        req_id = f"{config.name}_req_{i:04d}"

        if config.workload_type == "periodic":
            if i > 0:
                current_time += base_inter_arrival

        elif config.workload_type == "jittered":
            if i > 0:
                jitter = rng.uniform(-0.4 * base_inter_arrival, 0.4 * base_inter_arrival)
                current_time = max(current_time + 0.1, current_time + base_inter_arrival + jitter)

        elif config.workload_type == "bursty":
            # 20% chance of entering a burst of 3-6 tight requests
            if i > 0:
                is_burst = rng.random() < 0.25
                if is_burst:
                    delta = rng.uniform(0.4, 1.0)  # Very tight arrival
                else:
                    delta = base_inter_arrival * rng.uniform(1.2, 1.8)
                current_time += delta

        elif config.workload_type == "overload":
            # Sustained high arrival rate
            if i > 0:
                delta = rng.uniform(0.5 * base_inter_arrival, 0.9 * base_inter_arrival)
                current_time += delta

        else:
            if i > 0:
                current_time += base_inter_arrival

        rel_deadline = rng.uniform(d_min, d_max)

        # Attach real image metadata if available
        image_idx = i % num_bdd_samples if num_bdd_samples > 0 else None
        image_name = bdd_test_samples[image_idx]["image_name"] if image_idx is not None else None
        gt_label = bdd_test_samples[image_idx]["class_id"] if image_idx is not None else None

        req = Request(
            request_id=req_id,
            arrival_time_ms=round(current_time, 4),
            relative_deadline_ms=round(rel_deadline, 4),
            image_idx=image_idx,
            image_name=image_name,
            ground_truth_label=gt_label,
        )
        requests.append(req)

    return requests


def get_standard_workload_suite(bdd_test_samples: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[Request]]:
    """Build the standard experimental suite of 10 workloads covering all required patterns."""
    configs = [
        # 1. Periodic Baselines (varying load and tightness)
        WorkloadConfig("periodic_low_loose", "periodic", "low", "loose", seed=42),
        WorkloadConfig("periodic_mod_mod", "periodic", "moderate", "moderate", seed=43),
        WorkloadConfig("periodic_high_tight", "periodic", "high", "tight", seed=44),

        # 2. Jittered Traces
        WorkloadConfig("jittered_mod_mod", "jittered", "moderate", "moderate", seed=45),
        WorkloadConfig("jittered_high_tight", "jittered", "high", "tight", seed=46),

        # 3. Bursty Traces
        WorkloadConfig("bursty_mod_mod", "bursty", "moderate", "moderate", seed=47),
        WorkloadConfig("bursty_high_tight", "bursty", "high", "tight", seed=48),

        # 4. Overload Traces (Stress Tests)
        WorkloadConfig("overload_mod_deadline", "overload", "overload", "moderate", seed=49),
        WorkloadConfig("overload_tight_deadline", "overload", "overload", "tight", seed=50),

        # 5. Low load tight deadline
        WorkloadConfig("periodic_low_tight", "periodic", "low", "tight", seed=51),
    ]

    suite = {}
    for cfg in configs:
        suite[cfg.name] = generate_trace(cfg, bdd_test_samples=bdd_test_samples)

    return suite


def save_workload_definitions(suite: Dict[str, List[Request]], out_path: str):
    """Serialize workload definitions to JSON."""
    payload = {}
    for name, req_list in suite.items():
        payload[name] = [
            {
                "request_id": r.request_id,
                "arrival_time_ms": r.arrival_time_ms,
                "relative_deadline_ms": r.relative_deadline_ms,
                "absolute_deadline_ms": r.absolute_deadline_ms,
                "image_idx": r.image_idx,
                "image_name": r.image_name,
                "ground_truth_label": r.ground_truth_label,
            }
            for r in req_list
        ]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
