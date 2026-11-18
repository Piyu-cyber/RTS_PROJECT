"""Phase 24B Multi-Seed Robustness Suite Generator.

Generates statistically independent workload traces across periodic, jittered,
bursty, and overload conditions with explicit random seeds and pre-declared parameters.
"""

from __future__ import annotations

import random
from typing import List, Dict, Any, Optional
from src.phase24.request import Request
from src.phase24.workload_generator import DEADLINE_RANGES, WorkloadConfig


LOAD_LEVEL_TARGETS = {
    "low": 0.50,         # Lambda = 0.50 / C_nom -> generous inter-arrival
    "moderate": 0.85,    # Lambda = 0.85 / C_nom -> standard high throughput
    "high": 1.15,        # Lambda = 1.15 / C_nom -> exceeds capacity under baseline
    "overload": 1.50,    # Lambda = 1.50 / C_nom -> severe saturation
}


def generate_trace_exact(
    config: WorkloadConfig,
    bdd_test_samples: Optional[List[Dict[str, Any]]] = None,
) -> List[Request]:
    """Generate a reproducible sequence of Request objects with pre-declared seed."""
    rng = random.Random(config.seed)
    d_min, d_max = DEADLINE_RANGES[config.deadline_tightness]

    target_load = LOAD_LEVEL_TARGETS.get(config.load_level, 0.85)
    base_inter_arrival = config.nominal_service_time_ms / target_load

    num_bdd_samples = len(bdd_test_samples) if bdd_test_samples else 0

    requests: List[Request] = []
    current_time = 0.0

    for i in range(config.num_requests):
        req_id = f"{config.name}_s{config.seed}_req_{i:04d}"

        if config.workload_type == "periodic":
            if i > 0:
                current_time += base_inter_arrival

        elif config.workload_type == "jittered":
            if i > 0:
                jitter = rng.uniform(-0.4 * base_inter_arrival, 0.4 * base_inter_arrival)
                current_time = max(current_time + 0.1, current_time + base_inter_arrival + jitter)

        elif config.workload_type == "bursty":
            if i > 0:
                is_burst = rng.random() < 0.25
                if is_burst:
                    delta = rng.uniform(0.4, 1.0)  # Very tight cluster
                else:
                    delta = base_inter_arrival * rng.uniform(1.2, 1.8)
                current_time += delta

        elif config.workload_type == "overload":
            if i > 0:
                delta = rng.uniform(0.5 * base_inter_arrival, 0.9 * base_inter_arrival)
                current_time += delta

        else:
            if i > 0:
                current_time += base_inter_arrival

        rel_deadline = rng.uniform(d_min, d_max)

        # Map to actual held-out test sample if available
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


def get_canonical_suite(bdd_test_samples: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Tuple[WorkloadConfig, List[Request]]]:
    """Returns the canonical 10 workloads from Phase 24 (seeds 42-51)."""
    configs = [
        WorkloadConfig("periodic_low_loose", "periodic", "low", "loose", seed=42),
        WorkloadConfig("periodic_mod_mod", "periodic", "moderate", "moderate", seed=43),
        WorkloadConfig("periodic_high_tight", "periodic", "high", "tight", seed=44),
        WorkloadConfig("jittered_mod_mod", "jittered", "moderate", "moderate", seed=45),
        WorkloadConfig("jittered_high_tight", "jittered", "high", "tight", seed=46),
        WorkloadConfig("bursty_mod_mod", "bursty", "moderate", "moderate", seed=47),
        WorkloadConfig("bursty_high_tight", "bursty", "high", "tight", seed=48),
        WorkloadConfig("overload_mod_deadline", "overload", "overload", "moderate", seed=49),
        WorkloadConfig("overload_tight_deadline", "overload", "overload", "tight", seed=50),
        WorkloadConfig("periodic_low_tight", "periodic", "low", "tight", seed=51),
    ]
    suite = {}
    for cfg in configs:
        suite[cfg.name] = (cfg, generate_trace_exact(cfg, bdd_test_samples=bdd_test_samples))
    return suite


def get_robustness_suite(
    seeds: List[int] = [101, 102, 103, 104, 105],
    bdd_test_samples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Tuple[WorkloadConfig, List[Request]]]:
    """Build multi-seed evaluation suite covering 6 representative regimes x N seeds."""
    conditions = [
        ("periodic_low_loose", "periodic", "low", "loose"),
        ("periodic_mod_mod", "periodic", "moderate", "moderate"),
        ("periodic_high_tight", "periodic", "high", "tight"),
        ("jittered_high_tight", "jittered", "high", "tight"),
        ("bursty_high_tight", "bursty", "high", "tight"),
        ("overload_tight_deadline", "overload", "overload", "tight"),
    ]

    suite = {}
    for base_name, w_type, load, tight in conditions:
        for s in seeds:
            cfg = WorkloadConfig(f"{base_name}_s{s}", w_type, load, tight, seed=s)
            suite[cfg.name] = (cfg, generate_trace_exact(cfg, bdd_test_samples=bdd_test_samples))
    return suite
