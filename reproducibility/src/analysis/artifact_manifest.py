"""Artifact manifest module documenting end-to-end experimental provenance."""

from __future__ import annotations

import json
import os
from typing import Dict, Any, List

ARTIFACT_MANIFEST: Dict[str, Dict[str, Any]] = {
    "Table_I": {
        "title": "Cross-Keep-Rate Perception Performance Comparison",
        "manuscript_table": "Table I",
        "generating_script": "reproducibility/scripts/07_run_statistics.py",
        "source_data": "outputs/cross_keep_rate_study/cross_keep_rate_results.csv",
        "output_tex": "reproducibility/artifacts/tables/table1_cross_keep_rate.tex",
        "metrics": ["Top-1 Accuracy", "Macro-F1", "Weighted-F1", "Per-Class F1", "Bootstrap CIs"],
        "num_models": 11,
    },
    "Table_II": {
        "title": "Multi-Seed Reproducibility Validation (Seeds 42, 123, 456)",
        "manuscript_table": "Table II",
        "generating_script": "reproducibility/scripts/07_run_statistics.py",
        "source_data": [
            "outputs/cross_keep_rate_study/multi_seed_validation/per_seed_results.csv",
            "outputs/cross_keep_rate_study/multi_seed_validation/aggregate_results.csv",
        ],
        "output_tex": "reproducibility/artifacts/tables/table2_multi_seed.tex",
        "metrics": ["Per-Seed Top-1", "Per-Seed Macro-F1", "Paired Delta (CMPR - CB)"],
        "num_seeds": 3,
    },
    "Table_III": {
        "title": "NVIDIA RTX 4060 GPU Latency and Throughput Profiles",
        "manuscript_table": "Table III",
        "generating_script": "reproducibility/scripts/08_benchmark_models.py",
        "source_data": "outputs/bdd100k_migration/full_dataset_latency_profiles.csv",
        "output_tex": "reproducibility/artifacts/tables/table3_hardware_profiles.tex",
        "metrics": ["Eager Latency (Mean, p95)", "CUDA Graph Latency (Mean, p95)", "FPS Throughput"],
    },
    "Table_V": {
        "title": "Real-Time Scheduling Simulation Across 10 Workloads and 7 Policies",
        "manuscript_table": "Table V",
        "generating_script": "reproducibility/scripts/09_run_scheduler_simulation.py",
        "source_data": "outputs/bdd100k_migration/simulation_results.csv",
        "output_csv": "reproducibility/artifacts/tables/table5_simulation_results.csv",
        "metrics": ["On-Time Ratio", "Response Time (Mean, p95)", "GPU Utilization", "Admitted/Expired/Rejected"],
        "num_configurations": 70,
        "conservation_laws_satisfied": 4,
    },
    "Table_VI": {
        "title": "Physical GPU Trace Replay vs Discrete-Event Simulation Validation",
        "manuscript_table": "Table VI",
        "generating_script": "reproducibility/scripts/10_run_hardware_replay.py",
        "source_data": "outputs/bdd100k_migration/corrected_replay/hardware_replay_comparison.csv",
        "output_tex": "reproducibility/artifacts/tables/table6_hardware_replay.tex",
        "metrics": ["Simulation On-Time %", "GPU Replay On-Time %", "Makespan Delta %", "Empirical Test Acc %"],
        "num_configurations": 9,
    },
    "Figure_2": {
        "title": "Perception-Efficiency Pareto Frontier",
        "manuscript_figure": "Figure 2",
        "generating_script": "reproducibility/scripts/11_generate_artifacts.py",
        "source_data": "outputs/cross_keep_rate_study/cross_keep_rate_results.csv",
        "output_image": "reproducibility/artifacts/figures/fig2_perception_tradeoff.png",
    },
    "Figure_3": {
        "title": "GPU Execution Latency Distributions",
        "manuscript_figure": "Figure 3",
        "generating_script": "reproducibility/scripts/11_generate_artifacts.py",
        "source_data": "outputs/bdd100k_migration/full_dataset_latency_profiles.csv",
        "output_image": "reproducibility/artifacts/figures/fig3_gpu_latency_profiles.png",
    },
    "Figure_4": {
        "title": "Scheduling On-Time Completion Comparison across 10 Workloads",
        "manuscript_figure": "Figure 4",
        "generating_script": "reproducibility/scripts/11_generate_artifacts.py",
        "source_data": "outputs/bdd100k_migration/simulation_results.csv",
        "output_image": "reproducibility/artifacts/figures/fig4_scheduling_ontime_comparison.png",
    },
    "Figure_5": {
        "title": "Request Terminal Outcomes under Heavy Overload Stress",
        "manuscript_figure": "Figure 5",
        "generating_script": "reproducibility/scripts/11_generate_artifacts.py",
        "source_data": "outputs/bdd100k_migration/simulation_results.csv",
        "output_image": "reproducibility/artifacts/figures/fig5_request_outcomes_stacked.png",
    },
    "Figure_6": {
        "title": "Simulation vs Physical GPU Hardware Trace Replay Validation",
        "manuscript_figure": "Figure 6",
        "generating_script": "reproducibility/scripts/11_generate_artifacts.py",
        "source_data": "outputs/bdd100k_migration/corrected_replay/hardware_replay_comparison.csv",
        "output_image": "reproducibility/artifacts/figures/fig6_hardware_replay_validation.png",
    },
}


def export_artifact_manifest(output_json: str) -> None:
    """Save artifact manifest to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(ARTIFACT_MANIFEST, f, indent=2)
