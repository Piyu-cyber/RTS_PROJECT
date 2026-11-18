"""Script 11: Master artifact generator recreating all manuscript tables and figures."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reproducibility.src.analysis.generate_tables import (
    generate_table1_latex,
    generate_table2_latex,
    generate_table3_latex,
    generate_table6_latex,
)
from reproducibility.src.analysis.generate_figures import (
    plot_figure2_pareto,
    plot_figure3_latency,
    plot_figure4_scheduling_ontime,
    plot_figure5_stacked_outcomes,
    plot_figure6_hardware_replay,
)
from reproducibility.src.analysis.artifact_manifest import export_artifact_manifest


def main():
    parser = argparse.ArgumentParser(description="Generate all manuscript tables and figures.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "reproducibility", "artifacts"),
        help="Root directory for generated artifacts.",
    )
    args = parser.parse_args()

    tables_dir = os.path.join(args.output_dir, "tables")
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 80)
    print("STEP 11: GENERATING MANUSCRIPT ARTIFACTS (TABLES I-VI & FIGURES 2-6)")
    print(f"Artifacts output directory: {args.output_dir}")
    print("=" * 80)

    # 1. Table I (Cross-Keep-Rate Comparison)
    t1_src = os.path.join(PROJECT_ROOT, "outputs", "cross_keep_rate_study", "cross_keep_rate_results.csv")
    t1_tex = os.path.join(tables_dir, "table1_cross_keep_rate.tex")
    generate_table1_latex(t1_src, t1_tex)
    print(f"[GENERATED] Table I LaTeX -> {t1_tex}")

    # 2. Table II (Multi-Seed Validation)
    t2_per_seed = os.path.join(PROJECT_ROOT, "outputs", "cross_keep_rate_study", "multi_seed_validation", "per_seed_results.csv")
    t2_agg = os.path.join(PROJECT_ROOT, "outputs", "cross_keep_rate_study", "multi_seed_validation", "aggregate_results.csv")
    t2_tex = os.path.join(tables_dir, "table2_multi_seed.tex")
    generate_table2_latex(t2_per_seed, t2_agg, t2_tex)
    print(f"[GENERATED] Table II LaTeX -> {t2_tex}")

    # 3. Table III (Hardware Profiles)
    t3_src = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration", "full_dataset_latency_profiles.csv")
    t3_tex = os.path.join(tables_dir, "table3_hardware_profiles.tex")
    generate_table3_latex(t3_src, t3_tex)
    print(f"[GENERATED] Table III LaTeX -> {t3_tex}")

    # 4. Table VI (Hardware Replay Comparison)
    t6_src = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration", "corrected_replay", "hardware_replay_comparison.csv")
    t6_tex = os.path.join(tables_dir, "table6_hardware_replay.tex")
    generate_table6_latex(t6_src, t6_tex)
    print(f"[GENERATED] Table VI LaTeX -> {t6_tex}")

    # 5. Figure 2 (Perception Pareto Front)
    fig2_png = os.path.join(figures_dir, "fig2_perception_tradeoff.png")
    plot_figure2_pareto(t1_src, fig2_png)
    print(f"[GENERATED] Figure 2 PNG -> {fig2_png}")

    # 6. Figure 3 (GPU Latency Profiles)
    fig3_png = os.path.join(figures_dir, "fig3_gpu_latency_profiles.png")
    plot_figure3_latency(t3_src, fig3_png)
    print(f"[GENERATED] Figure 3 PNG -> {fig3_png}")

    # 7. Figure 4 (Scheduling On-Time Comparison)
    t5_src = os.path.join(PROJECT_ROOT, "outputs", "bdd100k_migration", "simulation_results.csv")
    fig4_png = os.path.join(figures_dir, "fig4_scheduling_ontime_comparison.png")
    plot_figure4_scheduling_ontime(t5_src, fig4_png)
    print(f"[GENERATED] Figure 4 PNG -> {fig4_png}")

    # 8. Figure 5 (Stacked Request Outcomes)
    fig5_png = os.path.join(figures_dir, "fig5_request_outcomes_stacked.png")
    plot_figure5_stacked_outcomes(t5_src, fig5_png)
    print(f"[GENERATED] Figure 5 PNG -> {fig5_png}")

    # 9. Figure 6 (Hardware Replay Validation)
    fig6_png = os.path.join(figures_dir, "fig6_hardware_replay_validation.png")
    plot_figure6_hardware_replay(t6_src, fig6_png)
    print(f"[GENERATED] Figure 6 PNG -> {fig6_png}")

    # 10. Export Artifact Manifest
    manifest_json = os.path.join(args.output_dir, "artifact_manifest.json")
    export_artifact_manifest(manifest_json)
    print(f"[GENERATED] Artifact Manifest JSON -> {manifest_json}")

    print("\nAll manuscript tables, figures, and manifests generated successfully (exit code 0).\n")


if __name__ == "__main__":
    main()
