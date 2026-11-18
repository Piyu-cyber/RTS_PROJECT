"""Analysis, table generation, and figure generation package."""

from .aggregate_results import aggregate_multiseed_records
from .generate_tables import (
    generate_table1_latex,
    generate_table2_latex,
    generate_table3_latex,
    generate_table6_latex,
)
from .generate_figures import (
    plot_figure2_pareto,
    plot_figure3_latency,
    plot_figure4_scheduling_ontime,
    plot_figure5_stacked_outcomes,
    plot_figure6_hardware_replay,
)
from .artifact_manifest import ARTIFACT_MANIFEST, export_artifact_manifest

__all__ = [
    "aggregate_multiseed_records",
    "generate_table1_latex",
    "generate_table2_latex",
    "generate_table3_latex",
    "generate_table6_latex",
    "plot_figure2_pareto",
    "plot_figure3_latency",
    "plot_figure4_scheduling_ontime",
    "plot_figure5_stacked_outcomes",
    "plot_figure6_hardware_replay",
    "ARTIFACT_MANIFEST",
    "export_artifact_manifest",
]
