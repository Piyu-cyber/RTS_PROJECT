# Research Paper Artifacts: IEEE Manuscript Package

This directory contains the IEEE research manuscript and publication-readiness artifacts for the project:

**Deadline-Aware Adaptive Inference with Drop-in Depthwise Convolution: A Runtime Evaluation on an RTX 4060**

---

## 1. Directory Structure

- `main.tex`: Full IEEE-style research manuscript written in standard IEEEtran LaTeX.
- `references.bib`: BibTeX bibliography file with authentic, verifiable citations.
- `audit_notes.md`: Detailed evidence audit notes mapping every empirical claim and table value to its underlying JSON/CSV source.
- `figures/`: Diagnostic figures copied from verified Phase 9.1 and Phase 9.2 analyses:
  - `fair_e2e_hit_rate_p95.png`: End-to-end deadline hit rate comparison under p95.
  - `fair_model_hit_rate_p95.png`: Model-only deadline hit rate comparison under p95.
  - `gpu_steady_state_latency.png`: 100-iteration steady-state forward pass latencies.
  - `paired_hit_rate_delta.png`: Paired McNemar hit rate deltas across all three metrics.
- `tables/`: Structured data sources used in the paper.

---

## 2. Compilation Instructions

The manuscript `main.tex` is configured for compilation with `pdflatex` or `xelatex`:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

*Note on Local Environment:* The local host Windows environment currently lacks a pre-installed TeX distribution (`pdflatex` / `xelatex`). The LaTeX source `main.tex` and BibTeX file `references.bib` are completely self-contained and ready for compilation on Overleaf, arXiv, or any standard TeX Live / MiKTeX environment.
