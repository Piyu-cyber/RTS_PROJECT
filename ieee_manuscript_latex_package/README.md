# IEEE Manuscript LaTeX Source Package

This self-contained folder contains all source components required to edit and compile `revised_ieee_manuscript_refined.pdf`.

## Contents

- `revised_ieee_manuscript_refined.tex` — Main IEEEtran LaTeX manuscript source
- `references.bib` — BibTeX bibliography database (15 citations)
- `revised_ieee_manuscript_refined.pdf` — Compiled 10-page IEEE conference manuscript
- `compile.bat` — Windows build script (`pdflatex` + `bibtex` + cleanup)
- `figures/` — All 6 figures referenced in the manuscript:
  1. `system_architecture.jpg` (Fig. 1: End-to-end architecture)
  2. `fig2_perception_tradeoff.png` (Fig. 2: Perception accuracy vs. token pruning)
  3. `fig3_gpu_latency_profiles.png` (Fig. 3: Eager vs. CUDA Graph latency & percentiles)
  4. `fig4_scheduling_ontime_comparison.png` (Fig. 4: On-time completion across stress workloads)
  5. `fig5_request_outcomes_stacked.png` (Fig. 5: Stacked request outcomes breakdown)
  6. `fig6_hardware_replay_validation.png` (Fig. 6: Simulation vs. physical GPU makespan)

## How to Edit & Compile

### Overleaf
Zip this entire `ieee_manuscript_latex_package` folder and upload it as a new project in Overleaf. Set the compiler to `pdfLaTeX` and the main document to `revised_ieee_manuscript_refined.tex`.

### Local Command Line (Windows / MiKTeX or TeX Live)
Run `compile.bat` inside this folder, or execute:
```powershell
pdflatex -interaction=nonstopmode revised_ieee_manuscript_refined.tex
bibtex revised_ieee_manuscript_refined
pdflatex -interaction=nonstopmode revised_ieee_manuscript_refined.tex
pdflatex -interaction=nonstopmode revised_ieee_manuscript_refined.tex
```
