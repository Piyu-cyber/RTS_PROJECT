# Comprehensive Reproducibility and Codebase Audit

**Project:** *Deadline-Aware Adaptive Vision Transformer Inference for Soft Real-Time Edge Workloads*  
**Date:** November 6, 2026  
**Author:** Senior Research Software Engineering Audit  
**Location:** `reproducibility/REPRODUCIBILITY_AUDIT.md`  

---

## 1. Executive Summary

This audit assesses the state of the research repository at `C:\Users\palak\Desktop\RTS_PROJECT` to transition it into an independent, end-to-end reproducible research package. 

The project investigates edge-deployed Vision Transformers for 6-class autonomous driving scene perception on the Berkeley DeepDrive (BDD100K) dataset. It spans model architecture (DeiT-Small baseline and EViT token pruning across $k \in \{0.9, 0.7, 0.5\}$), loss formulations (Class-Balanced Cross-Entropy and teacher-guided Class-Margin-Preserving Regularization), physical GPU benchmarking (eager mode vs. static CUDA Graphs on an NVIDIA GeForce RTX 4060 Laptop GPU), and soft real-time scheduling (EDF, slack-aware model selection, and queue-tail admission control).

The experimental findings culminated in a 10-page IEEE conference manuscript. The underlying experimental artifacts (checkpoints, training configurations, raw simulation results, GPU traces, and confusion matrices) are preserved across `outputs/`. However, the repository evolved across multiple iterative project phases (from early Pet-dataset tests up to Phase 24 and the final 3-seed validation). As a result, code was distributed across phase-specific subpackages (`src/phase23b/`, `src/phase24/`, loose `scripts/`), with local absolute paths, duplicated dataset loading logic, and no unified reproducible entrypoint.

This document establishes the inventory of existing assets, diagnoses missing and incomplete components, documents resolved inconsistencies, and defines the target modular structure for the clean, isolated `reproducibility/` package.

---

## 2. Repository Structure and Inventory

### 2.1 Directory Structure Overview

```text
RTS_PROJECT/
├── baseline/                   # Early baseline experiments
├── benchmarks/                 # Early latency and throughput logs
├── data/
│   └── bdd100k/
│       ├── annotations/        # Split JSON manifests and summaries
│       │   ├── train_full.json # 69,634 valid training annotations
│       │   ├── val_full.json   # 9,947 valid validation annotations
│       │   ├── test_full.json  # 19,902 valid test annotations
│       │   ├── train.json      # 1,049-sample smoke test split
│       │   ├── val.json        # 225-sample smoke test split
│       │   └── test.json       # 225-sample smoke test split
│       └── images/             # Extracted image directories
├── datasets/                   # Legacy Oxford-IIIT Pet datasets
├── outputs/
│   ├── bdd100k_migration/      # Initial BDD100K baseline training & simulation
│   │   ├── checkpoints/        # Baseline DeiT and EViT checkpoints
│   │   ├── corrected_replay/   # Physical GPU hardware replay results (Table VI)
│   │   ├── full_dataset_latency_profiles.csv # Batch-1 latency percentiles (Table III)
│   │   └── simulation_results.csv # 70-configuration simulation matrix (Table V)
│   ├── cmpr_matched_training/  # Matched study comparing M0–M7 on EViT k05
│   │   └── checkpoints/        # Checkpoints for M0 through M7 (seed 42)
│   ├── cross_keep_rate_study/  # Complete 11-model cross-keep-rate study
│   │   ├── checkpoints/        # Best and final checkpoints (seed 42)
│   │   ├── cross_keep_rate_results.csv # 11-model comparison metrics (Table I)
│   │   ├── efficiency_benchmark.csv    # GMACs and throughput (Table IV)
│   │   └── multi_seed_validation/      # 3-seed validation of K07-CB vs K07-CB-CMPR
│   │       ├── seed_123/       # Checkpoints, configs, logs, confusion matrices
│   │       ├── seed_456/       # Checkpoints, configs, logs, confusion matrices
│   │       ├── aggregate_results.csv   # Aggregated 3-seed metrics (Table II)
│   │       ├── efficiency_results.csv  # Verified hardware metrics
│   │       ├── five_class_macro_f1.csv # Supplementary 5-class Macro-F1
│   │       └── gas_station_macro_contribution.csv # Decomposition of support effect
│   └── manuscript_revision/    # LaTeX sources, figures, and compiled IEEE manuscript
├── reference/
│   ├── evit/                   # Official EViT reference implementation (Liang et al., ICLR 2022)
│   └── evit_weights/           # Official ImageNet-1K pretrained DeiT and EViT weights
├── scripts/                    # 35 loose workflow scripts spanning phases
├── src/
│   ├── interactive_rts/        # Interactive demonstration server
│   ├── phase23b/               # BDD dataset loaders and model wrappers
│   ├── phase23_evit/           # EViT migration utilities
│   ├── phase24/                # Discrete-event simulator & admission control
│   └── phase24b/               # Robustness test harness
└── tests/                      # Legacy unit tests for earlier phases
```

### 2.2 Inventory of Experimental Checkpoints

All experimental checkpoints reported in the manuscript are preserved on disk with intact weights:

| Model Configuration | Seed | Location in `outputs/` | Status |
| :--- | :---: | :--- | :--- |
| **DeiT Baseline (Teacher)** | 42 | `cmpr_matched_training/checkpoints/M0_seed42_best.pth` | Verified |
| **K09-CE** | 42 | `cross_keep_rate_study/checkpoints/K09-CE_seed42_best.pth` | Verified |
| **K09-CB** | 42 | `cross_keep_rate_study/checkpoints/K09-CB_seed42_best.pth` | Verified |
| **K09-CB-CMPR** | 42 | `cross_keep_rate_study/checkpoints/K09-CB-CMPR_seed42_best.pth` | Verified |
| **K07-CE** | 42 | `cross_keep_rate_study/checkpoints/K07-CE_seed42_best.pth` | Verified |
| **K07-CB** | 42 | `cross_keep_rate_study/checkpoints/K07-CB_seed42_best.pth` | Verified |
| **K07-CB-CMPR** | 42 | `cross_keep_rate_study/checkpoints/K07-CB-CMPR_seed42_best.pth` | Verified |
| **K05-CE** | 42 | `cmpr_matched_training/checkpoints/M1_seed42_best.pth` | Verified |
| **K05-CMPR** | 42 | `cmpr_matched_training/checkpoints/M5_seed42_best.pth` | Verified |
| **K05-CB** | 42 | `cross_keep_rate_study/checkpoints/K05-CB_seed42_best.pth` | Verified (retrained) |
| **K05-CB-CMPR** | 42 | `cross_keep_rate_study/checkpoints/K05-CB-CMPR_seed42_best.pth` | Verified (retrained) |
| **K07-CB** | 123 | `multi_seed_validation/seed_123/checkpoints/K07-CB_seed123_best.pth` | Verified |
| **K07-CB-CMPR** | 123 | `multi_seed_validation/seed_123/checkpoints/K07-CB-CMPR_seed123_best.pth` | Verified |
| **K07-CB** | 456 | `multi_seed_validation/seed_456/checkpoints/K07-CB_seed456_best.pth` | Verified |
| **K07-CB-CMPR** | 456 | `multi_seed_validation/seed_456/checkpoints/K07-CB-CMPR_seed456_best.pth` | Verified |

---

## 3. Scripts That Are Incomplete, Missing, or Fragile

While the core algorithms were successfully executed, an external researcher would face substantial roadblocks reproducing the project from the root repository:

1. **Hardcoded Machine-Specific Paths:** Multiple scripts in `scripts/` contain literal paths such as `C:\Users\palak\Desktop\RTS_PROJECT`, preventing direct cloning and execution on another machine or path.
2. **Scattered Execution Scripts:** Running the pipeline currently requires knowing an undocumented sequence of scripts across `scripts/` (e.g., running `parse_full_100k_annotations.py`, then `run_cross_keep_rate_experiments.py`, then `compile_multi_seed_validation.py`, then `run_phase24.py`).
3. **No Standalone Loss & Architecture Modules:** The CMPR loss was defined inline inside `scripts/run_cross_keep_rate_experiments.py` rather than as a reusable, importable module with comprehensive unit tests.
4. **No Formal Configuration System:** Experiment parameters were passed via ad-hoc command-line arguments and scattered dictionaries rather than structured YAML configuration files.
5. **No Single Artifact Regeneration Script:** Tables and plots were generated by disparate scripts (`compile_cross_keep_rate_analysis.py`, `compile_multi_seed_validation.py`, `generate_statistical_report.py`), making it difficult to regenerate all manuscript figures from existing data in one command.
6. **No Self-Contained Smoke Test Suite:** There was no automated test verifying the entire pipeline end-to-end on a minimal dataset subset before committing to hours of GPU training.

---

## 4. Dependencies and Environment Assumptions

The repository was developed and profiled under:
- **Operating System:** Windows 11 Home / Pro (x64)
- **Host GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (AD107M, 3,072 CUDA cores, 8 GB GDDR6)
- **Graphics Driver:** NVIDIA 551.xx / 560.xx
- **Python Version:** 3.11.9
- **PyTorch Build:** `torch==2.5.1+cu121`
- **TorchVision Build:** `torchvision==0.20.1+cu121`
- **Key Libraries:** `numpy==2.3.3`, `scipy>=1.13.0`, `pandas>=2.2.0`, `matplotlib>=3.9.0`, `scikit-learn>=1.5.0`, `pyyaml>=6.0`, `timm>=0.9.0`, `pytest>=8.0.0`

### Hardware Constraints & Assumptions:
- **CUDA Graphs on Windows:** CUDA Graph capture requires static memory allocations and fixed tensor shapes (FP32, batch size 1). Dynamic sequence lengths or variable batch sizes during graph capture trigger runtime CUDA exceptions.
- **VRAM Headroom:** Training with batch size 64 and mixed precision (`torch.amp.autocast`) consumes approximately 4.2 GB of VRAM, fitting comfortably within the 8 GB budget.

---

## 5. Results That Can Already Be Regenerated vs. Results Requiring Retraining

### 5.1 Fully Regenerable Immediately from Preserved Predictions & Logs
Because all checkpoint files, evaluation JSONs, confusion matrices, and trace replays are preserved in `outputs/`, the following manuscript deliverables can be regenerated instantly without GPU retraining:
1. **Manuscript Table I (Cross-Keep-Rate Generalization):** Reconstructed directly from `outputs/cross_keep_rate_study/cross_keep_rate_results.csv` and confusion matrices.
2. **Manuscript Table II (Multi-Seed Validation):** Reconstructed directly from `outputs/cross_keep_rate_study/multi_seed_validation/` per-seed results.
3. **Manuscript Table III (CUDA Graph Latency Profiles):** Reconstructed from `full_dataset_latency_profiles.csv`.
4. **Manuscript Table IV (Computational Complexity & Throughput):** Reconstructed from `efficiency_results.csv`.
5. **Manuscript Table V (70-Configuration Scheduling Results):** Reconstructed from `simulation_results.csv`.
6. **Manuscript Table VI (Physical GPU Replay Parity):** Reconstructed from `corrected_replay/hardware_replay_comparison.csv`.
7. **Figures 1 through 6:** Regenerated cleanly using the analysis plotting scripts.

### 5.2 Results Requiring Full Training Runs
If a researcher chooses to retrain all models from scratch:
- **Teacher Training (DeiT-Small):** ~45 minutes (3 epochs on 69,634 images, batch size 64).
- **Cross-Keep-Rate Study (11 models):** ~8 hours total GPU time.
- **Three-Seed Validation (K07-CB and K07-CB-CMPR across seeds 123 and 456):** ~3.2 hours total GPU time.
- Total compute budget for full training reproduction: **~12 hours** on an NVIDIA RTX 4060 GPU.

---

## 6. Known Inconsistencies and Their Resolutions

During the research phases, three specific inconsistencies were identified and resolved:

### 6.1 Validation Sample Count: 9,947 vs. 9,951
- **Root Cause:** The official BDD100K validation split contains 10,000 images. Exactly 53 images have undefined scene labels. Some early phase scripts used a loose filter that retained 4 undefined images, reporting 9,951 samples.
- **Resolution:** The audited dataset manifest strictly filters all 53 undefined scenes, establishing the authoritative validation count of **9,947 valid images** (Total valid: $69,634 + 9,947 + 19,902 = 99,483$). The codebase strictly uses 9,947.

### 6.2 Class Weights Formula and Runtime Tensor
- **Root Cause:** Historical experiments in early Phase 10 used unnormalized class weights.
- **Resolution:** The smoothed formula $w_c^{raw} = (N / (C \cdot n_c))^{0.25}$ with mean normalization $w_c = w_c^{raw} / (\frac{1}{C}\sum_j w_j^{raw})$ was programmatically implemented and audited. The verified runtime tensor:
  $$\mathbf{w} = [0.3393, 0.4268, 0.5167, 1.1118, 2.1507, 1.4547]$$
  All cross-keep-rate and multi-seed checkpoints strictly use this verified tensor.

### 6.3 EViT Token Sequence Lengths
- **Verification:** Forward pass hooks on the actual EViT implementation confirm the exact sequence lengths across all drop stages (Blocks 3, 6, 9) including the fused summary token and CLS token:
  - **DeiT Baseline:** 197 tokens
  - **EViT K09:** $197 \to 179 \to 163 \to 148$ tokens (Final: **148**)
  - **EViT K07:** $197 \to 140 \to 100 \to 72$ tokens (Final: **72**)
  - **EViT K05:** $197 \to 100 \to 52 \to 28$ tokens (Final: **28**)

---

## 7. Proposed Reproducibility Codebase Architecture

To ensure clean isolation and complete reproducibility without mutating the original project artifacts, the entire reproducible system will be housed in `reproducibility/`:

```text
reproducibility/
├── README.md                           # Master step-by-step reproduction guide
├── REPRODUCIBILITY_AUDIT.md            # This audit report
├── FINAL_VALIDATION_REPORT.md          # Comprehensive verification report
├── requirements.txt                    # Exact pinned dependencies
├── pyproject.toml                      # Standard packaging manifest
├── .gitignore                          # Clean VCS exclusions
├── configs/                            # Modular YAML configuration files
│   ├── dataset.yaml                    # Splits, paths, class names, counts
│   ├── deit.yaml                       # Baseline model hyperparameters
│   ├── evit_k09.yaml                   # EViT 0.9 pruning configuration
│   ├── evit_k07.yaml                   # EViT 0.7 pruning configuration
│   ├── evit_k05.yaml                   # EViT 0.5 pruning configuration
│   ├── training.yaml                   # Optimizer, learning rate, warmup
│   ├── cmpr.yaml                       # Margin loss, teacher gating, gamma
│   ├── benchmark.yaml                  # Latency and throughput benchmark configs
│   └── scheduler.yaml                  # Real-time simulation & workload configs
├── src/                                # Self-contained modular Python source
│   ├── data/                           # Manifest generation, transforms, validation
│   ├── models/                         # DeiT and EViT implementations & builder
│   ├── losses/                         # Class-balanced CE & gated CMPR
│   ├── training/                       # Engine, trainer, checkpointing
│   ├── evaluation/                     # Metric calculator, bootstrap CI, Gas analysis
│   ├── benchmarking/                   # Eager, CUDA Graph, throughput, memory
│   ├── scheduler/                      # DA-ASAC, EDF priority queue, simulator, replay
│   ├── analysis/                       # Table and figure generators, manifest
│   └── utils/                          # Seed control, logging, device environment
├── scripts/                            # Ordered numerical CLI entrypoints (00 to 11)
├── tests/                              # PyTest test suite (unit + smoke tests)
└── docs/                               # Detailed technical methodology documentation
```
