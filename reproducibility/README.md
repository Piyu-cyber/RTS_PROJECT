# Deadline-Aware Adaptive Vision Transformer Inference for Soft Real-Time Autonomous Driving

[![Python 3.11](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-red.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 19 Passed](https://img.shields.io/badge/Tests-19%20Passed-brightgreen.svg)]()

> **End-to-End Reproducibility Package**  
> Authors: **Piyush Prashant, Priyanshu Mittal, Thejas Gowda U M, Shriyansh S Sahoo**  
> Target Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)  
> Operating System: Windows 11 / Linux (Ubuntu 22.04+)

---

## Table of Contents
1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Setup & Verification](#4-dataset-setup--verification)
5. [End-to-End Reproduction Pipeline (Steps 00 to 11)](#5-end-to-end-reproduction-pipeline)
6. [Running the Test Suite](#6-running-the-test-suite)
7. [Paper Artifact Mapping (Tables I–VI, Figures 2–6)](#7-paper-artifact-mapping)
8. [Citation & Contact](#8-citation--contact)

---

## 1. Overview

This repository provides the complete, self-contained, reproducible codebase for our IEEE manuscript investigating **Deadline-Aware Adaptive Vision Transformer Inference for Soft Real-Time Autonomous Driving**.

### Core Contributions:
1. **Perception Optimization**:
   - **Class-Balanced Cross-Entropy (CB-CE)**: Programmatic inverse-frequency weighting addressing extreme BDD100K class imbalance ($43,581$ city street samples vs. $27$ gas station samples).
   - **Correctness-Gated Margin Preservation Regularization (CMPR)**: Safety-margin distillation from a full-capacity teacher to dynamically pruned student models, preventing tail-class degradation.
2. **Real-Time Systems Co-Design**:
   - **Deadline-Aware Adaptive Scheduling with Admission Control (DA-ASAC)**: Cross-layer inference controller combining $O(|Q|)$ queue-tail admission control, Earliest Deadline First (EDF) dispatch, and slack-aware dynamic model switching across EViT keep-rates ($k \in \{1.0, 0.9, 0.7, 0.5\}$).
   - **Static CUDA Graph Acceleration**: Eliminates host driver overhead, providing microsecond-level deterministic execution bounds ($3.68\text{ ms}$ for DeiT, $2.81\text{ ms}$ for EViT-K05).

---

## 2. Repository Structure

```
reproducibility/
├── README.md                           # Master reproduction tutorial (this file)
├── FINAL_VALIDATION_REPORT.md          # Comprehensive verification report & audit
├── requirements.txt                    # Pinned Python package dependencies
├── pyproject.toml                      # Build specification & author metadata
├── configs/                            # Modular experiment configurations
│   ├── dataset.yaml                    # Split counts, classes, and paths
│   ├── deit.yaml                       # DeiT-Small baseline architecture
│   ├── evit_k09.yaml                   # EViT k=0.9 (148 tokens, 4.03 GMACs)
│   ├── evit_k07.yaml                   # EViT k=0.7 (72 tokens, 3.03 GMACs)
│   ├── evit_k05.yaml                   # EViT k=0.5 (28 tokens, 2.31 GMACs)
│   ├── training.yaml                   # Optimization hyperparameters & schedules
│   ├── cmpr.yaml                       # CMPR loss parameters (lambda=0.5, gamma=0.0)
│   ├── benchmark.yaml                  # Latency & throughput benchmark parameters
│   └── scheduler.yaml                  # DA-ASAC parameters, 10 workloads, 7 policies
├── src/                                # Core Python source library
│   ├── utils/                          # Seeding, SHA-256 hashing, environment logging
│   ├── data/                           # BDD100K datasets, transforms, manifest validators
│   ├── models/                         # DeiT-Small, EViT pruning, token fusion
│   ├── losses/                         # Class-balanced CE, CMPR loss, composite objective
│   ├── training/                       # Teacher loader, checkpointing, AMP trainer loop
│   ├── evaluation/                     # Metrics, confusion matrices, Gas Station analysis
│   ├── benchmarking/                   # Eager PyTorch, CUDA Graph, throughput, memory
│   ├── scheduler/                      # Request lifecycle, queues, DA-ASAC simulator, replay
│   └── analysis/                       # Table generators, figure plotters, manifest
├── scripts/                            # Sequential executable CLI entrypoints
│   ├── 00_check_environment.py         # Hardware & dependency diagnostic
│   ├── 01_prepare_dataset.py           # Split manifest generation & filtering
│   ├── 02_validate_dataset.py          # Disjointness check & class weights verification
│   ├── 03_train_teacher.py             # DeiT teacher baseline training / check
│   ├── 04_train_cross_keep_rate.py     # 11-model cross-keep-rate study
│   ├── 05_train_multiseed.py           # 3-seed reproducibility matrix (42, 123, 456)
│   ├── 06_evaluate_all.py              # Test set evaluation & metric reconstruction
│   ├── 07_run_statistics.py            # Bootstrap CIs, paired t-tests, McNemar tests
│   ├── 08_benchmark_models.py          # RTX 4060 latency & throughput profiling
│   ├── 09_run_scheduler_simulation.py  # 70 workload-policy simulations & conservation checks
│   ├── 10_run_hardware_replay.py       # Physical GPU trace replay on 100 images
│   └── 11_generate_artifacts.py        # Master artifact recreation (Tables & Figures)
├── tests/                              # PyTest test suite (19 unit/integration tests)
│   ├── test_accounting.py              # Verification of 4 real-time conservation laws
│   ├── test_cmpr.py                    # Margin, teacher gate, zero-correct safety
│   ├── test_dataset.py                 # Split counts, disjointness, class weights
│   ├── test_evit.py                    # Model forward shapes & token sequence lengths
│   ├── test_metrics.py                 # Classification metrics & Gas Station sensitivity
│   ├── test_scheduler.py               # FIFO, EDF, admission control, adaptive dispatch
│   └── test_token_pruning.py           # Top-k selection, complement indexing, fusion
└── docs/                               # Detailed design & methodology documentation
    ├── methodology.md                  # Cross-layer co-design rationale
    ├── model_architecture.md           # DeiT and EViT technical specifications
    ├── training_protocol.md            # Loss formulations & optimizer configurations
    ├── benchmarking_protocol.md        # Deterministic CUDA Graph capture & timing
    ├── scheduler_design.md             # DA-ASAC algorithm & workload profiles
    └── troubleshooting.md              # Common errors & solutions
```

---

## 3. Environment Setup

### Prerequisites
- Python 3.11.x (tested on 3.11.9)
- NVIDIA GPU with CUDA 12.1+ compute capability (tested on RTX 4060 Laptop GPU, 8 GB VRAM)
- Git

### Installation
Clone the repository and set up a clean Python virtual environment:

```powershell
# Clone or navigate to the project directory
cd C:\Users\palak\Desktop\RTS_PROJECT

# Create virtual environment
python -m venv .venv_evit

# Activate environment (Windows PowerShell)
.\.venv_evit\Scripts\Activate.ps1

# Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r reproducibility/requirements.txt
```

Verify your environment using Script 00:
```powershell
python reproducibility/scripts/00_check_environment.py
```

---

## 4. Dataset Setup & Verification

The project evaluates on the **BDD100K Driving Scene Classification** benchmark with 6 standardized categories:
`city street`, `highway`, `residential`, `parking lot`, `gas stations`, `tunnel`.

Run Script 01 to build the cleaned manifests (filtering 53 undefined scenes):
```powershell
python reproducibility/scripts/01_prepare_dataset.py
```

Run Script 02 to mathematically verify split counts, mutual disjointness, and class weights:
```powershell
python reproducibility/scripts/02_validate_dataset.py
```
**Expected Output:**
- `Train`: $69,634$ samples
- `Val`: $9,947$ samples (exactly 53 undefined scenes excluded from 10,000)
- `Test`: $19,902$ samples
- `Total`: $99,483$ valid samples
- `Mutual Disjointness`: $\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$
- `Class Weights`: $[0.3393, 0.4268, 0.5167, 1.1118, 2.1507, 1.4547]$ (max error $= 0.000000$)

---

## 5. End-to-End Reproduction Pipeline

Execute the numbered scripts in sequence to reproduce all findings:

### Step 03: Teacher Model Verification
```powershell
python reproducibility/scripts/03_train_teacher.py --verify-only
```
Verifies the DeiT-Small baseline teacher checkpoint and SHA-256 integrity.

### Step 04: Cross-Keep-Rate Matrix Verification
```powershell
python reproducibility/scripts/04_train_cross_keep_rate.py --verify-only
```
Audits all 11 model checkpoints across keep-rates $k \in \{1.0, 0.9, 0.7, 0.5\}$ and objectives (CE, CB-CE, CMPR, CB-CE+CMPR).

### Step 05: Multi-Seed Reproducibility Verification
```powershell
python reproducibility/scripts/05_train_multiseed.py --verify-only
```
Audits the complete 3-seed validation matrix (seeds 42, 123, 456) for K07-CB and K07-CB-CMPR.

### Step 06: Test Set Evaluation & Metric Reconstruction
```powershell
python reproducibility/scripts/06_evaluate_all.py
```
Reconstructs classification accuracy, Macro-F1, Weighted-F1, and per-class metrics from saved predictions.

### Step 07: Statistical Hypothesis Testing & Gas Station Analysis
```powershell
python reproducibility/scripts/07_run_statistics.py
```
Computes paired Student's t-tests, bootstrap confidence intervals, and the mathematical decomposition of the Gas Station class ($N_{\text{test}} = 6$).

### Step 08: Hardware Latency & Throughput Benchmarking
```powershell
# Verify canonical benchmark records (Table III)
python reproducibility/scripts/08_benchmark_models.py

# Optional: Run live GPU hardware profiling
python reproducibility/scripts/08_benchmark_models.py --benchmark-hardware
```

### Step 09: Real-Time Scheduling Discrete-Event Simulation
```powershell
python reproducibility/scripts/09_run_scheduler_simulation.py
```
Executes all 70 configurations ($10\text{ workloads} \times 7\text{ policies}$) and validates the **4 fundamental conservation laws** on every single run.

### Step 10: Physical GPU Trace Replay Validation
```powershell
python reproducibility/scripts/10_run_hardware_replay.py
```
Validates discrete-event simulation fidelity against live GPU execution with static CUDA Graphs over 100 test images across 9 configurations.

### Step 11: Master Artifact Generation (Tables & Figures)
```powershell
python reproducibility/scripts/11_generate_artifacts.py
```
Recreates all manuscript tables in LaTeX and CSV format, and generates Figures 2 to 6 in high-resolution PNG format.

---

## 6. Running the Test Suite

The test suite contains 19 comprehensive unit and integration tests covering data integrity, model forward passes, pruning algorithms, loss functions, scheduling priority queues, and real-time accounting equations.

Run the test suite with:
```powershell
pytest reproducibility/tests/
```

**Expected Result:**
```
============================= 19 passed in 17.21s =============================
```

---

## 7. Paper Artifact Mapping

Every table and figure in the manuscript is generated programmatically from verified data:

| Manuscript Item | Description | Generating Script | Output Artifact |
| :--- | :--- | :--- | :--- |
| **Table I** | 11-Model Cross-Keep-Rate Comparison | `scripts/11_generate_artifacts.py` | `artifacts/tables/table1_cross_keep_rate.tex` |
| **Table II** | 3-Seed Validation Matrix ($k=0.7$) | `scripts/11_generate_artifacts.py` | `artifacts/tables/table2_multi_seed.tex` |
| **Table III** | Hardware Profiles on RTX 4060 | `scripts/11_generate_artifacts.py` | `artifacts/tables/table3_hardware_profiles.tex` |
| **Table V** | 70-Config Simulation Results | `scripts/09_run_scheduler_simulation.py` | `artifacts/scheduler/simulation_results.csv` |
| **Table VI** | Simulation vs. GPU Replay Validation | `scripts/11_generate_artifacts.py` | `artifacts/tables/table6_hardware_replay.tex` |
| **Figure 2** | Perception-Efficiency Pareto Front | `scripts/11_generate_artifacts.py` | `artifacts/figures/fig2_perception_tradeoff.png` |
| **Figure 3** | GPU Latency Distributions | `scripts/11_generate_artifacts.py` | `artifacts/figures/fig3_gpu_latency_profiles.png` |
| **Figure 4** | Real-Time QoS Across 10 Workloads | `scripts/11_generate_artifacts.py` | `artifacts/figures/fig4_scheduling_ontime_comparison.png` |
| **Figure 5** | Stacked Request Outcomes under Stress | `scripts/11_generate_artifacts.py` | `artifacts/figures/fig5_request_outcomes_stacked.png` |
| **Figure 6** | Hardware Replay Fidelity Validation | `scripts/11_generate_artifacts.py` | `artifacts/figures/fig6_hardware_replay_validation.png` |

The complete traceability mapping is serialized in machine-readable JSON at `reproducibility/artifacts/artifact_manifest.json`.

---

## 8. Citation & Contact

If you use this research codebase or findings in your academic work, please cite our paper:

```bibtex
@article{prashant2026deadline,
  title={Deadline-Aware Adaptive Vision Transformer Inference for Soft Real-Time Autonomous Driving},
  author={Prashant, Piyush and Mittal, Priyanshu and Gowda U M, Thejas and Sahoo, Shriyansh S},
  journal={IEEE Transactions on Intelligent Vehicles (Under Review)},
  year={2026}
}
```
