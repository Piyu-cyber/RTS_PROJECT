# Deadline-Aware Adaptive Inference-Mode Controller

## Purpose

This module implements a deterministic, standalone controller that selects the
best DINOv2 inference mode (full attention, moderate DWConv, or aggressive
DWConv) for a given remaining deadline.  It uses empirical latency bounds and
validation mIoU from the Phase 6B benchmark to make the selection.

**The controller is a deterministic selector.  It does not load neural-network
checkpoints or execute inference.**

## Scope

The controller answers: *"Given X milliseconds remaining before a deadline,
which inference mode should a real-time scheduler dispatch?"*

It reads the Phase 6B benchmark JSON, compares each mode's empirical latency
bound against the deadline, and returns the highest-accuracy mode that fits.

## Algorithm

### Inputs

| Input | Description |
|---|---|
| `remaining_deadline_ms` | Remaining time budget before the request deadline (ms). |
| `modes` | Collection of available inference modes with latency and accuracy profiles. |
| `latency_metric` | Which empirical statistic to use as the feasibility bound (`p95`, `p99`, or `max_observed`). |

### Step A — Compute Feasibility

A mode is **feasible** when:

```
mode.empirical_latency_bound_ms <= remaining_deadline_ms
```

A mode whose bound **exactly equals** the remaining deadline is feasible.

### Step B — Select the Most Accurate Feasible Mode

If one or more modes are feasible:

```python
selected = min(
    feasible_modes,
    key=lambda m: (-m.validation_miou, m.empirical_latency_bound_ms, m.name),
)
likely_deadline_miss = False
```

1. Highest validation mIoU.
2. Ties broken by lower empirical latency bound.
3. Further ties broken by lexicographically smaller mode name.

### Step C — Fallback When No Mode Is Feasible

```python
selected = min(
    all_modes,
    key=lambda m: (m.empirical_latency_bound_ms, -m.validation_miou, m.name),
)
likely_deadline_miss = True
```

1. Lowest empirical latency bound.
2. Ties broken by higher validation mIoU.
3. Further ties broken by lexicographically smaller mode name.

The controller always returns a selected mode rather than failing.

## Output Structure

The `SelectionResult` dataclass contains:

| Field | Type | Description |
|---|---|---|
| `selected_mode` | `str` | Name of the chosen inference mode. |
| `selected_miou` | `float` | Validation mIoU of the chosen mode. |
| `remaining_deadline_ms` | `float` | The input deadline. |
| `latency_metric` | `str` | Which metric was used (`p95`, `p99`, `max_observed`). |
| `selected_empirical_latency_bound_ms` | `float` | The selected mode's latency bound for the chosen metric. |
| `feasible_modes` | `list[str]` | Sorted names of all feasible modes. |
| `likely_deadline_miss` | `bool` | `True` if no mode's bound fits the deadline. |
| `all_mode_bounds` | `dict[str, float]` | Latency bound for every available mode. |
| `reason` | `str` | Human-readable explanation of the selection. |

## Benchmark JSON Source

The authoritative benchmark file is:

```
outputs/inference_benchmark/phase6_finetuned_results.json
```

This file is produced by Phase 6B (`benchmarks/benchmark_inference_modes.py`).
The controller reads the following fields from the JSON:

```
modes.<mode_name>.accuracy.mean_iou          → validation mIoU
modes.<mode_name>.latency_ms.p95_ms          → p95 latency
modes.<mode_name>.latency_ms.p99_ms          → p99 latency
modes.<mode_name>.latency_ms.max_observed_ms → maximum observed latency
```

No values are hardcoded or fabricated.  The controller loads them at runtime.

## Supported Empirical Latency Metrics

| CLI flag | JSON field | Description |
|---|---|---|
| `p95` | `latency_ms.p95_ms` | 95th percentile of observed latency. |
| `p99` | `latency_ms.p99_ms` | 99th percentile of observed latency. |
| `max_observed` | `latency_ms.max_observed_ms` | Maximum observed latency. |

The controller never silently substitutes one metric for another.

## How to Run the CLI

From the project root (`C:\Users\palak\Desktop\RTS_PROJECT`):

```powershell
# All modes feasible at 25 ms p95
.\.venv\Scripts\python.exe -m realtime.run_controller --deadline-ms 25 --metric p95

# No modes feasible at 15 ms p95
.\.venv\Scripts\python.exe -m realtime.run_controller --deadline-ms 15 --metric p95

# No modes feasible at 10 ms p95
.\.venv\Scripts\python.exe -m realtime.run_controller --deadline-ms 10 --metric p95

# No modes feasible at 10 ms max_observed
.\.venv\Scripts\python.exe -m realtime.run_controller --deadline-ms 10 --metric max_observed

# Boundary test at 18.1 ms p95
.\.venv\Scripts\python.exe -m realtime.run_controller --deadline-ms 18.1 --metric p95

# Custom benchmark file
.\.venv\Scripts\python.exe -m realtime.run_controller --deadline-ms 25 --benchmark path\to\custom.json
```

### CLI Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--deadline-ms` | Yes | — | Remaining deadline in milliseconds. |
| `--metric` | No | `p95` | Latency metric: `p95`, `p99`, or `max_observed`. |
| `--benchmark` | No | Phase 6B file | Path to the benchmark JSON file. |

### Exit Codes

| Code | Meaning |
|---|---|
| 0 | Successful selection (even if `likely_deadline_miss=true`). |
| 1 | Invalid arguments, malformed benchmark, or validation error. |

## How to Run Unit Tests

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_deadline_controller -v
```

## Feasibility Definition

A mode is **feasible** if and only if its configured empirical latency bound
(for the selected metric) is less than or equal to the remaining deadline.

```
empirical_latency_bound_ms <= remaining_deadline_ms
```

This is a necessary condition for meeting the deadline under the empirical
model, but it is **not sufficient** — actual execution times are stochastic.

## Accuracy-First Selection

Among feasible modes, the controller always selects the mode with the highest
validation mIoU.  This maximizes output quality within the timing constraint.

## Tie-Breaking Rules

1. **Feasible selection:** highest mIoU → lowest latency bound → lexicographic name.
2. **Fallback selection:** lowest latency bound → highest mIoU → lexicographic name.

These rules ensure deterministic behavior regardless of input ordering.

## Fastest-Mode Fallback

When no mode is feasible, the controller selects the mode with the lowest
empirical latency bound.  This gives the best chance of meeting (or nearly
meeting) the deadline, even though a miss is likely.

## Meaning of `likely_deadline_miss`

| Value | Meaning |
|---|---|
| `false` | The selected mode's empirical bound fits the deadline. |
| `true` | No mode's empirical bound fits the deadline.  A deadline miss is likely. |

This flag is an empirical prediction, not a formal guarantee.  When `false`,
the actual execution may still exceed the deadline (the bound is statistical).
When `true`, execution may still meet the deadline (actual latency may fall
below the statistical bound).

## Limitations and Assumptions

1. **Deterministic selector only.**  The controller does not execute inference,
   load checkpoints, or interact with the GPU.

2. **Measurement scope.**  The latency measurements cover only device-resident
   backbone feature extraction plus the segmentation head.  **Preprocessing and
   host-to-device transfer are excluded** from the current measurements.

3. **Empirical statistics, not WCET.**  The p95, p99, and maximum observed
   latency values are empirical statistics from 100 CUDA-event measurements,
   not formal worst-case execution-time guarantees.

4. **A feasible empirical bound is not a hard real-time guarantee.**  Even when
   `likely_deadline_miss=false`, the actual inference latency may exceed the
   selected bound on any given invocation.

5. **Workload specificity.**  All results come from the current Oxford-IIIT Pet
   binary foreground/background segmentation validation setup (32 images,
   224×224, RTX 4060 Laptop GPU).  They should not be generalized to all
   workloads without further evaluation.

6. **Static mode set.**  The controller does not discover modes dynamically; it
   uses the modes present in the benchmark JSON.

7. **No preemption or scheduling.**  The controller assumes exclusive GPU
   access.  Contention from other processes may increase actual latency.

---

# Phase 8 — Simulation and Comparative Evaluation of Scheduling Strategies

## Purpose

Phase 8 provides an offline, reproducible experimental comparison of three
deadline-scheduling strategies on exactly identical synthetic request streams:

1. **`always_full_attention`**: Static baseline always executing full attention.
2. **`always_fastest`**: Static baseline always selecting the mode with the lowest configured empirical latency bound.
3. **`adaptive_controller`**: Dynamic selector using the Phase 7 accuracy-first deadline-aware algorithm.

The simulation uses metadata from [`outputs/inference_benchmark/phase6_finetuned_results.json`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/inference_benchmark/phase6_finetuned_results.json). It does not execute live neural network inference or retrain models.

## Strategies

| Strategy | Mode Selection Policy | Feasibility Check |
|---|---|---|
| `always_full_attention` | Always selects `full_attention`. | `full_attention_bound <= deadline` |
| `always_fastest` | Always selects the mode with the lowest empirical bound under the chosen metric. | `fastest_bound <= deadline` |
| `adaptive_controller` | Selects highest-mIoU feasible mode; falls back to lowest-bound mode if none fit. | `selected_bound <= deadline` |

## Synthetic Workload Generator

Workloads are generated deterministically using a configurable random seed and request count ($N \ge 1{,}000$):

### `default_mixture`
- **Tight (`[10.0, 17.0]` ms, 30%)**: Deadlines challenging all modes; under p95, only deadlines in $[16.62, 17.0]$ ms are feasible for full attention.
- **Moderate (`[17.0, 22.0]` ms, 40%)**: Spans the thresholds differentiating mode feasibility across p95 and p99.
- **Relaxed (`[22.0, 32.0]` ms, 30%)**: Feasible for all or most modes across all metrics.

Every strategy receives the exact same request IDs, deadlines, and categories in identical order.

## How to Run Phase 8 Experiments

```powershell
# 1. Principal experiment (1,000 requests, seed 42, p95 metric)
.\.venv\Scripts\python.exe -m realtime.simulate_strategies --n-requests 1000 --seed 42 --metric p95 --output-dir outputs\phase8_simulation

# 2. Sensitivity experiment: p99 metric
.\.venv\Scripts\python.exe -m realtime.simulate_strategies --n-requests 1000 --seed 42 --metric p99 --output-dir outputs\phase8_simulation\sensitivity_p99

# 3. Sensitivity experiment: max_observed metric
.\.venv\Scripts\python.exe -m realtime.simulate_strategies --n-requests 1000 --seed 42 --metric max_observed --output-dir outputs\phase8_simulation\sensitivity_max_observed

# 4. Run all unit tests (Phases 7 & 8)
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

## Output Artifacts

All artifacts are saved to `outputs/phase8_simulation/`:
- `summary.json`: Configuration, mode profiles, aggregate & per-category metrics, and interpretation notes.
- `request_trace.csv`: Request-level trace (request ID, deadline, category, strategy, mode, mIoU, bound, hit/miss, fallbacks).
- `deadline_stream.json`: Full serialized synthetic request stream for independent replication.
- `hit_rate_by_strategy.png` / `.svg`: Visual comparison of empirical deadline-hit rate across strategies.
- `avg_miou_by_strategy.png` / `.svg`: Visual comparison of average selected mIoU across strategies.

## Metric Definitions

1. **Empirical Deadline-Hit Rate**: Fraction of requests where the selected mode's empirical bound $\le$ remaining deadline.
2. **Empirical Deadline-Miss Rate**: $1.0 - \text{Hit Rate}$.
3. **Average Selected mIoU**: Arithmetic mean of validation mIoU of selected modes across requests.
4. **Average Selected Latency Bound**: Mean empirical bound of selected modes across requests.
5. **Mode-Selection Frequency**: Count and percentage of requests assigned to each mode.
6. **Adaptive Fallback Count**: Requests where no mode was feasible, triggering fastest-mode fallback.
7. **Per-Category Metrics**: Separate hit rate, mean mIoU, and mean latency bound for tight, moderate, and relaxed categories.

## Scientific Findings and Interpretation

### 1. The p95 Inversion Phenomenon
Under the primary `p95` metric, `full_attention` has an empirical bound of **16.619 ms**, which is faster than both `aggressive_dwconv` (18.061 ms) and `moderate_dwconv` (18.650 ms). Because `full_attention` simultaneously possesses the highest accuracy (0.742990) and the lowest p95 latency bound, it strictly dominates all modes. Consequently:
- `always_full_attention`, `always_fastest`, and `adaptive_controller` achieve identical hit rates (73.40%) and identical mean mIoU (0.742990).
- The adaptive controller selects `full_attention` for 100% of requests.

### 2. High-Tail Latency Regimes (p99 and max_observed)
In contrast, in tail-risk regimes, DWConv patching provides genuine empirical latency advantages:
- **Under p99**: `moderate_dwconv` bound is **20.828 ms** vs. `full_attention` at **21.334 ms**.
  - `always_full_attention` hit rate drops to **35.50%**.
  - `always_fastest` (moderate) achieves **40.00%** hit rate (+4.5 pp), but sacrifices accuracy (mIoU 0.703581).
  - `adaptive_controller` achieves the **same maximum hit rate as fastest (40.00%)** while achieving **higher accuracy (0.717571)** by switching to `full_attention` whenever the deadline is relaxed ($\ge 21.33$ ms, 35.5% of requests) and using `moderate_dwconv` when the deadline is tight/moderate (64.5% of requests).
- **Under max_observed**: `moderate_dwconv` bound is **23.999 ms** vs. `full_attention` at **27.563 ms** (a 3.56 ms saving).
  - `always_full_attention` hit rate drops to **13.30%**.
  - `always_fastest` and `adaptive_controller` both achieve **24.50%** (+11.2 pp over full attention).
  - `adaptive_controller` delivers **0.708822 mean mIoU**, outperforming `always_fastest` (0.703581) by selecting `full_attention` on the 13.3% of requests with deadlines $\ge 27.56$ ms.

### 3. Aggressive DWConv Behavior
`aggressive_dwconv` is never selected under p95, p99, or max_observed. This occurs because its accuracy is substantially lower (0.456 vs 0.704) and its empirical latency bounds are worse than `moderate_dwconv` across all three metrics (and worse than `full_attention` under p95).

### 4. Real-Time Systems Limitations
- **Empirical bound vs. WCET**: All hit rates reflect compliance with empirical statistical thresholds (p95, p99, max_observed from 100 samples), not mathematical WCET.
- **Exclusion of Host Transfers**: Measurements only cover device-resident backbone + head inference. End-to-end pipelines must add image acquisition, resizing, and host-to-device copy latencies.
- **Single-Profile Assumption**: Simulation assumes static latency distributions; run-time GPU frequency scaling or thermal throttling would shift actual execution times.

---

# Phase 9 — Runtime Deadline-Aware Adaptive Inference Prototype

## Purpose

Phase 9 transitions the research from offline simulation (Phase 8) to a **live runtime prototype** executing neural network inference on an NVIDIA GeForce RTX 4060 Laptop GPU.

It evaluates whether the Phase 7 deadline-aware controller successfully guides live PyTorch inference on actual images, measures component-level latencies with hardware synchronization, and evaluates how empirical predicted feasibility compares with actual deadline satisfaction on an unmanaged desktop operating system.

## Key Architecture & Design

1. **Preloaded Zero-Overhead Multi-Model Registry (`RuntimeModelRegistry`)**:
   - Rather than dynamically patching attention weights on every request (which incurs 50–100 ms CPU overhead), all three models (`full_attention`, `moderate_dwconv`, and `aggressive_dwconv`) are instantiated, strict-checkpoint loaded, and preloaded into GPU VRAM at startup.
   - Total GPU memory footprint across all 3 models combined is only **246.5 MiB** (<3.5% of the 8 GB VRAM on the RTX 4060).
   - Dynamic mode switching is instantaneous ($O(1)$ dictionary lookup, 0.00 ms switching overhead).

2. **Component-Level Timing Instrumentation**:
   - `controller_time_ms`: Monotonic timer measuring controller mode selection (pure CPU).
   - `preprocess_time_ms`: Monotonic timer measuring PIL image resizing, RGB conversion, float normalization, and tensor allocation.
   - `host_to_device_time_ms`: Monotonic timer measuring CPU tensor transfer to GPU (`.to(device, non_blocking=True)`).
   - `model_time_ms`: Precise GPU execution duration measured via synchronized `torch.cuda.Event(enable_timing=True)`.
   - `postprocess_time_ms`: Monotonic timer measuring segmentation argmax extraction and mask generation.
   - `e2e_time_ms`: Complete request processing duration from request arrival to mask generation, bounded by `torch.cuda.synchronize()`.

3. **Predicted Feasibility vs. Actual Deadline Satisfaction**:
   - **`predicted_empirical_feasibility`**: Mode's configured empirical latency bound $\le$ request deadline (the Phase 7 controller's statistical expectation based on device-resident forward passes).
   - **`actual_deadline_hit`**: Live measured end-to-end latency $\le$ request deadline (`e2e_time_ms <= deadline_ms`).

## How to Run Phase 9 Experiments

```powershell
# 1. Run unit test suite
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_inference -v

# 2. Run principal runtime experiment (50 requests/strategy, seed 42, metric p95)
.\.venv\Scripts\python.exe -m realtime.run_runtime_experiment --n-requests 50 --seed 42 --metric p95 --output-dir outputs\phase9_runtime

# 3. Run sensitivity experiment: p99 metric
.\.venv\Scripts\python.exe -m realtime.run_runtime_experiment --n-requests 50 --seed 42 --metric p99 --output-dir outputs\phase9_runtime\sensitivity_p99

# 4. Run sensitivity experiment: max_observed metric
.\.venv\Scripts\python.exe -m realtime.run_runtime_experiment --n-requests 50 --seed 42 --metric max_observed --output-dir outputs\phase9_runtime\sensitivity_max_observed
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--n-requests` | `50` | Number of sequential requests per strategy. |
| `--seed` | `42` | Random seed for synthetic deadline stream generation. |
| `--metric` | `p95` | Empirical latency metric (`p95`, `p99`, `max_observed`). |
| `--benchmark` | Phase 6B JSON | Path to benchmark JSON. |
| `--output-dir` | `outputs/phase9_runtime` | Output directory for logs, CSV traces, summary JSON, and charts. |
| `--warmups` | `5` | Number of warm-up iterations executed on GPU per model before measurement. |
| `--save-predictions` | `False` | Whether to write output segmentation PNG masks to disk. |

## Runtime Empirical Results

### Component Latency Breakdown (RTX 4060 Laptop GPU, 224×224 input)

| Component | Median Latency | Notes |
|---|---|---|
| **Controller Decision** | ~0.051 ms | Pure CPU metadata evaluation. |
| **Preprocessing** | ~4.35 ms | PIL image load, resize 224×224, normalize. |
| **Host-to-Device Copy** | ~0.21 ms | CPU tensor to CUDA VRAM. |
| **Model Forward Pass** | ~13.01 – 14.13 ms | PyTorch backbone + linear segmentation head. |
| **Postprocessing** | ~0.0005 ms | Class argmax (without PNG disk write). |
| **Total End-to-End** | **~17.93 – 19.24 ms** | End-to-end synchronized latency. |

### Comparative Strategy Performance (50 Requests per Strategy, Seed 42)

#### Primary Experiment (`--metric p95`)

| Strategy | Actual Hit Rate | Predicted Feasibility | Mean Selected mIoU | Median Model ms | Median E2E ms |
|---|---|---|---|---|---|
| `always_full_attention` | 12.00% | 66.00% | 0.742990 | 13.48 ms | 28.66 ms |
| `always_fastest` | 60.00% | 66.00% | 0.742990 | 13.16 ms | 18.13 ms |
| `adaptive_controller` | 56.00% | 66.00% | 0.742990 | 13.01 ms | 17.93 ms |

#### Sensitivity Experiment (`--metric p99`)

| Strategy | Actual Hit Rate | Predicted Feasibility | Mean Selected mIoU | Median Model ms | Median E2E ms |
|---|---|---|---|---|---|
| `always_full_attention` | 58.00% | 40.00% | 0.742990 | 13.32 ms | 18.43 ms |
| `always_fastest` | 52.00% | 40.00% | 0.703581 | 14.15 ms | 19.36 ms |
| `adaptive_controller` | 58.00% | 40.00% | 0.719344 | 13.48 ms | 18.52 ms |

#### Sensitivity Experiment (`--metric max_observed`)

| Strategy | Actual Hit Rate | Predicted Feasibility | Mean Selected mIoU | Median Model ms | Median E2E ms |
|---|---|---|---|---|---|
| `always_full_attention` | 54.00% | 18.00% | 0.742990 | 13.27 ms | 18.42 ms |
| `always_fastest` | 56.00% | 30.00% | 0.703581 | 13.96 ms | 18.72 ms |
| `adaptive_controller` | 52.00% | 30.00% | 0.710675 | 14.13 ms | 19.24 ms |

## Scientific Insights and Differences from Phase 8 Offline Simulation

1. **The End-to-End Latency Tax**:
   - In Phase 8, deadlines were evaluated solely against device-resident forward latency bounds (where `p95` is ~16.6 ms).
   - In live runtime execution, preprocessing (~4.35 ms) and host transfer (~0.21 ms) add **~4.6 ms of overhead**. When pipelines do not pipelining preprocessing, requests with deadlines in $[16.6, 21.2]$ ms that were predicted as feasible forward passes experience actual end-to-end deadline misses.
2. **Dynamic Accuracy Retention in Tail Regimes**:
   - Under `p99` sensitivity, `adaptive_controller` matches `always_full_attention` in actual hit rate (**58.00%** vs **52.00%** for `always_fastest`) while maintaining **0.719344 mIoU** (significantly better than `always_fastest` at **0.703581**).
   - The adaptive controller opportunistically selects `full_attention` when deadlines permit, falling back to `moderate_dwconv` when budgets tighten.
3. **Execution Time Jitter & Operating System Noise**:
   - On Windows 11 with an RTX 4060 Laptop GPU, GPU kernel execution jitter, dynamic clock boost fluctuations, and driver overhead lead to variance between empirical benchmark quantiles and single-invocation runtimes.
   - This reinforces that statistical bounds from microbenchmarks must be budgeted with an engineering margin when building end-to-end real-time systems.


