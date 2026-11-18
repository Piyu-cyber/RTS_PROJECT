# Phase 24B Data Dictionary & Metrics Specification

**Project:** Deadline-Aware Adaptive Token Pruning for Real-Time Vision Transformers  
**Document:** Canonical Schema, Metrics Definitions, and Conservation Laws  
**Status:** Canonical & Frozen  
**Date:** October 2026  

---

## 1. Overview

This document specifies the authoritative data dictionary for all Phase 24B tabular outputs:
- [`outputs/phase24b/corrected_results.csv`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/corrected_results.csv)
- [`outputs/phase24b/per_trace_results.csv`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/per_trace_results.csv)
- [`outputs/phase24b/simulation_hardware_comparison.csv`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/simulation_hardware_comparison.csv)
- [`outputs/phase24b/claim_provenance.csv`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/claim_provenance.csv)

---

## 2. Fundamental Conservation Laws

Every record in the Phase 24B evaluation suite strictly satisfies three physical conservation identities:

$$\begin{aligned}
1. \quad & N_{\text{admitted}} + N_{\text{rejected}} = N_{\text{offered}} \\
2. \quad & N_{\text{completed}} + N_{\text{expired}} = N_{\text{admitted}} \\
3. \quad & N_{\text{completed\_on\_time}} + N_{\text{completed\_late}} = N_{\text{completed}}
\end{aligned}$$

No request is double-counted, orphaned, or omitted.

---

## 3. Schema & Column Specifications

| Column Name | Data Type | Units / Range | Description |
|---|---|---|---|
| `policy` | String | Categorical (7 policies) | Scheduling policy under evaluation. |
| `workload_name` | String | Identifier | Unique trace name including arrival pattern, load level, tightness, and seed. |
| `workload_category` | String | `periodic`, `jittered`, `bursty`, `overload` | Workload arrival process pattern. |
| `load_level` | String | `low`, `moderate`, `high`, `overload` | Traffic density relative to baseline nominal compute capacity. |
| `deadline_tightness` | String | `loose`, `moderate`, `tight` | Relative deadline distribution range ($[15, 25]$, $[6, 12]$, or $[3.5, 6.0]\text{ ms}$). |
| `seed` | Integer | $\ge 0$ | Explicit random seed used for request sequence generation. |
| `experiment_type` | String | `SIMULATOR_P95`, `SIMULATOR_MEDIAN`, `SIMULATOR_MEAN`, `GPU_REPLAY` | Execution backend and parameterization model. |
| `num_offered` | Integer | $\ge 1$ | Total requests offered to the system ($N_{\text{off}}$). |
| `num_admitted` | Integer | $[0, N_{\text{off}}]$ | Requests accepted into the priority queue by admission control. |
| `num_rejected` | Integer | $[0, N_{\text{off}}]$ | Requests rejected at arrival due to provably insufficient queue slack. |
| `num_expired` | Integer | $[0, N_{\text{adm}}]$ | Admitted requests that expired while queued before dispatch (firm-deadline drop). |
| `num_completed` | Integer | $[0, N_{\text{adm}}]$ | Requests dispatched and executed on the GPU. |
| `num_completed_on_time` | Integer | $[0, N_{\text{comp}}]$ | Executed requests whose completion timestamp $C_i \le D_i$. |
| `num_completed_late` | Integer | $[0, N_{\text{comp}}]$ | Executed requests whose completion timestamp $C_i > D_i$ (wasted compute). |
| `on_time_ratio_offered` | Float | $[0.0, 1.0]$ | **Primary Goodput Metric:** $N_{\text{completed\_on\_time}} / N_{\text{offered}}$. |
| `on_time_ratio_accepted` | Float | $[0.0, 1.0]$ | **Admission Fidelity:** $N_{\text{completed\_on\_time}} / N_{\text{admitted}}$. |
| `deadline_miss_ratio_completed` | Float | $[0.0, 1.0]$ | **Wasted Compute Metric:** $N_{\text{completed\_late}} / N_{\text{completed}}$. |
| `rejection_ratio` | Float | $[0.0, 1.0]$ | Ratio of requests rejected at arrival ($N_{\text{rejected}} / N_{\text{offered}}$). |
| `expiration_ratio` | Float | $[0.0, 1.0]$ | Ratio of requests expired in queue ($N_{\text{expired}} / N_{\text{offered}}$). |
| `mean_response_time_ms` | Float | Milliseconds ($\text{ms}$) | Average elapsed time from arrival to completion ($C_i - r_i$). |
| `median_response_time_ms`| Float | Milliseconds ($\text{ms}$) | 50th percentile of response time over completed requests. |
| `p95_response_time_ms` | Float | Milliseconds ($\text{ms}$) | 95th percentile of response time over completed requests. |
| `p99_response_time_ms` | Float | Milliseconds ($\text{ms}$) | 99th percentile of response time over completed requests. |
| `mean_waiting_time_ms` | Float | Milliseconds ($\text{ms}$) | Average queuing delay from arrival to GPU dispatch ($s_i - r_i$). |
| `median_waiting_time_ms`| Float | Milliseconds ($\text{ms}$) | Median queuing delay over completed requests. |
| `mean_queue_length` | Float | Tasks | Time-weighted average number of requests waiting in queue. |
| `max_queue_length` | Integer | Tasks | Maximum observed backlog in queue. |
| `total_gpu_busy_time_ms` | Float | Milliseconds ($\text{ms}$) | Sum of active GPU execution durations across completed requests. |
| `total_makespan_ms` | Float | Milliseconds ($\text{ms}$) | Total elapsed wall-clock duration from start to final event. |
| `gpu_utilization_pct` | Float | $[0.0, 100.0] \%$ | Ratio of active GPU execution time to total makespan. |
| `mean_val_accuracy` | Float | $[0.0, 100.0] \%$ | **Validation Metric:** Average validation accuracy of dispatched models. |
| `synthetic_expected_test_accuracy` | Float | $[0.0, 100.0] \%$ | **Expected Dataset Accuracy:** Model-weighted test accuracy over completed requests. |
| `empirical_test_accuracy` | Float / String | $[0.0, 100.0] \%$ or `N/A` | **Empirical Evaluation:** Top-1 classification accuracy on real held-out test images (GPU replay only). |
| `count_deit_small` | Integer | $\ge 0$ | Number of requests executed with `deit_small_baseline` ($\rho=1.0$). |
| `count_evit_k09` | Integer | $\ge 0$ | Number of requests executed with `evit_small_k09_fuse` ($\rho=0.9$). |
| `count_evit_k07` | Integer | $\ge 0$ | Number of requests executed with `evit_small_k07_fuse` ($\rho=0.7$). |
| `count_evit_k05` | Integer | $\ge 0$ | Number of requests executed with `evit_small_k05_fuse` ($\rho=0.5$). |

---

## 4. Policy Name Taxonomy

1. **`FIFO_fixed_DeiT`**: Traditional First-In First-Out queue dispatching unpruned DeiT-Small. No admission control, no adaptation.
2. **`EDF_fixed_DeiT`**: Earliest-Deadline-First priority queue dispatching unpruned DeiT-Small. No admission control, no adaptation.
3. **`FIFO_fixed_EViT`**: FIFO queue dispatching fixed static pruned EViT ($\rho=0.7$, keep rate 0.7).
4. **`EDF_fixed_EViT`**: EDF queue dispatching fixed static pruned EViT ($\rho=0.7$).
5. **`EDF_AC_fixed_DeiT`**: EDF queue with queue-aware admission control dispatching unpruned DeiT-Small. Sheds infeasible requests at arrival.
6. **`EDF_Adaptive`**: EDF queue with dynamic slack-based model selection ($\rho \in \{1.0, 0.9, 0.7, 0.5\}$). Sheds expired tasks at dispatch (firm deadlines).
7. **`EDF_Adaptive_AC` (Proposed)**: Integrated architecture combining EDF priority queueing, queue-aware admission control, and dynamic slack-based token pruning.
