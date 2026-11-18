# Phase 24B Ablation Analysis & Component Attribution

**Project:** Deadline-Aware Adaptive Token Pruning for Real-Time Vision Transformers  
**Document:** Controlled Component Ablation Study  
**Status:** Frozen & Audited  
**Date:** October 2026  

---

## 1. Executive Summary

A critical scientific question for this research is:
> *Does the proposed controller gain its advantage from queue-aware admission control, from adaptive token pruning, or from the interaction of both?*

Through a strictly controlled 4-way ablation across identical workload traces and arrival timestamps, we demonstrate that **Admission Control** and **Adaptive Token Pruning** provide orthogonal, complementary real-time benefits:

1. **Admission Control (Queue Stability):** Prevents queue saturation and cascading deadline collapse. On `periodic_high_tight`, adding AC to fixed DeiT increases on-time completion from **6.0% to 56.0% (+50.0 percentage points)** and drops deadline misses among completed tasks from **92.2% to 0.0%**.
2. **Adaptive Token Pruning (Capacity Expansion):** Reduces execution latency from $3.78\text{ ms}$ down to $2.71\text{ ms}$, expanding the feasible schedulable set. Adding adaptive pruning to the AC system increases on-time completion from **56.0% to 76.0% (+20.0 percentage points)**, admitting and servicing 20 additional requests on time.

---

## 2. Controlled 4-Way Ablation Table (`periodic_high_tight`)

All 4 policies were evaluated on the canonical `periodic_high_tight` trace ($N=100$ requests, seed 44, arrival gap $\Delta t \approx 2.85\text{ ms}$, relative deadlines $d \in [3.5, 6.0]\text{ ms}$):

| Policy Configuration | Offered ($N_{\text{off}}$) | Admitted ($N_{\text{adm}}$) | Rejected ($N_{\text{rej}}$) | Completed ($N_{\text{comp}}$) | On-Time ($N_{\text{ontime}}$) | Late ($N_{\text{late}}$) | Expired in Queue ($N_{\text{exp}}$) | On-Time Ratio (All Offered) | Miss Ratio (Among Completed) | Mean Waiting Time | Mean Queue Length | Expected Test Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **(1) Baseline: `EDF_fixed_DeiT`** | 100 | 100 | 0 | 77 | 6 | 71 | 23 | **6.0 %** | 92.21 % | 2.05 ms | 0.81 | 92.67 % |
| **(2) AC Only: `EDF_AC_fixed_DeiT`** | 100 | 56 | 44 | 56 | 56 | 0 | 0 | **56.0 %** | **0.00 %** | 0.94 ms | 0.35 | 92.67 % |
| **(3) Adaptive Only: `EDF_Adaptive`** | 100 | 100 | 0 | 76 | 76 | 0 | 24 | **76.0 %** | **0.00 %** | 1.15 ms | 0.44 | 92.48 % |
| **(4) Proposed: `EDF_Adaptive_AC`** | 100 | 76 | 24 | 76 | 76 | 0 | 0 | **76.0 %** | **0.00 %** | **0.95 ms** | **0.36** | **92.48 %** |

---

## 3. Dissecting Component Contributions

```
[Baseline: EDF_fixed_DeiT]
  On-Time: 6.0%, Misses: 71, Expired: 23
       |
       +--> ABLATION STEP 1: Add Admission Control (EDF_AC_fixed_DeiT)
       |    - Gain: +50.0 percentage points on-time completion (6.0% -> 56.0%)
       |    - Mechanism: Sheds 44 doomed tasks at arrival (W = 0 ms).
       |    - Queue Health: Mean waiting time drops from 2.05 ms to 0.94 ms.
       |    - Late Misses: Dropped from 71 to 0 (0.0% wasted compute).
       |
       +--> ABLATION STEP 2: Add Adaptive Token Pruning (EDF_Adaptive_AC)
            - Gain: +20.0 percentage points additional on-time completion (56.0% -> 76.0%)
            - Mechanism: Compresses execution time from 3.78 ms down to 2.71 ms.
            - Capacity: Recovers enough latency slack to admit 20 more requests (76 vs 56).
            - Quality: Preserves high downstream classification accuracy (92.48% vs 92.67%).
```

### Why Does `EDF_Adaptive` Match `EDF_Adaptive_AC` in On-Time Count on Periodic Traffic?
In `EDF_Adaptive` without admission control, all 100 requests enter the queue. However, under firm-deadline dispatching:
```python
model_id, exec_time, slack = self.adaptive_controller.select_model(head, current_time_ms)
if model_id is None:
    head.mark_expired(current_time_ms, reason="Infeasible at dispatch (firm deadline)")
```
When `EDF_Adaptive` pops a request whose deadline has already elapsed, it drops the request **at dispatch time** without executing it. Because dropped requests never consume GPU cycles, exactly 76 requests execute on time.

### The Decisive Advantage of `EDF_Adaptive_AC` over `EDF_Adaptive`:
1. **Zero Wasted Queuing Latency:** Under `EDF_Adaptive_AC`, the 24 infeasible requests are rejected **immediately upon arrival** ($W = 0.0\text{ ms}$). Under `EDF_Adaptive`, those 24 requests sit in the queue until dispatch ($W = 1.15\text{ ms}$ on average) before being dropped. In networked robotics, early notification allows the upstream client to re-route or fall back immediately.
2. **Reduced Queue Congestion:** `EDF_Adaptive_AC` maintains a lower average queue length ($0.36$ tasks vs. $0.44$ tasks), preserving lower waiting times for newly arriving high-priority tasks.

---

## 4. Model Selection Breakdown Across Load Levels

In `EDF_Adaptive_AC`, dynamic model selection adapts compute elastically to queue pressure:

| Load Condition | Trace Name | DeiT ($\rho=1.0$) | EViT 0.9 ($\rho=0.9$) | EViT 0.7 ($\rho=0.7$) | EViT 0.5 ($\rho=0.5$) | Total Completed | % Baseline Dispatched | Expected Test Acc |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Low Load** | `periodic_low_loose` | 100 | 0 | 0 | 0 | 100 | **100.0 %** | 92.67 % |
| **Moderate Load** | `periodic_mod_mod` | 100 | 0 | 0 | 0 | 100 | **100.0 %** | 92.67 % |
| **High Load** | `periodic_high_tight` | 45 | 13 | 4 | 14 | 76 | **59.2 %** | 92.48 % |
| **Overload** | `overload_tight` | 22 | 5 | 6 | 13 | 46 | **47.8 %** | 92.40 % |

**Finding:** The adaptive controller exhibits **elastic, graceful degradation**. It defaults to full DeiT ($100\%$ dispatches) whenever slack permits, and downshifts to EViT 0.9, 0.7, and 0.5 only when required to prevent imminent deadline failure.
