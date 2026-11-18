# Implementation Guide

This document details the module design, classes, interfaces, and interactions across the codebase.

---

## 1. Request Representation and Lifecycle (`src/phase24/request.py`)

### Module Responsibility
Defines the atomic data structure for inference tasks and tracks timing timestamps, dispatch decisions, and terminal outcomes.

### Core Types
- **`RequestOutcome` (Enum)**: `ON_TIME`, `LATE`, `EXPIRED`, `REJECTED`.
- **`RequestState` (Enum)**: `ARRIVED`, `QUEUED`, `DISPATCHED`, `TERMINATED`.
- **`Request` (Dataclass)**:
  - `request_id: str`: Unique identifier.
  - `arrival_time_ms: float`: Arrival clock time $r_i$.
  - `relative_deadline_ms: float`: Relative deadline $d_i$.
  - `absolute_deadline_ms: float`: Calculated in `__post_init__` as $D_i = r_i + d_i$.
  - `selected_model: Optional[str]`: Identifier of the selected model.
  - `estimated_exec_time_ms: Optional[float]`: Estimated latency budget used during dispatch.
  - `actual_exec_time_ms: Optional[float]`: Measured GPU hardware execution time.
  - `dispatch_time_ms: Optional[float]`: Timestamp $s_i$ when request left queue and entered GPU.
  - `completion_time_ms: Optional[float]`: Timestamp $c_i$ when inference finished.

### Derived Metrics (Properties)
- `waiting_time_ms = dispatch_time_ms - arrival_time_ms`
- `response_time_ms = completion_time_ms - arrival_time_ms`
- `lateness_ms = completion_time_ms - absolute_deadline_ms`
- `remaining_slack_ms = relative_deadline_ms - estimated_exec_time_ms`

---

## 2. Priority Queueing Engine (`src/phase24/priority_queue.py`)

### Module Responsibility
Provides min-heap backed priority queues with deterministic secondary and tertiary tie-breaking to prevent nondeterministic dispatch order.

### Implementations
1. **`FIFOQueue`**:
   - Heap key: `(arrival_time_ms, insertion_counter, request_id, request)`.
   - Preserves strict arrival order. If two requests arrive simultaneously, `insertion_counter` enforces stable ordering.
2. **`EDFQueue`**:
   - Heap key: `(absolute_deadline_ms, arrival_time_ms, insertion_counter, request_id, request)`.
   - Prioritizes requests with smaller absolute deadlines ($D_i$).
   - Tie-breaks first on earlier arrival ($r_i$), then on insertion order.

---

## 3. Admission Control (`src/phase24/admission_control.py`)

### Module Responsibility
Assesses whether an arriving request can be completed on-time without violating deadlines, given currently pending work in the queue and ongoing GPU execution.

### Feasibility Test
For candidate request $i$ arriving at current time $t$, with ongoing GPU busy time extending to $t_{\text{busy}}$:
$$\text{Start Time } s_i = \max(t, t_{\text{busy}})$$
$$\text{Accumulated Queue Latency } W_Q = \sum_{q \in \mathcal{Q}} \hat{e}(q)$$
$$\text{Predicted Finish Time } c_i = s_i + W_Q + \hat{e}_i$$
The request is accepted if and only if:
$$c_i \le D_i + \epsilon \quad (\epsilon = 10^{-6}\text{ ms})$$

In adaptive mode, $\hat{e}_i$ uses the minimum execution time across quality-eligible models ($\min_{m} \hat{e}(m)$), allowing requests that can succeed under token pruning to enter the queue.

---

## 4. Adaptive Model Controller (`src/phase24/adaptive_controller.py`)

### Module Responsibility
Selects the model configuration at dispatch time by evaluating dynamic remaining slack against validated model profiles.

### Candidate Sorting and Quality Gating
1. Candidate profiles are filtered by validation accuracy:
   $$\text{eligible} = \{ m \in \mathcal{M} \mid \text{Acc}_{\text{val}}(m) \ge \tau_{\text{qual}} \}$$
2. Eligible candidates are ordered by validation accuracy descending, tie-breaking by lowest execution time.
3. At dispatch time $t_{\text{disp}}$, for each eligible model $m$:
   $$\text{Predicted Completion } \hat{c}(m) = t_{\text{disp}} + \hat{e}(m)$$
   $$\text{Slack } S_i(m) = D_i - \hat{c}(m)$$
4. The controller selects candidate $m^*$ with highest accuracy where $S_i(m^*) \ge 0$.
5. If no candidate has non-negative slack:
   - Under `firm_deadlines=True`: Request is expired immediately to preserve GPU time.
   - Under `firm_deadlines=False`: Controller dispatches the fastest model to minimize lateness.

---

## 5. Model Profiles (`src/phase24/model_profiles.py`)

### Module Responsibility
Loads validated timing and accuracy distributions measured empirically on target hardware (NVIDIA RTX 4060).

### Profile Schema
- `config_id`: e.g., `deit_small_baseline`, `evit_small_k09_fuse`, `evit_small_k07_fuse`, `evit_small_k05_fuse`.
- `keep_rate`: Token retention ratio (1.0, 0.9, 0.7, 0.5).
- `parameters_m`: Parameter count (21.67M for 6-class BDD100K head; 22.05M for 37-class Pet head).
- `macs_g`: Computational complexity in GMACs (4.60, 4.03, 3.03, 2.31).
- `cuda_graph_latency_ms`: Dictionary containing `mean`, `median`, `p95`, `p99`, `min`, `max`, `std`.
- `eager_latency_ms`: Unaccelerated eager execution latency distribution.

---

## 6. Discrete-Event Simulator (`src/phase24/discrete_event_simulator.py`)

### Module Responsibility
Simulates non-preemptive GPU task execution under exact discrete-event timing using a min-heap event queue.

### Event Loop
- Handles `ARRIVAL` and `COMPLETION` events.
- Enforces arrival batching: concurrent arrivals at identical timestamps are all enqueued before dispatching to allow priority ordering over arrival batches.
- Invariant assertions guarantee strict compute accounting:
  - $N_{\text{offered}} = N_{\text{admitted}} + N_{\text{rejected}}$
  - $N_{\text{admitted}} = N_{\text{completed}} + N_{\text{expired}}$
  - $N_{\text{completed}} = N_{\text{on\_time}} + N_{\text{late}}$

---

## 7. GPU Trace Replayer (`src/phase24/gpu_trace_replay.py`)

### Module Responsibility
Replays request traces on physical NVIDIA RTX 4060 hardware using real images from the BDD100K driving dataset.

### Pipeline
- Measures actual execution time via `torch.cuda.Event(enable_timing=True)`.
- Synchronizes streams via `torch.cuda.synchronize()`.
- Records empirical top-1 prediction accuracy and confusion matrices alongside physical millisecond timestamps.
