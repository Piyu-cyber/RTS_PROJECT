# System Architecture

## 1. Overview and Core Philosophy

In soft real-time vision inference, processing an image correctly but delivering the result past its deadline renders the prediction degraded or useless. Conventional inference engines run a fixed neural network model (e.g., full DeiT-Small) in first-come, first-served order. When incoming request arrival rates exceed the GPU processing capacity, requests accumulate in the software queue. Under high queueing delay, subsequent requests miss their deadlines even if the GPU execution itself takes only a few milliseconds.

This system addresses queue-induced deadline collapse through a three-stage real-time architecture:
1. **Queue-Aware Feasibility Gating (Admission Control)**: Drops infeasible requests upon arrival to prevent useless work from saturating GPU pipelines.
2. **Dynamic Priority Queueing (Earliest Deadline First)**: Dispatches requests in order of absolute deadline urgency rather than arrival time.
3. **Slack-Guided Model Adaptation (Adaptive EViT Selection)**: Evaluates remaining deadline slack at the moment of GPU dispatch and dynamically selects the highest-accuracy Vision Transformer variant that can complete before the deadline.

---

## 2. Request Lifecycle State Machine

Each incoming inference request moves through a formal lifecycle state machine:

```mermaid
stateDiagram-v2
    [*] --> ARRIVED: Request arrives (r_i, d_i)
    ARRIVED --> REJECTED: Admission check fails (predicted completion > D_i)
    ARRIVED --> QUEUED: Admission check passes (or AC disabled)
    QUEUED --> EXPIRED: Clock exceeds D_i while waiting in queue
    QUEUED --> DISPATCHED: GPU idle & request popped from queue
    DISPATCHED --> ON_TIME: GPU completes with c_i <= D_i
    DISPATCHED --> LATE: GPU completes with c_i > D_i
    REJECTED --> [*]
    EXPIRED --> [*]
    ON_TIME --> [*]
    LATE --> [*]
```

### Formal States:
- **`ARRIVED`**: Request has been initialized with arrival timestamp $r_i$ and relative deadline $d_i$. Absolute deadline is $D_i = r_i + d_i$.
- **`QUEUED`**: Request is accepted and enqueued into the priority queue (FIFO or EDF).
- **`DISPATCHED`**: Request is dequeued by the scheduler, bound to a specific Vision Transformer configuration, and launched on the GPU stream.
- **`TERMINATED`**: Request has finished its lifecycle. Exactly one terminal outcome is assigned.

### Terminal Outcomes:
- **`ON_TIME`**: Dispatched and completed with completion timestamp $c_i \le D_i$.
- **`LATE`**: Dispatched and completed with $c_i > D_i$.
- **`EXPIRED`**: Dropped from the queue before dispatch because $t_{\text{now}} \ge D_i$ (or insufficient slack for even the fastest model under firm semantics).
- **`REJECTED`**: Dropped at arrival by admission control because estimated queuing and execution latency exceeds $D_i$.

---

## 3. End-to-End Inference Pipeline

The architectural pipeline connects request generation, scheduling, neural network execution, and outcome accounting:

```mermaid
flowchart TD
    ImgIn["Image Input Stream\n(Real JPEG / Synthetic Tensor)"] --> ReqGen["Request Ingestion\nAssigns (r_i, d_i, D_i)"]
    ReqGen --> AC{"Admission Control\nFeasibility Test"}
    
    AC -- "Infeasible (Slack < 0)" --> RejOut["REJECTED Outcome\n(Saves GPU Compute)"]
    AC -- "Feasible" --> Enqueue["Priority Queue\n(FIFO or EDF Heap)"]
    
    Enqueue --> SchedWait["Queue Waiting Phase\nMonitors Expiration"]
    SchedWait -- "Clock >= D_i" --> ExpOut["EXPIRED Outcome\n(Queue Timeout)"]
    
    SchedWait --> DispCheck{"GPU Idle?"}
    DispCheck -- "Yes" --> AdaptCtrl["Adaptive Controller\nCalculates Slack: D_i - (t_now + Latency)"]
    
    AdaptCtrl --> ModelSel{"Candidate Feasibility\n(Validation Quality Gated)"}
    ModelSel -- "Feasible candidates exist" --> PickMax["Select Highest Quality\nFeasible Candidate\n(DeiT, k=0.9, 0.7, 0.5)"]
    ModelSel -- "No candidate feasible & Firm" --> ExpOut
    ModelSel -- "No candidate feasible & Soft" --> PickFast["Fallback to Fastest (k=0.5)"]
    
    PickMax --> GPURun["GPU Execution\nCUDA Graph Replay\n(Zero CPU Launch Overhead)"]
    PickFast --> GPURun
    
    GPURun --> Sync["CUDA Stream Completion\nTimestamp c_i Measured"]
    Sync --> Eval{"c_i <= D_i ?"}
    Eval -- "Yes" --> OntimeOut["ON_TIME Outcome"]
    Eval -- "No" --> LateOut["LATE Outcome"]
```

---

## 4. CUDA Graph Acceleration Architecture

PyTorch kernel launch overhead in eager execution mode introduces between 8 ms and 15 ms of host-to-device synchronization and kernel scheduling jitter per inference call. In real-time workloads where deadlines range from 5 ms to 20 ms, eager launch overhead consumes up to 80% of the entire timing budget.

To eliminate host launch jitter, the system encapsulates each Vision Transformer configuration inside a static `CUDAGraphRunner`:
- **Static Memory Buffers**: Pre-allocates fixed input tensor (`1 x 3 x 224 x 224`) and output logit buffer (`1 x C`) in GPU device memory.
- **Warmup Execution**: Warms up model weights and PyTorch runtime on dedicated CUDA streams for 15 iterations.
- **Graph Capture**: Captures the entire forward-pass kernel sequence into a static hardware graph.
- **Replay Execution**: At runtime, image data is copied into the pre-allocated static buffer and `graph.replay()` is triggered.

This reduces median inference latency on the NVIDIA RTX 4060 GPU from ~13.1 ms (eager) to **3.15 ms** (CUDA Graph) for DeiT-Small, and down to **2.38 ms** for EViT $k=0.5$, providing predictable execution profiles required for real-time scheduling.
