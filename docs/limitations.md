# Technical Limitations and Scope Boundaries

This document defines the engineering boundaries, theoretical constraints, and operating limits of the RTS framework.

---

## 1. Soft Real-Time vs. Hard Real-Time Guarantees

This architecture is explicitly designed for **soft real-time systems** where deadline misses degrade service quality but do not cause catastrophic system failure.

- **Empirical Execution Bounds**: The admission controller and adaptive selector operate using empirical latency quantiles ($p95$, median) measured over hundreds of warmup iterations on a specific GPU model.
- **Absence of Formal WCET**: Modern commodity GPU microarchitectures feature dynamic frequency scaling, L2 cache contention, thread-block scheduling nondeterminism, and driver-level memory management. Consequently, measured $p95$ and $p99$ latencies do not constitute formal Worst-Case Execution Time (WCET) bounds.
- **Safety Criticality**: This software cannot be used in isolation for hard real-time safety actuation (such as emergency steering or braking) without a certified, formally bounded secondary safety supervisor.

---

## 2. Non-Preemptive GPU Scheduling Boundaries

Commercial NVIDIA GPUs and the PyTorch execution stack do not support fine-grained, low-overhead preemption of actively running compute kernels.
- **Dispatch Non-Preemption**: Once an inference kernel has been launched on the CUDA stream, it executes to completion.
- **Head-of-Line Blocking**: An urgent request with an immediate deadline arriving while an unpruned DeiT-Small model is running must wait until the current kernel completes (~3.7 ms). This bounds the minimum achievable response time for late-arriving requests.

---

## 3. EDF Domino Misses Under Overload

While Earliest Deadline First (EDF) is optimal for underloaded uniprocessor systems with independent preemption, **non-preemptive EDF degrades severely under sustained overload if admission control is omitted**:
- When $\sum e_i > \Delta t$, EDF repeatedly dispatches the most urgent request that is already on the verge of missing its deadline.
- By the time that request finishes, subsequent requests have also passed their deadlines, creating cascading "domino" deadline misses.
- **Ablation Finding**: In Phase 24B experiments, unconstrained `EDF_fixed_DeiT` completed only $1.0\%$ of requests on-time under severe overload (`overload_tight`), whereas enabling admission control (`EDF_AC_fixed_DeiT`) restored on-time completion to $37.0\%$, and adaptive pruning (`EDF_Adaptive_AC`) reached $46.0\%$.

---

## 4. Sensitivity of Simulation to Tail Latency Modeling

As documented in the Phase 24B audit:
- When the discrete-event simulator is parameterized with conservative $p95$ latency ($3.78\text{ ms}$ for DeiT-Small), it assumes every single task experiences tail latency. Under bursty traffic (`bursty_high_tight`), this caused the simulator to predict $27.0\%$ on-time completion.
- In reality, physical GPU runs execute most requests near median latency ($3.15\text{ ms}$), achieving $60.0\%$ on-time completion on physical hardware.
- When parameterized with median latency, the simulator predicted $67.0\%$, closely matching physical measurements.
- **Conclusion**: Discrete-event simulation must be interpreted alongside physical hardware trace replays, particularly for bursty arrival patterns.

---

## 5. Hardware and Architecture Transferability

- **GPU Dependency**: Model latency profiles are tightly coupled to the specific GPU microarchitecture (NVIDIA Ada Lovelace AD107, 8 GB VRAM). Deploying to Jetson Orin Nano, AGX Orin, or server GPUs (A100/H100) requires regenerating the empirical model profiles (`model_profiles.json`).
- **Static Resolution Requirement**: Captured CUDA Graphs require fixed tensor dimensions ($1 \times 3 \times 224 \times 224$). Dynamic batching or variable input resolutions require separate graph captures or fallback to stream-ordered eager execution.
