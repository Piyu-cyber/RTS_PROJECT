# Benchmarking Protocol: Deterministic Latency, CUDA Graphs, and GPU Profiling

## 1. Soft Real-Time Perception Requirements

In cyber-physical autonomous driving, camera sensor frames arrive continuously at a fixed frame rate (e.g., $10\text{ to }30\text{ Hz}$).
Unlike hard real-time systems where a single deadline miss causes total system failure, autonomous perception operates under **soft/firm real-time constraints**:
- An inference completed before its deadline ($C_i \le D_i$) contributes full perceptual value to vehicle motion planning.
- An inference completing after its deadline ($C_i > D_i$) provides stale, obsolete scene information that can mislead collision avoidance; under firm semantics, executing late frames wastes GPU cycles.

---

## 2. Eager PyTorch vs. Static CUDA Graph Replay

### The Problem with Eager Execution
In standard PyTorch (`eager` mode):
- Every transformer layer incurs individual CPU-to-GPU kernel dispatch overheads (host driver latency).
- On small batch sizes (batch size 1 for single-stream camera pipelines), CPU overhead dominates total latency, causing high tail jitter (p99 latency up to $25\text{ ms}$).

### Deterministic CUDA Graph Capture
To achieve predictable real-time execution bounds:
1. **Warmup**: Run 15 to 50 iterations on an isolated CUDA stream to stabilize dynamic memory allocations and CUDA driver state.
2. **Graph Capture**: Enclose a forward pass in `torch.cuda.graph(self.graph)` to capture the full directed acyclic graph (DAG) of GPU kernels into hardware memory.
3. **Graph Replay**: Replay the captured graph via `self.graph.replay()` in a single kernel launch.

### Results on NVIDIA RTX 4060 Laptop GPU
| Model | Eager Mean | Eager p95 | CUDA Graph Mean | CUDA Graph p95 | Tail Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DeiT-Small | $13.11\text{ ms}$ | $15.20\text{ ms}$ | $3.25\text{ ms}$ | $\mathbf{3.68\text{ ms}}$ | $\mathbf{75.8\%}$ |
| EViT-K09 | $18.67\text{ ms}$ | $21.40\text{ ms}$ | $3.04\text{ ms}$ | $\mathbf{3.54\text{ ms}}$ | $\mathbf{83.5\%}$ |
| EViT-K07 | $25.27\text{ ms}$ | $28.10\text{ ms}$ | $2.91\text{ ms}$ | $\mathbf{3.49\text{ ms}}$ | $\mathbf{87.6\%}$ |
| EViT-K05 | $25.12\text{ ms}$ | $27.90\text{ ms}$ | $2.31\text{ ms}$ | $\mathbf{2.81\text{ ms}}$ | $\mathbf{89.9\%}$ |

---

## 3. Profiling Methodology

- **Timing Mechanism**: Hardware CUDA Events (`torch.cuda.Event(enable_timing=True)`).
- **Synchronization**: Explicit stream synchronization (`torch.cuda.synchronize()`) before and after measured iterations.
- **Statistical Percentiles**:
  - Minimum, Median (p50), Mean, p90, p95, p99, and Maximum observed latencies calculated over 200 measured iterations.
  - p95 latency is chosen as the conservative execution bound $L_m^{\text{p95}}$ for real-time feasibility checks.
- **Throughput (FPS)**: Measured using batches of 64 images over 30 steady-state iterations.
