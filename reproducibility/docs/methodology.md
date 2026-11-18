# System Methodology: Cross-Layer Co-Design Rationale

## 1. The Autonomous Driving Real-Time Dilemma

Modern autonomous driving perception systems must satisfy two competing demands:
1. **High Semantic Fidelity**: Safety-critical classification across diverse and severe class imbalances (e.g., highway vs. tunnel vs. rare gas stations).
2. **Strict Temporal Guarantees**: Deadlines imposed by vehicle velocity (e.g., a vehicle moving at $30\text{ m/s}$ travels $0.3\text{ m}$ every $10\text{ ms}$).

Conventional systems address these demands in isolation:
- **Vision Model Pruning (Standalone)**: Compresses deep architectures statically (e.g., EViT with $k=0.5$ or $k=0.7$) to reduce computational latency (GMACs). However, uniform static compression discards fine-grained features necessary for rare tail classes, degrading perception quality (Macro-F1) when traffic or scene conditions are difficult.
- **Classic Real-Time Schedulers (Standalone)**: Schedule fixed monolithic models (e.g., DeiT-Small) using Earliest Deadline First (EDF) or First-In-First-Out (FIFO). Under bursty arrivals or heavy sensor loads ($\rho > 1.0$), fixed-model schedulers suffer catastrophic queue collapse, causing cascaded deadline misses for all subsequent requests.

---

## 2. The Cross-Layer Co-Design Solution

Our proposed system bridges perception training and runtime real-time systems through a co-designed architecture:

```
[ Incoming Sensor Frame (A_i, D_rel) ]
                  |
                  v
       [ Admission Controller ] ---> (Infeasible? Drop early to conserve GPU)
                  | (Admitted)
                  v
         [ EDF Priority Queue ]
                  |
                  v
     [ Deadline-Aware Adaptive ] ---> Selects highest-accuracy model
     [     Model Selector      ]      predicted to meet remaining slack
                  |
                  v
    [ Captured Static CUDA Graph ] ---> Executes on NVIDIA RTX 4060 GPU
                  |
                  v
         [ On-Time Prediction ]
```

### Layer 1: Perception Optimization (CMPR + CB-CE)
- **Class-Balanced Cross-Entropy (CB-CE)**: Offsets extreme class skew ($43,581$ city street samples vs. $27$ gas station samples) using smoothed inverse-frequency weights ($w_c \propto (N / (C \cdot n_c))^{0.25}$).
- **Correctness-Gated Margin Preservation Regularization (CMPR)**: Distills safety margin confidence from a full-capacity teacher ($m_t$) to pruned students ($m_s$), regularizing only on samples where the teacher prediction is correct ($g_i = 1[\arg\max(z_t) = y_i]$).

### Layer 2: Runtime Systems Co-Design (DA-ASAC)
- **Queue-Tail Admission Control (AC)**: Performs an $O(|Q|)$ feasibility check upon frame arrival. If existing queue backlog prevents meeting the frame's absolute deadline even under the fastest pruned model, the request is rejected immediately, saving precious GPU cycles.
- **Slack-Aware Model Selector**: At the exact instant of dispatch, examines remaining slack $S_i(t) = D_{\text{abs}} - (t + L_m^{\text{p95}})$. It dynamically selects the highest-accuracy model variant that remains deadline-feasible.
- **CUDA Graph Acceleration**: Eliminates CPU driver submission overhead, converting unpredictable eager PyTorch execution into microsecond-level deterministic execution.
