# Real-Time Scheduler Design: DA-ASAC and Accounting Invariants

## 1. System Model and Request Lifecycle

Each vision inference request $r_i$ is characterized by:
- $A_i$: Arrival timestamp (ms).
- $D_{\text{rel}, i}$: Relative deadline (ms).
- $D_{\text{abs}, i} = A_i + D_{\text{rel}, i}$: Absolute deadline (ms).
- $\mathbf{x}_i$: Image tensor payload ($3 \times 224 \times 224$).
- $y_i$: Ground-truth scene class.

A request progresses through the following states:
$$\text{ARRIVED} \longrightarrow \begin{cases} \text{REJECTED} & \text{(if infeasible at admission)} \\ \text{QUEUED} \longrightarrow \begin{cases} \text{EXPIRED} & \text{(if deadline expires in queue)} \\ \text{DISPATCHED} \longrightarrow \begin{cases} \text{ON\_TIME} & (C_i \le D_{\text{abs}, i}) \\ \text{LATE} & (C_i > D_{\text{abs}, i}) \end{cases} \end{cases} \end{cases}$$

---

## 2. Four Fundamental Request Conservation Laws

In any valid real-time discrete-event simulation, requests cannot be created, lost, or double-counted. All 70 configurations satisfy:

1. **Law of Offered Conservation**:
   $$N_{\text{offered}} = N_{\text{completed, on-time}} + N_{\text{completed, late}} + N_{\text{rejected}} + N_{\text{expired}}$$

2. **Law of Completion Conservation**:
   $$N_{\text{completed}} = N_{\text{completed, on-time}} + N_{\text{completed, late}}$$

3. **Law of Admitted Conservation**:
   $$N_{\text{admitted}} = N_{\text{completed}} + N_{\text{expired}}$$

4. **Law of Queue Conservation**:
   $$N_{\text{offered}} = N_{\text{admitted}} + N_{\text{rejected}}$$

---

## 3. The DA-ASAC Algorithm

The Deadline-Aware Adaptive Scheduling with Admission Control (DA-ASAC) policy operates in two stages:

### Stage 1: Queue-Tail Admission Control (at arrival $t = A_i$)
When request $r_i$ arrives:
1. Compute expected GPU availability time: $t_{\text{avail}} = \max(t, t_{\text{gpu\_busy\_until}})$.
2. Calculate accumulated workload in queue:
   $$W_{\text{queue}} = \sum_{q \in Q} L(m_q)$$
3. Verify feasibility under the fastest eligible candidate $m_{\text{fastest}}$:
   $$t_{\text{avail}} + W_{\text{queue}} + L(m_{\text{fastest}}) \le D_{\text{abs}, i}$$
4. If false, reject immediately ($r_i \to \text{REJECTED}$). If true, push to EDF priority queue ($r_i \to \text{QUEUED}$).

### Stage 2: Slack-Aware Model Selection (at dispatch $t = t_{\text{dispatch}}$)
When the GPU finishes execution and is ready for the next request $r^* = \operatorname{pop}(Q)$:
1. If $t \ge D_{\text{abs}}^*$, expire request ($r^* \to \text{EXPIRED}$).
2. Otherwise, for each eligible candidate model $m \in \mathcal{M}$:
   $$\text{Slack}(m) = D_{\text{abs}}^* - (t + L_m^{\text{p95}})$$
3. Find feasible candidates: $\mathcal{M}_{\text{feasible}} = \{m \in \mathcal{M} \mid \text{Slack}(m) \ge 0\}$.
4. If $\mathcal{M}_{\text{feasible}} \ne \emptyset$:
   Select $m^* = \arg\max_{m \in \mathcal{M}_{\text{feasible}}} \text{ValAccuracy}(m)$.
   Tie-break on largest slack, then model name.
5. If $\mathcal{M}_{\text{feasible}} = \emptyset$:
   Under firm deadline semantics, expire early to avoid wasting GPU compute ($r^* \to \text{EXPIRED}$).

---

## 4. The 10 Canonical Workload Scenarios

| Scenario Name | Pattern | Intensity ($\rho$) | Deadline Tightness | Deadline Range (ms) |
| :--- | :---: | :---: | :---: | :---: |
| `periodic_low_loose` | Periodic | $0.50$ | Loose | $[15.0, 25.0]$ |
| `periodic_mod_mod` | Periodic | $0.85$ | Moderate | $[6.0, 12.0]$ |
| `periodic_high_tight` | Periodic | $1.15$ | Tight | $[3.5, 6.0]$ |
| `periodic_low_tight` | Periodic | $0.50$ | Tight | $[3.5, 6.0]$ |
| `jittered_mod_mod` | Jittered | $0.85$ | Moderate | $[6.0, 12.0]$ |
| `jittered_high_tight` | Jittered | $1.15$ | Tight | $[3.5, 6.0]$ |
| `bursty_mod_mod` | Bursty | $0.85$ | Moderate | $[6.0, 12.0]$ |
| `bursty_high_tight` | Bursty | $1.15$ | Tight | $[3.5, 6.0]$ |
| `overload_mod_deadline` | Overload | $1.50$ | Moderate | $[6.0, 12.0]$ |
| `overload_tight_deadline`| Overload | $1.50$ | Tight | $[3.5, 6.0]$ |
