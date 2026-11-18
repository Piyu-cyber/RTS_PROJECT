# Experimental Methodology

## 1. Benchmarking Environment

All reported experiments were conducted on a dedicated physical edge workstation:
- **Operating System**: Windows 11 Home (Build 26100)
- **Processor**: Intel Core 13th Gen / AMD Ryzen x86_64 host
- **GPU Accelerator**: NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM)
- **NVIDIA Driver**: 591.66
- **CUDA Runtime Build**: 12.1
- **Deep Learning Framework**: PyTorch 2.5.1+cu121 (`.venv_evit`)
- **Python Runtime**: Python 3.11.9

---

## 2. Perception Datasets and Tasks

### Primary Operational Perception Task: BDD100K Scene Classification
The primary benchmark uses the Berkeley DeepDrive 100K (BDD100K) dataset mapped to 6 driving scene categories:
1. `city street` (43,267 train / 6,197 val / 12,382 test)
2. `highway` (16,211 train / 2,330 val / 4,642 test)
3. `residential` (8,061 train / 1,126 val / 2,308 test)
4. `parking lot` (1,234 train / 173 val / 344 test)
5. `gas stations` (207 train / 28 val / 53 test)
6. `tunnel` (654 train / 93 val / 173 test)

**Split Isolation**:
- Train: 69,634 valid annotated images (366 undefined excluded)
- Validation: 9,947 valid annotated images (53 undefined excluded)
- Held-out Test: 19,902 valid annotated images (98 undefined excluded)
- Total: 99,483 valid frames across 100,000 extracted 720p images.

### Reference Benchmark: Oxford-IIIT Pet
Prior development phases (Phase 23B / Phase 24) evaluated 37 fine-grained animal breeds to benchmark token pruning under high-class-count classification. Results from Oxford-IIIT Pet are documented in `outputs/phase24/` and are strictly separated from BDD100K driving scene results.

---

## 3. Real-Time Workload Generation

Workloads model image arrival patterns typical of autonomous driving and smart camera sensors:

| Workload Name | Pattern | Load Intensity ($\rho$) | Deadline Class | Inter-Arrival Distribution | Relative Deadline Range ($d_i$) |
|---|---|:---:|:---:|---|:---:|
| `periodic_low_loose` | Periodic | 0.50 | Loose | Constant ($2 \cdot e_{\text{ref}}$) | $[25, 40]$ ms |
| `periodic_mod_mod` | Periodic | 0.85 | Moderate | Constant ($1.18 \cdot e_{\text{ref}}$) | $[12, 20]$ ms |
| `periodic_high_tight` | Periodic | 1.15 | Tight | Constant ($0.87 \cdot e_{\text{ref}}$) | $[5, 10]$ ms |
| `periodic_low_tight` | Periodic | 0.50 | Tight | Constant ($2 \cdot e_{\text{ref}}$) | $[5, 10]$ ms |
| `jittered_mod_mod` | Jittered | 0.85 | Moderate | Gaussian jitter ($\sigma = 0.2 \cdot \mu$) | $[12, 20]$ ms |
| `jittered_high_tight` | Jittered | 1.15 | Tight | Gaussian jitter ($\sigma = 0.2 \cdot \mu$) | $[5, 10]$ ms |
| `bursty_mod_mod` | Bursty | 0.85 | Moderate | Bimodal batch arrivals | $[12, 20]$ ms |
| `bursty_high_tight` | Bursty | 1.15 | Tight | Bimodal batch arrivals | $[5, 10]$ ms |
| `overload_mod_deadline` | Overload | 1.50 | Moderate | Poisson arrival ($\lambda = 1.5$) | $[12, 20]$ ms |
| `overload_tight_deadline` | Overload | 1.50 | Tight | Poisson arrival ($\lambda = 1.5$) | $[5, 10]$ ms |

---

## 4. Evaluation Metrics and Conservation Identities

For any trace of $N_{\text{offered}}$ requests, the following identities must hold:
1. $N_{\text{offered}} = N_{\text{admitted}} + N_{\text{rejected}}$
2. $N_{\text{admitted}} = N_{\text{completed}} + N_{\text{expired}}$
3. $N_{\text{completed}} = N_{\text{on\_time}} + N_{\text{late}}$

### Core Metrics:
- **On-Time Completion Ratio**: $\text{OTR} = N_{\text{on\_time}} / N_{\text{offered}}$ (Primary real-time QoS metric).
- **Completed Deadline Miss Ratio**: $\text{DMR} = N_{\text{late}} / N_{\text{completed}}$.
- **Admission Rejection Ratio**: $\text{ARR} = N_{\text{rejected}} / N_{\text{offered}}$.
- **In-Queue Expiration Ratio**: $\text{EXP} = N_{\text{expired}} / N_{\text{offered}}$.
- **Mean Response Time**: $\bar{R} = \frac{1}{N_{\text{completed}}} \sum_{i} (c_i - r_i)$.
- **Mean Waiting Time**: $\bar{W} = \frac{1}{N_{\text{completed}}} \sum_{i} (s_i - r_i)$.

---

## 5. Strict Accuracy Provenance Discipline

To ensure reproducibility and prevent metric confusion:
1. **Validation Quality Gating ($\tau_{\text{qual}} \ge 45.0\%$ or $\ge 92.0\%$ for Pet)**: Used offline exclusively to eliminate non-converged checkpoints. Zero test data was used for policy tuning.
2. **Synthetic Expected Accuracy**: Computed as the model-weighted average of offline test accuracies over all completed requests:
   $$\text{Acc}_{\text{synth}} = \frac{1}{N_{\text{completed}}} \sum_{i \in \text{completed}} \text{Acc}_{\text{test}}(m_i)$$
3. **Empirical Hardware Replay Accuracy**: Evaluated on physical GPU inference runs passing actual JPEG images through captured CUDA Graphs.
