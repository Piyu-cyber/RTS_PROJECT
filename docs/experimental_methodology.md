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

**Split Isolation & Authoritative Counts**:
- Train: 69,634 valid annotated images (366 undefined excluded)
- Validation: 9,947 valid annotated images (53 undefined excluded from official 10,000 catalog)
- Held-out Test: 19,902 valid annotated images (98 undefined excluded)
- Total: 99,483 valid frames across 100,000 extracted 720p images.

### Reference Benchmark: Oxford-IIIT Pet
Fine-grained animal breed benchmark (37 classes) evaluated during baseline model development to benchmark token pruning under high-class-count classification. Documented in `outputs/phase24/` and strictly separated from BDD100K driving scene results.

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

## 5. Multi-Seed Model Training & Independent Forensic Audit

To rigorously validate whether token-pruned EViT models with Class-Balanced Margin-Preserving Regularization (CB-CMPR) provide reproducible improvements, three independent seeds (`seeds \in {42, 123, 456}`) were trained under identical protocols:

- **Matched Training Protocol**:
  - Backbone: EViT-Small ($k=0.7$, keep rate at layers 3, 6, 9 with token fusion)
  - Parameter Count: 21,668,022 (~21.67M)
  - Compute: 3.03 GMACs (vs. 4.60 GMACs for DeiT baseline)
  - Normalized Class Weights: $\mathbf{w}_{\text{norm}} = [0.3393, 0.4268, 0.5167, 1.1118, 2.1507, 1.4547]$
  - Optimization: AdamW, initial LR $5 \times 10^{-5}$, cosine annealing, batch size 64, 3 epochs (3,267 steps)
  - Checkpoint Selection: Highest validation Macro-F1 (zero test data leakage)
  - Teacher Model: Unpruned DeiT-Small fine-tuned on BDD100K (SHA-256 verified)

- **Audit Findings**:
  - Recomputing metrics directly from the raw test predictions and labels confirmed 100% numerical concordance across all confusion matrices and F1 scores.
  - Across all three seeds, `K07-CB-CMPR` delivered a strictly positive Top-1 accuracy gain ($+1.37\% \pm 0.33\%$, McNemar test $p < 10^{-15}$).
  - 5-Class Macro-F1 (excluding the 6-sample Gas Station test partition) improved by $+1.17\% \pm 0.50\%$ across 100% of seeds.
