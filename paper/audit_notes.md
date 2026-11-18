# Evidence Audit Notes: Phase 9.1 to Phase 9.3

This document records the exact numerical provenance, trace metrics, recomputations, and scientific limitations verified across the project for manuscript preparation.

---

## 1. Verified Core Metrics

### Model Architecture and Parameters
- **Backbone**: Meta Official DINOv2 ViT-S/14 (`dinov2_vits14_pretrain.pth`, SHA-256 `B938BF1BC15CD2EC0FEACFE3A1BB553FE8EA9CA46A7E1D8D00217F29AEF60CD9`).
- **Segmentation Head**: 2-class Conv2D head (`layers.0`: Conv2d(384, 128, 3, pad 1), `layers.1`: GELU, `layers.2`: Conv2d(128, 2, 1)).
- **Parameter Counts**:
  - `full_attention`: 22,499,330 total parameters.
  - `moderate_dwconv`: 21,332,018 total parameters (-5.2%).
  - `aggressive_dwconv`: 20,164,706 total parameters (-10.4%).
- **GPU VRAM Allocation**:
  - PyTorch Allocated Memory for all 3 models combined in `RuntimeModelRegistry`: **246.5 MiB**.
  - PyTorch Reserved Memory: **276.0 MiB**.

### Fine-Tuning Performance (Validation mIoU on Oxford-IIIT Pet)
- `full_attention`: **0.742990** (Phase 4 baseline best checkpoint)
- `moderate_dwconv`: **0.703581** (Blocks [2, 5, 8, 11] fine-tuned 10 epochs)
- `aggressive_dwconv`: **0.456155** (Blocks [1, 2, 3, 4, 5, 6, 7, 8] fine-tuned 10 epochs)

### Phase 9.1 Multi-Round Fair Evaluation (5 Rounds, 250 Requests per Strategy per Metric)
- **Primary p95 Metric**:
  - `always_full_attention`: End-to-End Hit Rate = **65.2%** [95% CI: 59.1%, 70.8%], Model-Only Hit Rate = **88.4%**, Median E2E = 17.51 ms.
  - `always_fastest`: End-to-End Hit Rate = **57.2%** [95% CI: 51.0%, 63.2%], Model-Only Hit Rate = **86.8%**, Median E2E = 18.11 ms.
  - `adaptive_controller`: End-to-End Hit Rate = **59.2%** [95% CI: 53.0%, 65.1%], Model-Only Hit Rate = **86.8%**, Median E2E = 18.23 ms.
- **Sensitivity p99 Metric**:
  - `always_full_attention`: End-to-End Hit Rate = **62.4%** [95% CI: 56.3%, 68.2%], Mean mIoU = 0.742990.
  - `always_fastest`: End-to-End Hit Rate = **53.2%** [95% CI: 47.0%, 59.3%], Mean mIoU = 0.703581.
  - `adaptive_controller`: End-to-End Hit Rate = **51.6%** [95% CI: 45.4%, 57.7%], Mean mIoU = 0.717610.
- **Sensitivity max_observed Metric**:
  - `always_full_attention`: End-to-End Hit Rate = **59.2%** [95% CI: 53.0%, 65.1%], Mean mIoU = 0.742990.
  - `always_fastest`: End-to-End Hit Rate = **54.0%** [95% CI: 47.8%, 60.1%], Mean mIoU = 0.703581.
  - `adaptive_controller`: End-to-End Hit Rate = **52.8%** [95% CI: 46.6%, 58.9%], Mean mIoU = 0.709413.

### Phase 9.2 Paired McNemar Tests (Recomputed from raw_traces.csv)
- **p95: Adaptive vs Full Attention**:
  - Discordant pairs: $b = 4$ (Adaptive hit, Full miss), $c = 19$ (Adaptive miss, Full hit).
  - Continuity-corrected $\chi^2_{\text{cc}} = 8.5217$, $p = 0.003509$.
  - Exact binomial $p = 0.002599$ (Statistically significant).
- **p99: Adaptive vs Full Attention**:
  - Discordant pairs: $b = 4$, $c = 31$.
  - Continuity-corrected $\chi^2_{\text{cc}} = 19.3143$, $p = 0.000011$.
  - Exact binomial $p = 0.000003$ (Statistically significant).
- **max_observed: Adaptive vs Full Attention**:
  - Discordant pairs: $b = 6$, $c = 22$.
  - Continuity-corrected $\chi^2_{\text{cc}} = 8.0357$, $p = 0.004586$.
  - Exact binomial $p = 0.003719$ (Statistically significant).

### Steady-State GPU Forward Pass Latency (100 Iterations Interleaved)
- `full_attention`: Median = **14.65 ms**, Mean = 14.20 ms, p95 = 17.00 ms.
- `moderate_dwconv`: Median = **15.39 ms**, Mean = 15.28 ms, p95 = 18.68 ms (+5.0% slower).
- `aggressive_dwconv`: Median = **15.84 ms**, Mean = 16.30 ms, p95 = 20.79 ms (+8.1% slower).

---

## 2. Documented Methodological Disclosures

1. **Selection of Replaced Blocks**:
   - Replaced attention blocks (`[2, 5, 8, 11]` for moderate and `[1, 2, 3, 4, 5, 6, 7, 8]` for aggressive) were selected manually/structurally, rather than using the data-driven layer-importance criterion $\Sigma_b$ from Scribano et al. This distinction is explicitly documented in the manuscript.
2. **Empirical Statistics vs. Worst-Case Guarantees**:
   - The controller's feasibility bounds ($p95, p99, \max$) are empirical statistics from 100 benchmark iterations, not formal worst-case execution time (WCET) bounds.
3. **Hardware-Level Causality vs. Hypotheses**:
   - PyTorch profiler confirms that `aten::repeat_interleave` consumed **57.7 ms** in aggressive DWConv due to dynamic parameter broadcasting for `[CLS]` tokens.
   - The reduction in Tensor Core matrix multiplication acceleration is an architectural hypothesis supported by kernel profiles, though direct hardware-counter proof (e.g., from NVIDIA Nsight Compute) was not collected.
4. **Resolution of the Markdown Reporting Transcription Error**:
   - The Phase 9.2 markdown table listed discordant counts `(12, 27)` with $\chi^2 = 8.3077$, while the true raw trace contains $b=4, c=19$ with $\chi^2 = 8.5217$. The raw JSON analysis (`paired_statistics.json`) was mathematically correct; the manuscript presents the audited true counts ($b=4, c=19$).
