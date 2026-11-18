# Phase 24B Manuscript Claims & Publication Evidence Registry

**Project:** Deadline-Aware Adaptive Token Pruning for Real-Time Vision Transformers  
**Document:** Manuscript Claims Classification and Evidence Audit  
**Status:** Canonical & Audited  
**Date:** October 2026  

---

## 1. Classification Taxonomy

To ensure scientific rigor and publication integrity, every claim proposed for the research manuscript is formally classified into one of six evidentiary categories:

1. **`SUPPORTED_BY_PUBLISHED_LITERATURE`**: Established theory or reference benchmarks published in prior peer-reviewed research.
2. **`MEASURED_LOCALLY`**: Raw physical hardware measurements collected on the local NVIDIA GeForce RTX 4060 Laptop GPU.
3. **`DERIVED_FROM_SIMULATION`**: Quantitative outputs produced by the deterministic discrete-event simulator using empirical profile inputs.
4. **`EMPIRICALLY_EVALUATED_ON_LOCAL_DATASET`**: Machine learning accuracy and inference metrics evaluated on the fine-tuned Oxford-IIIT Pet benchmark.
5. **`INTERPRETATION_OR_HYPOTHESIS`**: Mechanistic explanations and theoretical insights derived from controlled comparisons.
6. **`UNSUPPORTED_AND_EXCLUDED`**: Prior claims contradicted by empirical audit that must be retracted from the manuscript.

---

## 2. Manuscript Claims Registry

### Claim 1: Headline Real-Time On-Time Completion Rate
- **Claim Statement:** Under high-load, tight-deadline traffic (`periodic_high_tight`), the proposed controller (`EDF_Adaptive_AC`) achieves **76.0% on-time completion** ($77.8 \pm 0.7\%$ across 5 seeds), compared to **6.0%** ($4.0 \pm 0.6\%$) for standard fixed-model baseline systems.
- **Classification:** **`DERIVED_FROM_SIMULATION`** (and verified via GPU trace replay).
- **Exact Evidence Artifacts:**
  - Simulation records: [`outputs/phase24b/corrected_results.csv`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/corrected_results.csv) (row 17)
  - Multi-seed records: [`outputs/phase24b/per_trace_results.csv`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/per_trace_results.csv) (regime `periodic_high_tight`)
  - Accounting audit: [`outputs/phase24b/results_verification.md`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/results_verification.md)
- **Manuscript Phrasing Rule:** State as a **70 percentage-point increase in on-time completion rate**. Do **not** describe this as an execution speedup.

---

### Claim 2: Orthogonal Benefits of Admission Control vs. Adaptive Pruning
- **Claim Statement:** The proposed architecture derives its performance from two distinct mechanisms: admission control prevents queue collapse (+50.0% gain, eliminating late tasks), while dynamic token pruning recovers 28% latency slack, admitting 20 additional tasks on time (+20.0% gain).
- **Classification:** **`INTERPRETATION_OR_HYPOTHESIS`** (substantiated by controlled ablation).
- **Exact Evidence Artifacts:**
  - Ablation matrix: [`outputs/phase24b/ablation_report.md`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/ablation_report.md)
  - Rebuilt results: [`outputs/phase24b/corrected_results.csv`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/corrected_results.csv)

---

### Claim 3: Preservation of Downstream Classification Quality
- **Claim Statement:** Dynamic token pruning down to $\rho=0.5$ preserves high classification fidelity. The model-weighted synthetic test accuracy over completed requests is **92.60%** (a 0.07% drop from the 92.67% unpruned baseline). On physical RTX 4060 GPU execution over test image samples, empirical top-1 accuracy is **87.5% – 91.3%**.
- **Classification:** **`EMPIRICALLY_EVALUATED_ON_LOCAL_DATASET`**.
- **Exact Evidence Artifacts:**
  - Full-dataset baseline benchmarks: [`outputs/phase23b/model_comparison.csv`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase23b/model_comparison.csv)
  - Accuracy provenance audit: [`outputs/phase24b/accuracy_provenance.md`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/accuracy_provenance.md)
  - Physical replay accuracy: [`outputs/phase24b/simulation_hardware_comparison.csv`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/simulation_hardware_comparison.csv)
- **Manuscript Phrasing Rule:** Explicitly distinguish the synthetic dataset expectation (92.60%) from small-batch physical sample accuracy (87.5%–91.3%). Accurately state that 92.53% was the validation accuracy of EViT 0.5.

---

### Claim 4: CUDA Graph Acceleration on Consumer Edge GPUs
- **Claim Statement:** On an NVIDIA GeForce RTX 4060 Laptop GPU, executing DeiT/EViT with batch size 1 under eager PyTorch suffers from kernel launch overhead ($13.4\text{–}15.7\text{ ms}$). Capturing execution via CUDA Graphs eliminates launch overhead, reducing batch-1 latency to **$2.36\text{–}3.28\text{ ms}$** and unlocking practical token pruning acceleration.
- **Classification:** **`MEASURED_LOCALLY`**.
- **Exact Evidence Artifacts:**
  - Latency distributions: [`outputs/phase24/model_profiles.json`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24/model_profiles.json)
  - Raw benchmark logs: [`outputs/phase23b/raw_benchmark_data.json`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase23b/raw_benchmark_data.json)

---

### Claim 5: Simulation-to-Hardware Makespan and Bursty Behavior
- **Claim Statement:** Discrete-event simulation predicts physical GPU execution makespan within **$0.05\% \text{ to } 0.83\%$** relative error across workloads. For on-time completion under bursty traffic, simulation parameterized with p95 latency is conservative (27.0% simulated vs. 60.0% physical for FIFO), because the real GPU operates near median latency ($3.15\text{ ms}$), clearing burst queues faster. Under the proposed policy (`EDF_Adaptive_AC`), proactive shedding yields close agreement (79.0% simulated vs. 83.0% physical).
- **Classification:** **`MEASURED_LOCALLY`** (paired simulation vs. hardware replay).
- **Exact Evidence Artifacts:**
  - Hardware comparison table: [`outputs/phase24b/simulation_hardware_comparison.csv`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/simulation_hardware_comparison.csv)
  - Audit report: [`outputs/phase24b/data_dictionary.md`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/data_dictionary.md)

---

### Claim 6: Firm Real-Time Semantics vs. Hard WCET Guarantees
- **Claim Statement:** The proposed architecture operates under firm real-time semantics, optimizing on-time QoS. Measured latency distributions (mean, median, p95, p99) reflect empirical profiles and do **not** constitute certifiable hard real-time WCET guarantees due to OS interrupts, PCIe contention, and driver scheduling on Windows 11 / consumer NVIDIA platforms.
- **Classification:** **`INTERPRETATION_OR_HYPOTHESIS`** (honest bound disclosure).
- **Exact Evidence Artifacts:**
  - Formal bound disclaimer: [`outputs/phase24/model_profiles.json`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24/model_profiles.json) (metadata)
  - System specification: [`outputs/phase24b/data_dictionary.md`](file:///c:/Users/palak/Desktop/RTS_PROJECT/outputs/phase24b/data_dictionary.md)

---

### Claim 7: Official EViT Architecture and Token Reorganization
- **Claim Statement:** Spatial token reorganization in Vision Transformers expedites inference by identifying inattentive patch tokens via self-attention scores and fusing them into an aggregate token at layers 3, 6, and 9.
- **Classification:** **`SUPPORTED_BY_PUBLISHED_LITERATURE`**.
- **Published Reference:**
  - Youwei Liang et al., *"Not All Patches Are What You Need: Expediting Vision Transformers via Token Reorganizations,"* ICLR 2022.
  - Official Repository commit: `5ca95a7cf4a8637193284895c5bc0dd9dc365982`.

---

## 3. Unsupported Claims (Explicitly Retracted & Excluded)

| Retracted Claim | Prior Flawed Phrasing | Reason for Retraction | Replacement Truthful Phrasing |
|---|---|---|---|
| **EDF Priority Superiority** | *"EDF queue ordering systematically outperforms FIFO across paired workloads."* | **Contradicted by data.** FIFO and EDF tie in 8/10 workloads because absolute deadlines are monotonic with arrival order. In 2/10 overloaded workloads, FIFO beats EDF by 3%–4% due to non-preemptive domino misses. | *"Under non-preemptive GPU execution, EDF queue ordering alone provides no advantage over FIFO when deadlines are monotonic, and can induce cascading domino misses in unconstrained overload."* |
| **Universal 3% Simulation Fidelity** | *"Simulation and GPU replay agree within 3% relative error across all workloads."* | **Contradicted by data.** Table 4 in Phase 24 altered simulation numbers to manufacture this claim. Actual on-time completion diverged significantly on bursty traffic (27% vs 60%) due to p95 parameterization. | *"Simulation predicts makespan within 0.83%, but p95-parameterized simulation yields conservative on-time predictions under bursty traffic compared to physical hardware."* |
| **12.67x Execution Speedup** | *"Proposed policy delivers a 12.67x speedup in real-time performance."* | **Terminological error.** 76% vs 6% is an increase in on-time completion rate, not an algorithmic speedup. | *"Achieves a 70 percentage-point increase in on-time completion rate (from 6.0% to 76.0%)."* |
| **92.53% Empirical Test Accuracy** | *"Adaptive controller achieves 92.53% empirical test accuracy across workloads."* | **Factual error.** 92.53% was the validation accuracy of EViT 0.5. Synthetic expected test accuracy is 92.60%, and empirical GPU replay accuracy is 87.5%–91.3%. | *"Preserves an expected test accuracy of 92.60%, with physical sample accuracy ranging from 87.5% to 91.3%."* |
