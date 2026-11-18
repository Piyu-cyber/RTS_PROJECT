# Research Paper

## Title: Deadline-Aware Adaptive Vision Transformer Inference for Soft Real-Time Edge Workloads

**Authors:**
- **Piyush Prashant** (Dept. of Data Science and AI, IIIT Dharwad)
- **Priyanshu Mittal** (Dept. of Data Science and AI, IIIT Dharwad)
- **Thejas Gowda U M** (Dept. of Computer Science, IIIT Dharwad)
- **Shriyansh S Sahoo** (Dept. of Computer Science, IIIT Dharwad)
- **Swagatika Sahoo** (Dept. of Computer Science, IIIT Dharwad)

---

### PDF Document
The complete research manuscript is available at:
- [**`Deadline-Aware_Adaptive_ViT_Inference.pdf`**](Deadline-Aware_Adaptive_ViT_Inference.pdf)

### Abstract Summary
Vision Transformers (ViTs) offer strong perception accuracy, but their quadratic attention cost and execution jitter make them hard to deploy in latency-critical edge systems, and class imbalance in driving data degrades pruned models further. This paper combines Efficient Vision Transformer (EViT) token pruning, class-balanced cross-entropy with teacher-guided margin-preserving regularization (CB-CMPR), and a deadline-aware scheduler with admission control (DA-ASAC). On the six-class BDD100K scene dataset, EViT with keep rate 0.7 reaches 77.88 ± 0.15% Top-1 accuracy over three seeds, reduces computation by 34.1%, and sustains 629.0 FPS on an NVIDIA RTX 4060 GPU. Across 70 simulated workload-policy configurations, the scheduler raises on-time completion by up to 73 percentage points under saturation, and GPU trace replay matches simulated makespan within 0.57%.
