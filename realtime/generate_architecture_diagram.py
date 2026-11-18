"""Generate system architecture diagram showing what was taken from Scribano et al. vs what we contributed.

Components:
  1. Base Research Paper (Scribano et al. ICPR 2026):
     - Attention-to-DWConv Module (CnvAttnDW_blockwise, 7x7 spatial kernel, value weight transfer).
     - Attention Map Variability Selection (Sigma_h, Sigma_b).
  2. Our Contributions:
     - Downstream Dense Semantic Segmentation Head (Oxford-IIIT Pet 2-layer Conv Head).
     - Two-Stage Fine-Tuning Protocol (decoupling head adaptation from DWConv alignment).
     - Uniform Depth Interleaving Pattern ([2,5,8,11] vs failure of contiguous Sigma_b).
     - Accuracy-Constrained Real-Time Adaptive Controller (tau_mIoU floor, latency bound M).
     - Deterministic Fallback Policies (highest_quality_miss, fastest_miss, selective reject).
     - Hardware Execution Reality & Benchmarking (Pareto inversion on RTX 4060 GPU).
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase11_manuscript"
OUT_PATH = OUTPUT_DIR / "overall_system_architecture.png"


def render_system_architecture() -> Path:
    # 1200 x 700 px high-resolution architecture diagram
    w, h = 1200, 720
    img = Image.new("RGB", (w, h), color="#ffffff")
    draw = ImageDraw.Draw(img)

    # Header Banner
    draw.rectangle([0, 0, w, 55], fill="#0f172a")
    draw.text((30, 16), "DEADLINE-AWARE ADAPTIVE INFERENCE ARCHITECTURE FOR DROP-IN DWCONV ViTs", fill="#f8fafc")
    draw.text((850, 20), "SYSTEM TAXONOMY & CONTRIBUTIONS", fill="#94a3b8")

    # Legend / Color Codes
    draw.rectangle([30, 68, 50, 84], fill="#e0f2fe", outline="#0284c7", width=2)
    draw.text((58, 69), "Adopted from Base Paper (Scribano et al. ICPR 2026)", fill="#0369a1")

    draw.rectangle([450, 68, 470, 84], fill="#fef3c7", outline="#d97706", width=2)
    draw.text((478, 69), "Investigated / Ablated (Sigma_b Selection Criterion)", fill="#b45309")

    draw.rectangle([850, 68, 870, 84], fill="#dcfce7", outline="#16a34a", width=2)
    draw.text((878, 69), "Our Novel Contributions (Phases 1 - 10D)", fill="#15803d")

    # Container 1: Offline Training & Model Preparation (Left Side)
    c1_x0, c1_y0, c1_x1, c1_y1 = 30, 100, 570, 690
    draw.rectangle([c1_x0, c1_y0, c1_x1, c1_y1], fill="#f8fafc", outline="#cbd5e1", width=2)
    draw.rectangle([c1_x0, c1_y0, c1_x1, c1_y0 + 32], fill="#e2e8f0")
    draw.text((c1_x0 + 15, c1_y0 + 8), "1. MODEL ARCHITECTURE & TWO-STAGE TRAINING", fill="#1e293b")

    # Box 1.1: Pretrained Foundation Model
    draw.rectangle([50, 145, 550, 215], fill="#e0f2fe", outline="#0284c7", width=2)
    draw.text((65, 153), "Pretrained DINOv2-ViT-S/14 Foundation Backbone", fill="#0369a1")
    draw.text((65, 172), "12 Transformer Blocks | 384 Dim | 6 Heads | Patch 14x14 | 224x224 (257 Tokens)", fill="#475569")
    draw.text((65, 192), "[Scribano et al. Base Backbone: Pretrained feature extractor]", fill="#0284c7")

    # Arrow Down
    draw.line([(300, 215), (300, 235)], fill="#64748b", width=2)

    # Box 1.2: Stage 1 Training (Our addition)
    draw.rectangle([50, 235, 550, 315], fill="#dcfce7", outline="#16a34a", width=2)
    draw.text((65, 243), "Stage 1: Downstream Task Adaptation (Frozen Backbone)", fill="#15803d")
    draw.text((65, 262), "2-Layer Conv Segmentation Head (384 -> 128 -> 2 classes: Background vs Pet)", fill="#475569")
    draw.text((65, 281), "Oxford-IIIT Pet Split (32 val, 704k pixels) | AdamW lr=1e-3 | 3 Epochs -> mIoU 0.7430", fill="#166534")

    # Arrow Down
    draw.line([(300, 315), (300, 335)], fill="#64748b", width=2)

    # Box 1.3: Block Selection Ablation (Scribano vs Ours)
    draw.rectangle([50, 335, 550, 480], fill="#ffffff", outline="#94a3b8", width=1)
    draw.text((65, 343), "Attention-to-DWConv Replacement Policy Ablation:", fill="#1e293b")

    # Sub-box Sigma_b (Scribano)
    draw.rectangle([65, 365, 290, 465], fill="#fef3c7", outline="#d97706", width=2)
    draw.text((75, 372), "Statistical Sigma_b Selection", fill="#b45309")
    draw.text((75, 390), "Section 3.3 Scribano et al.", fill="#78350f")
    draw.text((75, 408), "Variance of attention E_h(X)", fill="#475569")
    draw.text((75, 426), "Replaces: [1, 2, 3, 4] (Early)", fill="#dc2626")
    draw.text((75, 444), "Result: mIoU 0.4801 (Collapse!)", fill="#b91c1c")

    # Sub-box Manual Interleaved (Ours)
    draw.rectangle([310, 365, 535, 465], fill="#dcfce7", outline="#16a34a", width=2)
    draw.text((320, 372), "Manual Interleaving Policy", fill="#15803d")
    draw.text((320, 390), "Our Proposed Best Practice", fill="#166534")
    draw.text((320, 408), "Uniform depth distribution", fill="#475569")
    draw.text((320, 426), "Replaces: [2, 5, 8, 11]", fill="#047857")
    draw.text((320, 444), "Result: mIoU 0.7036 (Preserved)", fill="#15803d")

    # Arrow Down
    draw.line([(300, 480), (300, 500)], fill="#64748b", width=2)

    # Box 1.4: Stage 2 Joint Fine-tuning
    draw.rectangle([50, 500, 550, 580], fill="#e0f2fe", outline="#0284c7", width=2)
    draw.text((65, 508), "Stage 2: Drop-in Weight Copy & Joint Fine-Tuning", fill="#0369a1")
    draw.text((65, 527), "Pretrained V-projection & Proj weights copied directly into CnvAttnDW_blockwise", fill="#475569")
    draw.text((65, 546), "AdamW lr=1e-4 | 10 Epochs | Drop-in DWConv weights recovered without drift", fill="#0284c7")

    # Arrow Down
    draw.line([(300, 580), (300, 600)], fill="#64748b", width=2)

    # Box 1.5: Validated Candidate Registry
    draw.rectangle([50, 600, 550, 675], fill="#dcfce7", outline="#16a34a", width=2)
    draw.text((65, 608), "Audited Candidate Model Registry (SHA-256 Verified)", fill="#15803d")
    draw.text((65, 627), "M_full: mIoU 0.7430 (13.13ms) | M_mod: mIoU 0.7036 (14.31ms) | M_sigma: mIoU 0.4801 (14.09ms)", fill="#1e293b")
    draw.text((65, 646), "Common-Protocol Device Benchmark on NVIDIA RTX 4060 Laptop GPU (150 trials)", fill="#475569")

    # Container 2: Online Real-Time Adaptive Scheduler & Runtime Execution (Right Side)
    c2_x0, c2_y0, c2_x1, c2_y1 = 610, 100, 1170, 690
    draw.rectangle([c2_x0, c2_y0, c2_x1, c2_y1], fill="#f8fafc", outline="#cbd5e1", width=2)
    draw.rectangle([c2_x0, c2_y0, c2_x1, c2_y0 + 32], fill="#e2e8f0")
    draw.text((c2_x0 + 15, c2_y0 + 8), "2. ACCURACY-CONSTRAINED DEADLINE SCHEDULER & RUNTIME", fill="#1e293b")

    # Box 2.1: Request Ingestion
    draw.rectangle([630, 145, 1150, 215], fill="#ffffff", outline="#94a3b8", width=1)
    draw.text((645, 153), "Incoming Inference Request: Image X | Deadline D (ms) | Quality Floor tau_mIoU", fill="#1e293b")
    draw.text((645, 172), "Workloads: Default Mixture (30% tight, 40% mod, 30% rel) & Stress Tight (60% tight)", fill="#475569")
    draw.text((645, 192), "Calibrated Latency Bound Metric M in {p90, p95, p99, median, max_observed}", fill="#0284c7")

    # Arrow Down
    draw.line([(890, 215), (890, 235)], fill="#64748b", width=2)

    # Box 2.2: Accuracy-Constrained Controller (Our Novel Module)
    draw.rectangle([630, 235, 1150, 395], fill="#dcfce7", outline="#16a34a", width=2)
    draw.text((645, 243), "Our Real-Time Accuracy-Constrained Controller (realtime/accuracy_scheduler.py)", fill="#15803d")
    draw.text((645, 264), "Step 1 (Quality Filter): C_quality = { c in C | validation_mIoU(c) >= tau_mIoU }", fill="#1e293b")
    draw.text((645, 285), "Step 2 (Deadline Filter): C_deadline = { c in C | Latency_M(c) <= Deadline D }", fill="#1e293b")
    draw.text((645, 306), "Step 3 (Feasible Set): F = C_quality intersect C_deadline", fill="#166534")
    draw.text((645, 327), "  - If F not empty: Select c* = argmax_{F} (mIoU(c), -Latency_M(c)) [Dispatched!]", fill="#047857")
    draw.text((645, 348), "  - If F empty (Infeasible): Trigger Deterministic Fallback Policy", fill="#b91c1c")
    draw.text((645, 369), "Controller Dispatch Overhead: 0.045 - 0.055 ms (< 0.3% of budget, verified soft real-time)", fill="#0f766e")

    # Arrow Down
    draw.line([(890, 395), (890, 415)], fill="#64748b", width=2)

    # Box 2.3: Fallback Policies (Our Novel Contribution)
    draw.rectangle([630, 415, 1150, 520], fill="#ffffff", outline="#94a3b8", width=1)
    draw.text((645, 423), "Infeasibility Fallback Execution Policies:", fill="#1e293b")

    # Policy 1: Reject
    draw.rectangle([645, 445, 795, 510], fill="#fee2e2", outline="#ef4444", width=2)
    draw.text((655, 452), "SELECTIVE REJECT", fill="#b91c1c")
    draw.text((655, 470), "Sheds 30% load", fill="#7f1d1d")
    draw.text((655, 488), "Disp. Hit: 77.1%", fill="#991b1b")

    # Policy 2: Highest Quality Miss
    draw.rectangle([815, 445, 975, 510], fill="#e0f2fe", outline="#0284c7", width=2)
    draw.text((825, 452), "HIGHEST QUALITY MISS", fill="#0369a1")
    draw.text((825, 470), "Dispatches M_full", fill="#075985")
    draw.text((825, 488), "Max Utility (0.7430)", fill="#0284c7")

    # Policy 3: Fastest Miss
    draw.rectangle([995, 445, 1135, 510], fill="#fef3c7", outline="#d97706", width=2)
    draw.text((1005, 452), "FASTEST MISS", fill="#b45309")
    draw.text((1005, 470), "Minimizes overrun", fill="#78350f")
    draw.text((1005, 488), "Dispatches min lat", fill="#d97706")

    # Arrow Down
    draw.line([(890, 520), (890, 540)], fill="#64748b", width=2)

    # Box 2.4: Runtime GPU Execution & Outcomes
    draw.rectangle([630, 540, 1150, 675], fill="#dcfce7", outline="#16a34a", width=2)
    draw.text((645, 548), "Runtime Execution on NVIDIA RTX 4060 Laptop GPU (Ada Lovelace)", fill="#15803d")
    draw.text((645, 568), "Pipeline: Preprocess (~4.9ms) -> PCIe Transfer (~0.28ms) -> Model (~13.1ms) -> Post (~0.45ms)", fill="#475569")
    draw.text((645, 588), "Total End-to-End Latency: 18.00 ms (Median) | Cold-Start First Frame: 240.51 ms (+1919%)", fill="#1e293b")
    draw.text((645, 608), "Prediction Calibration: p99 bounds yield 92.16% precision (75% reduction in false-feasible errors)", fill="#047857")
    draw.text((645, 628), "Real-Time Classification: Soft Real-Time Verified | Hard Real-Time Formally Disclaimed", fill="#15803d")

    # Connect Container 1 Registry to Container 2 Controller
    draw.line([(550, 637), (580, 637), (580, 275), (630, 275)], fill="#0284c7", width=3)

    img.save(OUT_PATH)
    return OUT_PATH


if __name__ == "__main__":
    p = render_system_architecture()
    print(f"Generated system architecture diagram: {p}")
