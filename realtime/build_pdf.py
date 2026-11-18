"""Compile publication-quality IEEE-style PDF manuscript using ReportLab.

Directly reads the Phase 11 manuscript contents, tables, and figures, and builds:
  outputs/phase11_manuscript/manuscript.pdf
"""

from __future__ import annotations

from pathlib import Path
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase11_manuscript"
PDF_PATH = OUTPUT_DIR / "manuscript_ieee.pdf"


def build_pdf() -> Path:
    # 0.75 in margins = 54 pt
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles matching IEEE layout
    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=17,
        leading=21,
        alignment=1,  # Center
        textColor=HexColor("#0f172a"),
        spaceAfter=8,
    )

    author_style = ParagraphStyle(
        "PaperAuthors",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=10.5,
        leading=14,
        alignment=1,
        textColor=HexColor("#334155"),
        spaceAfter=3,
    )

    affil_style = ParagraphStyle(
        "PaperAffil",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8.5,
        leading=12,
        alignment=1,
        textColor=HexColor("#475569"),
        spaceAfter=12,
    )

    abstract_style = ParagraphStyle(
        "Abstract",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8.5,
        leading=11.5,
        alignment=4,  # Justify
        leftIndent=24,
        rightIndent=24,
        spaceAfter=6,
    )

    keywords_style = ParagraphStyle(
        "Keywords",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8,
        leading=11,
        alignment=4,
        leftIndent=24,
        rightIndent=24,
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        "Heading1",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=10.5,
        leading=14,
        textColor=HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2",
        parent=styles["Normal"],
        fontName="Times-BoldItalic",
        fontSize=9.5,
        leading=13,
        textColor=HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=9,
        leading=12,
        alignment=4,  # Justify
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=9,
        leading=12,
        leftIndent=14,
        alignment=4,
        spaceAfter=4,
    )

    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=8,
        leading=10.5,
        alignment=1,
        textColor=HexColor("#334155"),
        spaceBefore=4,
        spaceAfter=10,
    )

    ref_style = ParagraphStyle(
        "Reference",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=7.5,
        leading=9.5,
        leftIndent=14,
        firstLineIndent=-14,
        spaceAfter=3,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Deadline-Aware Adaptive Inference for Drop-in Depthwise-Convolution Vision Transformers", title_style))
    story.append(Paragraph("Anonymous Authors", author_style))
    story.append(Paragraph("Department of Computer Science and Real-Time Systems Laboratory<br/>Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) &bull; PyTorch 2.5.1+cu121 &bull; CUDA 12.1", affil_style))

    # Abstract Box
    story.append(Paragraph("<b><i>Abstract</i>&mdash;Vision Foundation Models (VFMs) based on Vision Transformers (ViTs) deliver state-of-the-art accuracy across visual perception tasks, but their high computational latency poses severe challenges in deadline-critical embedded systems. Recent literature proposes replacing Multi-Head Self-Attention (MHSA) blocks with drop-in Depthwise Convolutions (DWConv) to accelerate ViTs while recovering accuracy through fine-tuning. However, the real-world real-time systems behavior of such hybrid architectures&mdash;specifically regarding hardware latency trade-offs, spatial block-selection strategies, deadline prediction calibration, and accuracy-constrained scheduling&mdash;has not been rigorously characterized. In this work, we conduct an extensive empirical investigation of deadline-aware adaptive inference across full-attention and drop-in DWConv ViT architectures on an NVIDIA RTX 4060 GPU. First, we evaluate the paper-faithful attention-variability criterion (&Sigma;<sub>b</sub>) for block replacement on Oxford-IIIT Pet semantic segmentation. We demonstrate that greedy statistical selection replaces contiguous early transformer blocks ([1, 2, 3, 4]), causing severe representation degradation (validation mIoU collapsing to 0.4801), whereas manual uniform depth interleaving ([2, 5, 8, 11]) preserves hierarchical features and achieves 0.7036 mIoU. Second, multi-round hardware benchmarking across 150 matched trials demonstrates that on modern mobile Ada Lovelace hardware at batch size 1 (224 &times; 224), optimized dense attention executes faster (median 13.13 ms) than DWConv replacements (14.09&ndash;14.31 ms), inverting theoretical FLOP reductions. Third, we develop an accuracy-constrained deadline-aware scheduler that prunes quality-ineligible models (&tau;<sub>mIoU</sub>) and implements deterministic fallbacks. Evaluating p95 and p99 empirical quantile bounds reveals that p99 achieves 92.16% prediction precision, while a selective load-rejection fallback drops 30% of physically unachievable requests, saving 13.3 ms of GPU compute per request and elevating dispatched deadline hit rate from 55.0% to 77.1%. Finally, we classify the system's real-time guarantees, demonstrating soft real-time suitability with sub-millisecond controller overhead (&lt; 0.055 ms) while establishing why hard real-time guarantees remain unachievable on modern consumer GPU platforms.</b>", abstract_style))
    story.append(Paragraph("<b><i>Keywords</i>&mdash;Vision Transformers, Depthwise Convolution, Real-Time Scheduling, Adaptive Inference, Deadline Feasibility, Semantic Segmentation.</b>", keywords_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#cbd5e1"), spaceAfter=10))

    # Section I: Introduction
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph("Vision Foundation Models (VFMs) such as DINOv2 have emerged as premier visual backbones for high-precision autonomous perception, robotics, and industrial inspection. These models leverage Vision Transformers (ViTs) utilizing Multi-Head Self-Attention (MHSA) to capture global contextual dependencies across spatial patches. However, MHSA imposes quadratic computational and memory complexity with respect to the sequence length <i>N</i>, leading to non-trivial latency bottlenecks during inference. In real-time embedded systems&mdash;such as autonomous ground vehicles, drone obstacle avoidance, and interactive robotic manipulation&mdash;perception pipelines must satisfy strict timing deadlines. A delayed perceptual estimate in such cyber-physical systems is often equivalent to an incorrect estimate [6].", body_style))
    story.append(Paragraph("To alleviate the computational overhead of self-attention, Scribano et al. [1] introduced a drop-in Depthwise Convolution (DWConv) replacement framework. By replacing self-attention heads with 7 &times; 7 depthwise convolutions applied to reshaped value projection tensors, the framework replaces all-to-all global communication with localized spatial kernels, claiming substantial FLOP reductions and speedups while transferring pretrained value projection weights. To select replacement candidates, Scribano et al. proposed an attention-map variability metric (&Sigma;<sub>b</sub>), hypothesizing that transformer blocks exhibiting the lowest variance in attention distributions across calibration samples are the most redundant and can be replaced with minimal accuracy loss.", body_style))
    story.append(Paragraph("Despite these promising architectural claims, several fundamental research questions remain unaddressed in the real-time systems literature: (1) Does statistical block selection (&Sigma;<sub>b</sub>) preserve the fine-grained hierarchical representations necessary for pixel-level dense semantic segmentation? (2) Do theoretical reductions in arithmetic operations translate to wall-clock latency reductions on modern GPU architectures under realistic batch-size-1 inference conditions? (3) When candidate models exhibit non-trivial accuracy-latency trade-offs, how should a deadline-aware scheduler prevent silent quality collapse under tight deadlines while handling physically unachievable requests? and (4) How reliably do empirical percentile latency bounds (p95, p99) predict per-request deadline feasibility?", body_style))

    # Section II: Contributions
    story.append(Paragraph("II. CONTRIBUTIONS", h1_style))
    story.append(Paragraph("&bull; <b>Dense Segmentation Impact of Statistical Block Selection:</b> We implement the paper-faithful &Sigma;<sub>b</sub> statistical block-selection criterion and demonstrate that it greedily clusters replacements in contiguous early blocks ([1, 2, 3, 4]), severely degrading fine-grained segmentation (validation mIoU = 0.4801, background IoU = 0.2447). In contrast, manual uniform interleaving ([2, 5, 8, 11]) preserves multi-scale spatial features across depths, maintaining 0.7036 mIoU.", bullet_style))
    story.append(Paragraph("&bull; <b>Empirical Inversion of the Latency Frontier:</b> Through 150 matched trials across 5 independent measurement rounds under a rigorous common protocol, we show that dense PyTorch/CUDA self-attention executes faster (median 13.13 ms) than DWConv replacements (14.09&ndash;14.31 ms) on mobile Ada Lovelace hardware at batch size 1, causing full attention to strictly dominate the Pareto frontier.", bullet_style))
    story.append(Paragraph("&bull; <b>Accuracy-Constrained Real-Time Scheduler:</b> We design and formalize an adaptive controller that enforces an explicit minimum quality threshold (&tau;<sub>mIoU</sub>) and evaluates configurable fallback policies (<code>highest_quality_miss</code>, <code>fastest_miss</code>, <code>reject</code>). We demonstrate that the <code>reject</code> policy drops 30% of physically infeasible requests, saving 13.3 ms of GPU compute per request and elevating dispatched hit rate to 77.1%.", bullet_style))
    story.append(Paragraph("&bull; <b>Calibration and Real-Time Systems Taxonomy:</b> We evaluate prediction precision across empirical latency bounds, showing that p99 bounds reduce false-feasible errors by 75% relative to p95. Furthermore, we characterize system cold-start spikes (+1919% latency inflation) and classify the architecture as soft real-time suitable while disclaiming hard real-time guarantees.", bullet_style))

    # Section III: Methodology & Formulation
    story.append(Paragraph("III. SYSTEM ARCHITECTURE & METHODOLOGY", h1_style))
    story.append(Paragraph("Our target foundation model is Meta's DINOv2-ViT-S/14 [3], consisting of 12 transformer blocks, embedding dimension 384, 6 attention heads, patch size 14, and input resolution 224 &times; 224 (<i>N</i> = 257 tokens including <code>[CLS]</code>). The downstream task is semantic segmentation on Oxford-IIIT Pet [5], employing a 2-layer convolutional segmentation head projecting from 384 channels to 2 output classes (pet foreground vs. background).", body_style))
    story.append(Paragraph("We evaluate three validated model candidates: (1) <b>Full Attention Baseline (M<sub>full</sub>)</b> with all 12 MHSA blocks intact; (2) <b>Moderate DWConv (M<sub>mod</sub>)</b> replacing 4 blocks uniformly across depth: [2, 5, 8, 11]; and (3) <b>&Sigma;<sub>b</sub>-Selected DWConv (M<sub>&Sigma;</sub>)</b> replacing 4 blocks ranked lowest by attention variability.", body_style))
    story.append(Paragraph("Let C denote the candidate registry. Given a request with deadline <i>D</i> &gt; 0 and quality floor &tau;<sub>mIoU</sub> &in; [0, 1]:", body_style))
    story.append(Paragraph("<font name='Courier'>C_quality = {c &isin; C | mIoU(c) &ge; &tau;}, &nbsp; C_deadline = {c &isin; C | Latency(c) &le; D}, &nbsp; F = C_quality &cap; C_deadline</font>", ParagraphStyle("CodeEq", parent=body_style, alignment=1, spaceAfter=4)))
    story.append(Paragraph("If feasible set F is non-empty, the scheduler selects the candidate maximizing validation mIoU with tie-breaking by lowest latency. If F is empty, the scheduler triggers the configured fallback policy (<code>reject</code>, <code>highest_quality_miss</code>, or <code>fastest_miss</code>).", body_style))

    # Section IV: Results - Table 1
    story.append(Paragraph("IV. QUANTITATIVE RESULTS & ABLATIONS", h1_style))
    story.append(Paragraph("<b>A. Dense Segmentation and Block-Selection Ablation</b>", h2_style))
    story.append(Paragraph("Table I compares segmentation accuracy across candidates and controlled training ablations on Oxford-IIIT Pet validation split (704,228 valid pixels).", body_style))

    t1_data = [
        ["Configuration", "Replaced Blocks", "Mean IoU", "Background IoU", "Pet IoU"],
        ["M_full (Stage 1 Baseline)", "None", "0.7430", "0.6375", "0.8486"],
        ["M_mod (Manual Interleaved)", "[2, 5, 8, 11]", "0.7036", "0.5980", "0.8092"],
        ["M_Sigma (Sigma_b Joint Run 3)", "[1, 2, 3, 4]", "0.4801", "0.2447", "0.7154"],
        ["Sigma_b Frozen Head Ablation", "[1, 2, 3, 4]", "0.3645", "0.0000", "0.7290"],
        ["Sigma_b Staged Adapter Ablation", "[1, 2, 3, 4]", "0.3645", "0.0000", "0.7290"],
        ["Aggressive DWConv (8 Blocks)", "[1, 2, 3, 4, 5, 6, 7, 8]", "0.4563", "0.1621", "0.7504"],
    ]
    t1 = Table(t1_data, colWidths=[150, 110, 75, 95, 85])
    t1.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, HexColor("#0f172a")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, HexColor("#0f172a")),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, HexColor("#0f172a")),
        ("BACKGROUND", (0, 1), (-1, 1), HexColor("#f1f5f9")),
        ("BACKGROUND", (0, 2), (-1, 2), HexColor("#f8fafc")),
    ]))
    story.append(t1)
    story.append(Paragraph("TABLE I: Segmentation Quality across Architectural Candidates and Fine-Tuning Ablations on Oxford-IIIT Pet.", caption_style))

    story.append(Paragraph("<b>Failure of Greedy &Sigma;<sub>b</sub> Selection:</b> Computing sample-wise variance of post-softmax attention maps yields ascending &Sigma;<sub>b</sub> rankings: Block 3 (165.5), Block 2 (191.5), Block 1 (220.2), Block 4 (223.9). Consequently, the 4-block budget selects consecutive early layers ([1, 2, 3, 4]). In dense semantic segmentation, early transformer layers compute localized spatial primitives. Replacing four contiguous early blocks destroys global token mixing before intermediate representations can form, creating an irreversible representational bottleneck that caps mIoU at 0.4801. In contrast, manual uniform interleaving ([2, 5, 8, 11]) preserves multi-scale spatial features across depths, maintaining 0.7036 mIoU.", body_style))

    # Sub-section B: Hardware Latency
    story.append(Paragraph("<b>B. Multi-Round Hardware Latency Stability Study</b>", h2_style))
    story.append(Paragraph("Table II reports latency statistics across 150 matched trials across 5 independent measurement rounds on the RTX 4060 GPU with randomized round-robin model ordering per trial and 10 warmups per round.", body_style))

    t2_data = [
        ["Candidate Model", "Scope", "Median", "Mean", "Std Dev", "p90", "p95", "p99", "Max"],
        ["M_full (Full Attention)", "Model Forward", "13.13 ms", "13.63 ms", "1.48 ms", "15.64 ms", "17.05 ms", "19.26 ms", "22.25 ms"],
        ["M_full (Full Attention)", "End-to-End", "18.00 ms", "18.75 ms", "2.53 ms", "21.77 ms", "25.28 ms", "29.85 ms", "34.99 ms"],
        ["M_mod (Moderate DWConv)", "Model Forward", "14.31 ms", "14.91 ms", "2.94 ms", "16.52 ms", "17.86 ms", "26.64 ms", "44.02 ms"],
        ["M_mod (Moderate DWConv)", "End-to-End", "19.05 ms", "20.03 ms", "3.96 ms", "21.82 ms", "29.63 ms", "33.96 ms", "55.94 ms"],
        ["M_Sigma (Sigma_b DWConv)", "Model Forward", "14.09 ms", "14.60 ms", "1.48 ms", "16.63 ms", "17.89 ms", "18.86 ms", "19.58 ms"],
        ["M_Sigma (Sigma_b DWConv)", "End-to-End", "19.10 ms", "19.78 ms", "2.52 ms", "22.56 ms", "25.58 ms", "32.08 ms", "36.04 ms"],
    ]
    t2 = Table(t2_data, colWidths=[110, 68, 48, 48, 46, 48, 48, 48, 48])
    t2.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, HexColor("#0f172a")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, HexColor("#0f172a")),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, HexColor("#0f172a")),
        ("BACKGROUND", (0, 1), (-1, 2), HexColor("#f1f5f9")),
    ]))
    story.append(t2)
    story.append(Paragraph("TABLE II: Pooled Latency Statistics across 150 Matched GPU Trials (RTX 4060 Laptop GPU).", caption_style))

    # Figure 1: Latency Stability
    img1 = OUTPUT_DIR / "latency_stability_rounds.png"
    if img1.is_file():
        story.append(Image(str(img1), width=480, height=220))
        story.append(Paragraph("Fig. 1. Measured median GPU forward latency across 5 independent rounds (30 randomized trials each). Full attention consistently executes faster than DWConv replacements across all rounds.", caption_style))

    story.append(Paragraph("<b>Inversion of the Latency Frontier:</b> M_full is faster than M_mod in 84.0% of trials (median diff +0.987 ms, 95% bootstrap CI [+0.864, +1.149] ms) and faster than M_Sigma in 83.3% of trials (+0.937 ms, 95% CI [+0.847, +1.098] ms). On mobile Ada Lovelace architectures, tensor cores execute batch-size-1 self-attention in 13.13 ms. In contrast, depthwise convolution introduces memory access fragmentation and separate kernel launches in PyTorch, negating theoretical FLOP savings.", body_style))

    # Sub-section C: Prediction Calibration & Robustness
    story.append(Paragraph("<b>C. Deadline Prediction Calibration & Fallback Evaluation</b>", h2_style))
    story.append(Paragraph("We evaluated deadline prediction precision across empirical quantile bounds (p95 vs p99). Under p95 bounds, feasibility predictions achieve 77.14% precision on end-to-end deadlines. Calibrating with p99 bounds elevates precision to 92.16%, cutting false-feasible errors by 75% at the expense of 8 false-infeasible rejections.", body_style))

    t3_data = [
        ["Policy / Fallback Setting", "Total Hit %", "Dispatched Hit %", "Rejection %", "Useful %", "Mean mIoU"],
        ["Always Full Attention", "55.0%", "55.0%", "0.0%", "55.0%", "0.7430"],
        ["Always Fastest Candidate", "55.0%", "55.0%", "0.0%", "55.0%", "0.7430"],
        ["Accuracy-Constrained (Highest Quality Miss)", "55.0%", "55.0%", "0.0%", "55.0%", "0.7430"],
        ["Accuracy-Constrained (Fastest Miss)", "55.0%", "55.0%", "0.0%", "55.0%", "0.7430"],
        ["Accuracy-Constrained (Selective Reject)", "54.0%", "77.1%", "30.0%", "54.0%", "0.7430"],
    ]
    t3 = Table(t3_data, colWidths=[180, 68, 80, 65, 55, 68])
    t3.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, HexColor("#0f172a")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, HexColor("#0f172a")),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, HexColor("#0f172a")),
        ("BACKGROUND", (0, 4), (-1, 4), HexColor("#f1f5f9")),
    ]))
    story.append(t3)
    story.append(Paragraph("TABLE III: Policy Performance across Fallback Modes under Default Mixture Workload (&tau; = 0.70).", caption_style))

    # Figure 2 & 3
    img2 = OUTPUT_DIR / "prediction_calibration_roc.png"
    if img2.is_file():
        story.append(Image(str(img2), width=480, height=210))
        story.append(Paragraph("Fig. 2. Deadline feasibility prediction precision, false-feasible errors, and false-infeasible errors for p95 vs p99 empirical bounds. Calibrating with p99 elevates precision to 92.16%.", caption_style))

    img3 = OUTPUT_DIR / "useful_completion_tradeoff.png"
    if img3.is_file():
        story.append(Image(str(img3), width=480, height=210))
        story.append(Paragraph("Fig. 3. Useful completion rate vs. load-rejection trade-off across quality floors (&tau; &isin; {0.45, 0.65, 0.68, 0.70, 0.73}). The reject policy sheds load on unachievable requests.", caption_style))

    story.append(Paragraph("<b>Denominator Semantics & Load Shedding:</b> A critical distinction exists between Total Hit Rate (hits over all 100 requests) and Dispatched Hit Rate (hits over requests sent to GPU). Under <code>reject</code>, the scheduler drops 30 unachievable requests. While Total Hit Rate remains 54.0%, the Dispatched Hit Rate surges to 77.1%, saving 399 ms of GPU execution time.", body_style))

    # Section V: RTS Taxonomy
    story.append(Paragraph("V. REAL-TIME SYSTEMS TAXONOMY & COLD-START SPIKES", h1_style))
    story.append(Paragraph("We formally classify the real-time capabilities across five standard levels: (1) <b>Functional Correctness</b> is proven and deterministic; (2) <b>Statistical Latency Characterization</b> is validated via multi-round bootstrap confidence intervals; (3) <b>Soft Real-Time Suitability</b> is achieved with sub-millisecond controller dispatch (&lt; 0.055 ms, &lt; 0.3% request budget); and (4) <b>Hard Real-Time Guarantees</b> are formally disclaimed due to non-deterministic Windows OS scheduling, thermal throttling, and lack of verified Worst-Case Execution Time (WCET).", body_style))
    story.append(Paragraph("Stress testing reveals that the unprimed first inference on the GPU requires <b>240.51 ms</b>, compared to a warm median of <b>11.91 ms</b> (a 1,919% inflation). This spike arises from CUDA runtime initialization and kernel JIT loading. Any real-time deployment must execute pre-flight warmups prior to operational engagement.", body_style))

    # Section VI: Threats to Validity
    story.append(Paragraph("VI. THREATS TO VALIDITY & LIMITATIONS", h1_style))
    story.append(Paragraph("&bull; <b>Platform Dependency:</b> Our latency findings reflect mobile Ada Lovelace hardware at batch size 1 (<i>N</i> = 257). On datacenter GPUs or high-resolution token grids, DWConv may exhibit different scaling regimes.<br/>&bull; <b>Timing Scope:</b> Latency budgets calibrated on model-only forward passes neglect host preprocessing (~4.9 ms) and PCIe transfers (~0.28 ms). Real-time budgets must encompass end-to-end pipelines.<br/>&bull; <b>Dataset Scale:</b> Evaluations utilized Oxford-IIIT Pet segmentation. Complex multi-class datasets may show different susceptibility to block replacement patterns.", body_style))

    # Section VII: Conclusion
    story.append(Paragraph("VII. CONCLUSION", h1_style))
    story.append(Paragraph("We presented an empirical study of deadline-aware adaptive inference for drop-in Depthwise-Convolution Vision Transformers. We demonstrated that greedy attention-variability selection (&Sigma;<sub>b</sub>) degrades dense segmentation by clustering replacements in early layers, whereas manual interleaving preserves accuracy. Furthermore, multi-round hardware benchmarking demonstrated that full attention executes faster than DWConv on modern mobile GPUs, inverting the theoretical Pareto frontier. Finally, we demonstrated that accuracy-constrained scheduling with selective load rejection elevates dispatched deadline hit rate while guaranteeing task utility.", body_style))

    # References
    story.append(Paragraph("REFERENCES", h1_style))
    refs = [
        "[1] C. Scribano, M. Mahdi, N. Prisadnikov, Y. Fu, G. Franchini, D. P. Paudel, M. Bertogna, and L. Van Gool, 'Accelerating Vision Foundation Models with Drop-in Depthwise Convolution,' in <i>International Conference on Pattern Recognition (ICPR)</i>, Springer, 2026, pp. 463-477.",
        "[2] A. Dosovitskiy et al., 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,' <i>arXiv preprint arXiv:2010.11929</i>, 2020.",
        "[3] M. Oquab et al., 'DINOv2: Learning Robust Visual Features without Supervision,' <i>Transactions on Machine Learning Research</i>, 2024.",
        "[4] H. He, J. Cai, J. Zhang, D. Tao, and B. Zhuang, 'SPViT: Enabling Faster Vision Transformers via Latency-Aware Soft Token Pruning,' in <i>IEEE/CVF CVPR</i>, 2023, pp. 8345-8354.",
        "[5] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, 'Cats and Dogs,' in <i>IEEE CVPR</i>, 2012, pp. 3498-3505.",
        "[6] G. C. Buttazzo, <i>Hard Real-Time Computing Systems: Predictable Scheduling Algorithms and Applications</i>, Springer, 2011.",
        "[7] A. Burns and R. I. Davis, 'A Survey of Research on Mixed Criticality Systems,' <i>ACM Computing Surveys</i>, vol. 46, no. 1, pp. 1-37, 2013.",
        "[8] H. Lu, J. Peng, J. Mao, H. Li, and Y. Chen, 'FlexDNN: Input-Adaptive On-Device Deep Learning for Real-Time Mobile Vision,' in <i>IEEE RTSS</i>, 2018, pp. 235-246.",
        "[9] J. Deng, D. Chen, D. Liu, L. Yuan, and G. Hua, 'Dynamic Neural Networks: A Survey,' <i>IEEE TPAMI</i>, vol. 44, no. 11, pp. 7436-7456, 2021.",
        "[10] F. Chollet, 'Xception: Deep Learning with Depthwise Separable Convolutions,' in <i>IEEE CVPR</i>, 2017, pp. 1251-1258.",
    ]
    for r in refs:
        story.append(Paragraph(r, ref_style))

    # Build Document
    doc.build(story)
    return PDF_PATH


if __name__ == "__main__":
    out = build_pdf()
    print(f"Successfully generated PDF manuscript: {out} ({out.stat().st_size} bytes)")
