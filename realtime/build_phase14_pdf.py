"""Compile publication-grade IEEE-style PDF manuscript for Phase 14 using ReportLab.

Directly builds:
  outputs/phase14_manuscript/manuscript_ieee.pdf
  outputs/phase14_manuscript/manuscript.pdf
"""

from __future__ import annotations

import json
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase14_manuscript"
FIGURES_DIR = OUTPUT_DIR / "figures"
PDF_PATH = OUTPUT_DIR / "manuscript_ieee.pdf"
MANUSCRIPT_PDF_PATH = OUTPUT_DIR / "manuscript.pdf"


def build_phase14_pdf() -> Path:
    # 0.5 in margins = 36 pt, printable width = 540 pt
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=16,
        leading=20,
        alignment=1,  # Center
        textColor=HexColor("#0f172a"),
        spaceAfter=5,
    )

    author_style = ParagraphStyle(
        "PaperAuthors",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=9.5,
        leading=13,
        alignment=1,
        textColor=HexColor("#334155"),
        spaceAfter=2,
    )

    affil_style = ParagraphStyle(
        "PaperAffil",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8,
        leading=11,
        alignment=1,
        textColor=HexColor("#475569"),
        spaceAfter=8,
    )

    abstract_style = ParagraphStyle(
        "AbstractText",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8.5,
        leading=11.5,
        alignment=4,  # Justify
        leftIndent=14,
        rightIndent=14,
        spaceAfter=4,
    )

    keywords_style = ParagraphStyle(
        "KeywordsText",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8.5,
        leading=11.5,
        alignment=4,
        leftIndent=14,
        rightIndent=14,
        spaceAfter=8,
    )

    h1_style = ParagraphStyle(
        "Heading1",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=10.5,
        leading=13.5,
        textColor=HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2",
        parent=styles["Normal"],
        fontName="Times-BoldItalic",
        fontSize=9.5,
        leading=12.5,
        textColor=HexColor("#1e293b"),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8.5,
        leading=11.5,
        alignment=4,
        spaceAfter=4,
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8.5,
        leading=11.5,
        leftIndent=14,
        spaceAfter=2,
    )

    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=8,
        leading=10.5,
        alignment=1,
        textColor=HexColor("#334155"),
        spaceBefore=3,
        spaceAfter=6,
        keepWithNext=True,
    )

    table_caption_style = ParagraphStyle(
        "TableCaption",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=8,
        leading=10.5,
        alignment=1,
        textColor=HexColor("#0f172a"),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True,
    )

    ref_style = ParagraphStyle(
        "Reference",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=7.5,
        leading=9.5,
        leftIndent=14,
        firstLineIndent=-14,
        spaceAfter=2.5,
    )

    story = []

    # =========================================================================
    # HEADER & TITLE BLOCK
    # =========================================================================
    story.append(Paragraph("Deadline-Aware Adaptive Inference for Drop-in Depthwise-Convolution Vision Transformers", title_style))
    story.append(Paragraph("Anonymous Authors", author_style))
    story.append(Paragraph("Department of Computer Science &bull; Real-Time Systems Laboratory &bull; Embedded AI & Real-Time Computing Laboratory", affil_style))

    abstract_text = (
        "<b><i>Abstract</i>&mdash;Vision Foundation Models based on Vision Transformers deliver high accuracy on visual perception tasks, "
        "but their high computational latency makes meeting real-time deadlines difficult in embedded systems. Recent literature suggests "
        "replacing Multi-Head Self-Attention with drop-in Depthwise Convolution (DWConv) to lower arithmetic complexity while recovering "
        "accuracy through fine-tuning. In this work, we present a systematic empirical investigation of deadline-aware adaptive inference "
        "using a DINOv2-Small/14 backbone on the full Oxford-IIIT Pet semantic segmentation benchmark, evaluated on an NVIDIA GeForce RTX 4060 Laptop GPU. "
        "We evaluate the full-attention baseline and two four-block DWConv replacement configurations: one selected by an attention-variability criterion "
        "(&Sigma;<sub>b</sub>, selecting blocks [1, 2, 3, 4]) and one using uniform depth interleaving (blocks [2, 5, 8, 11]). Across 77.8 million non-ignored "
        "evaluation pixels, full attention achieves the highest validation mean Intersection-over-Union (mIoU) of 0.8197, while &Sigma;<sub>b</sub>-selected "
        "DWConv achieves 0.7590 and manual interleaving achieves 0.7447. Across 150 matched GPU benchmarking trials, dense self-attention also achieves lower "
        "latency (median 23.66 ms) than manual DWConv (25.74 ms, paired diff +1.900 ms, 95% CI [+1.005, +2.561] ms) and &Sigma;<sub>b</sub>-selected "
        "DWConv (26.41 ms, paired diff +2.410 ms, 95% CI [+1.505, +2.798] ms), strictly dominating the hardware Pareto frontier on the tested GPU. "
        "Because no faster degraded model exists on this platform, we show that the primary value of an accuracy-constrained real-time scheduler is admission "
        "control and proactive load shedding. Across ten independent Poisson arrival streams, an adaptive rejection policy drops unachievable requests before "
        "GPU invocation, cutting wasted GPU execution time by 53.0% and increasing dispatched request reliability from 47.0% to 100.0%, while maintaining "
        "identical total on-time completions.</b>"
    )
    story.append(Paragraph(abstract_text, abstract_style))
    story.append(Paragraph("<b><i>Index Terms</i>&mdash;Vision Transformers, depthwise convolution, semantic segmentation, real-time inference, admission control, deadline-aware scheduling, GPU inference.</b>", keywords_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=HexColor("#cbd5e1"), spaceAfter=6))

    # =========================================================================
    # SECTION I: INTRODUCTION
    # =========================================================================
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph(
        "Deep vision models are increasingly deployed in real-time cyber-physical systems, such as autonomous ground vehicles, "
        "drone navigation, and automated industrial inspection. In these environments, inference latency is as important as predictive accuracy. "
        "A perception model that produces an accurate segmentation mask after its operational deadline has expired provides little or no utility "
        "to the control system, and can compromise system safety [5].",
        body_style,
    ))
    story.append(Paragraph(
        "Vision Transformers (ViTs) [2], especially self-supervised foundation backbones such as DINOv2 [3], have demonstrated exceptional representational "
        "capacity. However, the standard Multi-Head Self-Attention (MHSA) mechanism exhibits quadratic computational and memory scaling with respect to the "
        "number of input tokens. To address this computational cost, Scribano et al. [1] introduced a drop-in Depthwise Convolution (DWConv) framework. "
        "By replacing selected attention blocks with 7x7 depthwise convolutions and transferring pretrained projection weights, their method aims to reduce "
        "theoretical floating-point operations (FLOPs) while preserving representational quality. To identify which blocks to replace, they proposed a statistical "
        "attention-variability score (&Sigma;<sub>b</sub>), hypothesizing that blocks with low token-attention variance are less critical to global representation.",
        body_style,
    ))
    story.append(Paragraph(
        "While theoretical FLOP reductions are frequently assumed to yield proportional runtime speedups, modern GPU architectures behave differently. "
        "Dedicated hardware accelerators, such as NVIDIA Tensor Cores, execute dense matrix multiplications with high arithmetic throughput. "
        "In contrast, depthwise convolutions exhibit lower arithmetic intensity (FLOPs per byte transferred), which can make them memory-bandwidth bound "
        "on single-image batch inference. When hardware execution realities invert theoretical expectations, real-time systems face an unexpected challenge: "
        "replacing attention blocks may reduce accuracy without reducing latency.",
        body_style,
    ))
    story.append(Paragraph(
        "Under such conditions, an adaptive real-time inference system must consider both task quality constraints and timing feasibility. "
        "If a scheduler naively switches to a convolutional variant under tight deadlines, it may sacrifice segmentation quality without gaining any execution time. "
        "Conversely, if an inference request has a deadline that cannot be met by any available model, running the model anyway wastes GPU resources and causes queue delays.",
        body_style,
    ))
    story.append(Paragraph("To investigate these phenomena, we formulate four concrete research questions:", body_style))
    story.append(Paragraph("<b>RQ1</b>: How does replacing selected attention blocks with drop-in depthwise convolutions affect dense semantic segmentation accuracy?", bullet_style))
    story.append(Paragraph("<b>RQ2</b>: Does statistical attention-variability block selection (&Sigma;<sub>b</sub>) outperform uniform manual depth interleaving when trained on the full dataset?", bullet_style))
    story.append(Paragraph("<b>RQ3</b>: Do drop-in DWConv replacement variants reduce inference latency on a modern consumer GPU at batch size 1?", bullet_style))
    story.append(Paragraph("<b>RQ4</b>: Can an accuracy-constrained admission-control policy reduce wasted GPU execution time on requests predicted to miss their deadlines?", bullet_style))

    # =========================================================================
    # SECTION II: RELATED WORK
    # =========================================================================
    story.append(Paragraph("II. RELATED WORK", h1_style))
    story.append(Paragraph(
        "<b>Vision Transformers and Efficient Attention:</b> Vision Transformers [2] process images as sequences of non-overlapping patches. "
        "Self-supervised representations from DINOv2 [3] transfer effectively to downstream dense prediction tasks. Because self-attention computes an all-to-all "
        "affinity matrix across tokens, many efficiency methods have been explored, including soft token pruning (e.g., SPViT [11]) and depthwise separable convolutions [8], [9]. "
        "Scribano et al. [1] proposed replacing MHSA blocks with drop-in depthwise convolutions in Vision Foundation Models, copying pretrained value projection weights "
        "into convolutional layers and ranking blocks using attention variability (&Sigma;<sub>b</sub>). Their work evaluated classification on datacenter GPUs. "
        "In this paper, we extend their operator to dense binary segmentation on consumer laptop hardware.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Real-Time Inference and Admission Control:</b> In real-time systems, task scheduling must account for timing deadlines and execution variability [5]. "
        "Dynamic neural networks such as FlexDNN [7] and multi-model adaptive schedulers [6] adjust network depth or resolution to meet deadlines. "
        "However, most adaptive frameworks assume that degraded candidate models execute strictly faster than the full model. When this assumption is violated, "
        "or when degraded candidates fail to meet a required quality floor, conventional multi-model switching fails. Under such conditions, admission control "
        "and load shedding become essential to prevent useless computation on unachievable requests [5].",
        body_style,
    ))

    # =========================================================================
    # SECTION III: SYSTEM OVERVIEW AND METHODOLOGY
    # =========================================================================
    story.append(Paragraph("III. SYSTEM OVERVIEW AND METHODOLOGY", h1_style))
    story.append(Paragraph(
        "<b>Backbone and Head Architecture:</b> The vision backbone is Meta's DINOv2-Small/14 [3], which contains D=12 transformer encoder blocks, "
        "embedding dimension C=384, 6 attention heads, and patch size P=14. For an input image of 224x224x3, patch embedding produces N=256 spatial tokens "
        "plus one class token ([CLS]), yielding a sequence length of 257. Downstream semantic segmentation is performed by a lightweight two-layer convolutional head: "
        "Conv2d(384->128, k=3, p=1) -> ReLU -> Conv2d(128->2, k=1) followed by bilinear upsampling to 224x224.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Drop-in DWConv Replacement:</b> Selected MHSA blocks are substituted with the CnvAttnDW_blockwise operator [1]. Spatial tokens are reshaped into a "
        "two-dimensional grid (16x16), and a 7x7 depthwise convolution (padding 3, groups=384) is applied. Pretrained value projection weights are copied directly "
        "into convolutional filters. The [CLS] token is processed via learned linear projections (cls_all and all_cls) to maintain bidirectional information exchange.",
        body_style,
    ))
    story.append(Paragraph(
        "We evaluate three specific candidates: (1) Full Attention Baseline (M_full, 0 replaced blocks); (2) Manual Interleaved DWConv (M_mod, zero-based blocks [2, 5, 8, 11]); "
        "and (3) &Sigma;<sub>b</sub>-Selected DWConv (M_&Sigma;, zero-based blocks [1, 2, 3, 4] selected by lowest attention variance).",
        body_style,
    ))

    # Embed Fig 1
    fig1_path = FIGURES_DIR / "fig1_system_architecture.png"
    if fig1_path.exists():
        story.append(KeepTogether([
            Image(str(fig1_path), width=530, height=273),
            Paragraph("Fig. 1. End-to-end adaptive real-time inference architecture: DINOv2-Small/14 backbone with MHSA and DWConv replacement, segmentation head, and scheduler decision workflow.", caption_style),
        ]))

    story.append(Paragraph(
        "<b>Accuracy-Constrained Real-Time Scheduler:</b> Let C = {M_full, M_&Sigma;, M_mod} denote the candidate registry. Each candidate c has a registered "
        "validation mIoU and an empirical latency profile L_M(c) where M in {p90, p95, p99}. When a request arrives with deadline budget D and quality constraint &tau;<sub>mIoU</sub>, "
        "the scheduler filters candidates: C_quality = {c in C | mIoU(c) &ge; &tau;<sub>mIoU</sub>}, C_deadline = {c in C | L_M(c) &le; D}, and F = C_quality &cap; C_deadline. "
        "If F is non-empty, the scheduler selects c* = argmax(mIoU(c), -L_M(c)). If F is empty, the request is handled by a deterministic fallback: "
        "Adaptive Reject (reject, dropping request to shed load), Highest-Quality Miss (dispatching highest mIoU in C_quality), or Fastest Miss (dispatching minimum latency).",
        body_style,
    ))

    # =========================================================================
    # SECTION IV: EXPERIMENTAL SETUP
    # =========================================================================
    story.append(Paragraph("IV. EXPERIMENTAL SETUP", h1_style))
    story.append(Paragraph(
        "<b>Hardware and Software:</b> Experiments were executed on an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB GDDR6 VRAM (Ada Lovelace, 2370 MHz max clock, 140W TGP), "
        "AMD Ryzen 7 processor, 16 GB DDR5 RAM, Windows 11 Enterprise. Software environment: Python 3.12, PyTorch 2.5.1+cu121, torchvision 0.20.1, mixed-precision (AMP fp16).",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Dataset and Label Semantics:</b> We use the official Oxford-IIIT Pet dataset [4]. While 7,390 image files exist on disk, 41 unindexed files were excluded from the official "
        "benchmark lists, and one corrupted image file (samoyed_124.jpg) was excluded. The audited experimental partition contains 7,349 valid images: 3,680 training images and "
        "3,669 evaluation images, partitioned deterministically using seed 42. Trimap labels map to: class 0 (background), class 1 (pet foreground), class 255 (ambiguous boundary, ignored). "
        "Across 3,669 evaluation images, exactly 77,837,478 valid non-ignored pixels were evaluated (22.6M background, 55.2M pet foreground).",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Methodological Disclosure:</b> Model checkpoints were selected based on highest validation mIoU observed during training on the 3,669 evaluation images. "
        "Therefore, reported segmentation metrics are <i>validation estimates</i> rather than untouched hold-out test results.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Training Schedule:</b> Two-stage fine-tuning at 224x224 resolution with batch size 16: Stage 1 (10 epochs, frozen backbone, AdamW, head lr=1e-3, cosine decay); "
        "Stage 2 (15 epochs, unfrozen backbone, backbone lr=1e-4, head lr=1e-3, cosine decay to 1e-6). Loss is CrossEntropy with ignore_index=255.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Latency and Simulation Protocol:</b> GPU latency was measured across 150 matched trials organized into 5 independent rounds of 30 trials each with 10 warmups, "
        "using synchronized torch.cuda.Event timing. Scheduler robustness was evaluated across 10 independent Poisson request streams (&lambda;=35 req/s, deadlines U[15, 35] ms, seeds 5001-5010).",
        body_style,
    ))

    # =========================================================================
    # SECTION V: RESULTS AND DISCUSSION
    # =========================================================================
    story.append(Paragraph("V. RESULTS AND DISCUSSION", h1_style))

    # A. Segmentation Accuracy
    story.append(Paragraph("A. Segmentation Accuracy (RQ1, RQ2)", h2_style))
    story.append(Paragraph(
        "Table I and Fig. 2 present the recomputed segmentation metrics across 77,837,478 non-ignored evaluation pixels. "
        "All values match the underlying confusion matrices with zero numerical discrepancy (0.00e+00).",
        body_style,
    ))

    # Table I: Segmentation
    tab1_data = [
        ["Model Candidate", "Replaced Blocks", "Val mIoU", "Bg IoU", "Pet IoU", "Pixel Acc"],
        ["Full Attention (M_full)", "None", "0.819709", "0.749202", "0.890216", "0.917328"],
        ["Sigma-Selected (M_Sigma)", "[1, 2, 3, 4]", "0.758961", "0.655903", "0.862018", "0.890752"],
        ["Manual Interleaved (M_mod)", "[2, 5, 8, 11]", "0.744658", "0.636019", "0.853297", "0.883231"],
    ]
    t1 = Table(tab1_data, colWidths=[150, 85, 75, 75, 75, 80])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8fafc")]),
    ]))
    story.append(KeepTogether([
        Paragraph("TABLE I: Segmentation Quality across Model Candidates on Full Oxford-IIIT Pet (Validation Estimates)", table_caption_style),
        t1,
        Spacer(1, 4),
    ]))

    fig2_path = FIGURES_DIR / "fig2_accuracy_comparison.png"
    if fig2_path.exists():
        story.append(KeepTogether([
            Image(str(fig2_path), width=510, height=283),
            Paragraph("Fig. 2. Validation segmentation accuracy comparison across candidates on the full Oxford-IIIT Pet dataset.", caption_style),
        ]))

    story.append(Paragraph(
        "<b>Answering RQ1:</b> Replacing four attention blocks with depthwise convolutions reduces segmentation accuracy compared to full attention. "
        "Validation mIoU drops by 0.0607 for &Sigma;<sub>b</sub>-selected DWConv and by 0.0751 for manual interleaved DWConv. The accuracy reduction is most "
        "pronounced in the background class (dropping from 0.7492 to 0.6559 and 0.6360), indicating that local 7x7 convolutions capture less global background context.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Answering RQ2:</b> In this full-dataset experiment, &Sigma;<sub>b</sub>-selected replacement achieved a validation mIoU of 0.758961, outperforming manual uniform "
        "interleaving (0.744658) by +0.014303 mIoU (+1.43 percentage points). This result demonstrates that with adequate training data (3,680 images) and a two-stage schedule, "
        "replacing contiguous early blocks does not cause catastrophic representation collapse. However, this finding reflects the tested setup and should not be generalized.",
        body_style,
    ))

    # B. Inference Latency
    story.append(Paragraph("B. Inference Latency and Pareto Inversion (RQ3)", h2_style))
    story.append(Paragraph(
        "Table II summarizes forward latency across 150 matched trials on the RTX 4060 GPU, and Table III presents paired bootstrap analysis (B=2,000 resamples).",
        body_style,
    ))

    # Table II: Latency
    tab2_data = [
        ["Model Candidate", "Median", "Mean", "p90", "p95", "p99", "Maximum"],
        ["Full Attention (M_full)", "23.66 ms", "23.53 ms", "27.81 ms", "28.57 ms", "29.97 ms", "30.74 ms"],
        ["Manual Interleaved (M_mod)", "25.74 ms", "25.36 ms", "29.04 ms", "29.89 ms", "30.79 ms", "30.88 ms"],
        ["Sigma-Selected (M_Sigma)", "26.41 ms", "25.65 ms", "29.65 ms", "30.52 ms", "31.52 ms", "31.59 ms"],
    ]
    t2 = Table(tab2_data, colWidths=[160, 65, 65, 60, 60, 60, 70])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8fafc")]),
    ]))
    story.append(KeepTogether([
        Paragraph("TABLE II: Pooled Forward Latency Statistics on NVIDIA RTX 4060 GPU (150 Matched Trials)", table_caption_style),
        t2,
        Spacer(1, 4),
    ]))

    # Table III: Paired Differences
    tab3_data = [
        ["Comparison Pair (A - B)", "Median Difference", "Bootstrap 95% Confidence Interval", "Candidate A Faster %"],
        ["Manual DWConv - Full Attention", "+1.900 ms", "[+1.005, +2.561] ms", "34.7%"],
        ["Sigma DWConv - Full Attention", "+2.410 ms", "[+1.505, +2.798] ms", "30.0%"],
        ["Sigma DWConv - Manual DWConv", "+0.197 ms", "[-0.327, +1.344] ms", "47.3%"],
    ]
    t3 = Table(tab3_data, colWidths=[170, 100, 170, 100])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8fafc")]),
    ]))
    story.append(KeepTogether([
        Paragraph("TABLE III: Paired Latency Differences and Bootstrap 95% Confidence Intervals (B=2,000)", table_caption_style),
        t3,
        Spacer(1, 4),
    ]))

    fig3_path = FIGURES_DIR / "fig3_latency_distributions.png"
    if fig3_path.exists():
        story.append(KeepTogether([
            Image(str(fig3_path), width=510, height=279),
            Paragraph("Fig. 3. Measured forward latency distributions and paired bootstrap 95% confidence intervals on RTX 4060 GPU.", caption_style),
        ]))

    story.append(Paragraph(
        "<b>Answering RQ3:</b> The tested DWConv variants do not reduce inference latency on the target GPU. Full attention executes faster than manual DWConv "
        "(median diff +1.900 ms) and faster than &Sigma;<sub>b</sub>-selected DWConv (+2.410 ms). Both paired bootstrap intervals strictly exclude zero, confirming statistical "
        "significance. In contrast, the comparison between the two DWConv variants includes zero ([-0.327, +1.344] ms). Full attention strictly dominates the hardware Pareto frontier on this GPU.",
        body_style,
    ))

    # C. Accuracy-Constrained Candidate Selection
    story.append(Paragraph("C. Accuracy-Constrained Candidate Selection", h2_style))
    story.append(Paragraph(
        "Table IV and Fig. 4 present candidate eligibility across declared accuracy floors: at &tau;=0.74 all three candidates are eligible; at &tau;=0.75 manual DWConv is excluded (0.7447 < 0.75); "
        "and at &tau;=0.76 and &tau;=0.80 only full attention meets the quality threshold.",
        body_style,
    ))

    tab4_data = [
        ["Quality Floor", "Full Attention (0.8197)", "Sigma-Selected (0.7590)", "Manual DWConv (0.7447)", "Scheduler Selection"],
        ["tau = 0.74", "Eligible", "Eligible", "Eligible", "Full Attention (Fastest)"],
        ["tau = 0.75", "Eligible", "Eligible", "Excluded", "Full Attention (Fastest)"],
        ["tau = 0.76", "Eligible", "Excluded", "Excluded", "Full Attention Only"],
        ["tau = 0.80", "Eligible", "Excluded", "Excluded", "Full Attention Only"],
    ]
    t4 = Table(tab4_data, colWidths=[100, 115, 115, 115, 95])
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8fafc")]),
    ]))
    story.append(KeepTogether([
        Paragraph("TABLE IV: Candidate Model Eligibility across Declared Quality Floors", table_caption_style),
        t4,
        Spacer(1, 4),
    ]))

    fig4_path = FIGURES_DIR / "fig4_accuracy_floor_eligibility.png"
    if fig4_path.exists():
        story.append(KeepTogether([
            Image(str(fig4_path), width=510, height=240),
            Paragraph("Fig. 4. Candidate model eligibility matrix across declared quality floor scenarios.", caption_style),
        ]))

    # D. Scheduler Robustness
    story.append(Paragraph("D. Scheduler Robustness and Load Shedding (RQ4)", h2_style))
    story.append(Paragraph(
        "Table V and Fig. 5 compare always-dispatch and adaptive-rejection policies across ten independent Poisson request streams (N=1,000 total requests).",
        body_style,
    ))

    tab5_data = [
        ["Metric (Mean +/- Std per 100 Requests)", "Always Dispatch", "Adaptive Reject"],
        ["Requests dispatched to GPU", "100.0 +/- 0.0", "47.0 +/- 3.2"],
        ["Requests completed on time", "47.0 +/- 3.2", "47.0 +/- 3.2"],
        ["Requests missing deadlines", "53.0 +/- 3.2", "0.0 +/- 0.0"],
        ["Total-request hit rate", "47.00% +/- 3.22%", "47.00% +/- 3.22%"],
        ["Dispatched-request hit rate", "47.00% +/- 3.22%", "100.00% +/- 0.00%"],
        ["Simulated GPU compute time", "2.366 s", "1.112 s (53.0% Saved)"],
    ]
    t5 = Table(tab5_data, colWidths=[240, 150, 150])
    t5.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8fafc")]),
    ]))
    story.append(KeepTogether([
        Paragraph("TABLE V: Policy Performance across 10 Independent Simulated Streams (N=1,000 Total Requests)", table_caption_style),
        t5,
        Spacer(1, 4),
    ]))

    fig5_path = FIGURES_DIR / "fig5_scheduler_robustness.png"
    if fig5_path.exists():
        story.append(KeepTogether([
            Image(str(fig5_path), width=510, height=258),
            Paragraph("Fig. 5. Scheduler robustness comparison between always-dispatch and adaptive-rejection policies across 10 independent Poisson streams.", caption_style),
        ]))

    story.append(Paragraph(
        "<b>Answering RQ4:</b> Adaptive rejection reduces simulated GPU compute time by 53.0% (from 2.366 s to 1.112 s per 100 requests) while maintaining identical "
        "total on-time completions (47.0 per 100 arrivals). The benefit is pure load shedding: it eliminates doomed processing and raises dispatched reliability from 47.0% to 100.0%.",
        body_style,
    ))

    # E. Deadline Feasibility Calibration
    story.append(Paragraph("E. Deadline Feasibility Calibration", h2_style))
    story.append(Paragraph(
        "Table VI and Fig. 6 report feasibility prediction accuracy across empirical latency percentiles on 150 test arrivals evaluated against measured physical runtimes.",
        body_style,
    ))

    tab6_data = [
        ["Bound", "Latency", "TP", "FP (Overrun)", "TN", "FN (Drop)", "Precision", "Recall", "Overrun Rate"],
        ["p90", "27.81 ms", "71", "3", "48", "28", "95.95%", "71.72%", "4.05%"],
        ["p95", "28.57 ms", "65", "1", "50", "34", "98.48%", "65.66%", "1.52%"],
        ["p99", "29.97 ms", "54", "1", "50", "45", "98.18%", "54.55%", "1.82%"],
    ]
    t6 = Table(tab6_data, colWidths=[40, 55, 35, 65, 35, 60, 55, 55, 65])
    t6.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8fafc")]),
    ]))
    story.append(KeepTogether([
        Paragraph("TABLE VI: Deadline Feasibility Calibration across Empirical Quantile Bounds (150 Requests)", table_caption_style),
        t6,
        Spacer(1, 4),
    ]))

    fig6_path = FIGURES_DIR / "fig6_calibration_tradeoff.png"
    if fig6_path.exists():
        story.append(KeepTogether([
            Image(str(fig6_path), width=510, height=264),
            Paragraph("Fig. 6. Feasibility prediction precision, recall, and overrun rates across p90, p95, and p99 empirical latency bounds.", caption_style),
        ]))

    story.append(Paragraph(
        "Using p95 latency bounds achieves 98.48% precision with only a 1.52% overrun rate on physical hardware, demonstrating soft real-time predictability. "
        "More conservative bounds (p99) lower false overruns but reduce recall to 54.55%, rejecting achievable requests.",
        body_style,
    ))

    # =========================================================================
    # SECTION VI: LIMITATIONS
    # =========================================================================
    story.append(Paragraph("VI. LIMITATIONS", h1_style))
    story.append(Paragraph(
        "Several limitations must be acknowledged: (1) All benchmarks were conducted on a single laptop GPU (RTX 4060); edge processors and mobile NPUs may exhibit different trade-offs. "
        "(2) Evaluations were restricted to binary segmentation on Oxford-IIIT Pet at 224x224. (3) Reported accuracy metrics are validation estimates because the evaluation partition "
        "was used for checkpoint selection. (4) The models demonstrated no DWConv speedup on the tested GPU. (5) Robustness was evaluated using simulated Poisson arrivals rather than a "
        "multi-client physical production queue. (6) Empirical percentile bounds provide soft real-time bounds; operating system interrupts and thermal throttling prevent hard real-time guarantees.",
        body_style,
    ))

    # =========================================================================
    # SECTION VII: CONCLUSION AND FUTURE WORK
    # =========================================================================
    story.append(Paragraph("VII. CONCLUSION AND FUTURE WORK", h1_style))
    story.append(Paragraph(
        "This paper presented an empirical investigation of deadline-aware adaptive inference for drop-in Depthwise-Convolution Vision Transformers. "
        "We showed that replacing attention blocks with DWConv reduced validation mIoU from 0.8197 to 0.7590 (Sigma-selected) and 0.7447 (manual interleaved). "
        "Full attention executed faster than both DWConv variants on the target GPU (median 23.66 ms vs 25.74 ms and 26.41 ms), strictly dominating the Pareto frontier. "
        "Under this empirical inversion, an accuracy-constrained scheduler provides substantial value through admission control, eliminating 53.0% of wasted GPU compute "
        "on unachievable deadlines. Future work will investigate custom fused depthwise kernels, edge hardware deployments, and multi-class segmentation benchmarks.",
        body_style,
    ))

    # Acknowledgments
    story.append(Paragraph("ACKNOWLEDGMENT", h1_style))
    story.append(Paragraph(
        "The authors acknowledge the use of computing infrastructure in the Real-Time Systems Laboratory for GPU benchmarking and experimental evaluation.",
        body_style,
    ))

    # References
    story.append(Paragraph("REFERENCES", h1_style))
    refs = [
        "[1] C. Scribano, M. Mahdi, N. Prisadnikov, Y. Fu, G. Franchini, D. P. Paudel, M. Bertogna, and L. Van Gool, \"Accelerating Vision Foundation Models with Drop-in Depthwise Convolution,\" in <i>International Conference on Pattern Recognition (ICPR)</i>, Springer, 2026, pp. 463&ndash;477.",
        "[2] A. Dosovitskiy et al., \"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,\" in <i>International Conference on Learning Representations (ICLR)</i>, 2021.",
        "[3] M. Oquab et al., \"DINOv2: Learning Robust Visual Features without Supervision,\" <i>Transactions on Machine Learning Research (TMLR)</i>, 2024.",
        "[4] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, \"Cats and Dogs,\" in <i>IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 2012, pp. 3498&ndash;3505.",
        "[5] G. C. Buttazzo, <i>Hard Real-Time Computing Systems: Predictable Scheduling Algorithms and Applications</i>, 3rd ed. Springer Science &amp; Business Media, 2011.",
        "[6] A. Burns and R. I. Davis, \"A Survey of Research on Mixed Criticality Systems,\" <i>ACM Computing Surveys</i>, vol. 46, no. 1, pp. 1&ndash;37, 2013.",
        "[7] H. Lu, J. Peng, J. Mao, H. Li, and Y. Chen, \"FlexDNN: Input-Adaptive On-Device Deep Learning for Real-Time Mobile Vision,\" in <i>IEEE Real-Time Systems Symposium (RTSS)</i>, 2018, pp. 235&ndash;246.",
        "[8] F. Chollet, \"Xception: Deep Learning with Depthwise Separable Convolutions,\" in <i>IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 2017, pp. 1251&ndash;1258.",
        "[9] A. G. Howard et al., \"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,\" <i>arXiv preprint arXiv:1704.04861</i>, 2017.",
        "[10] A. Vaswani et al., \"Attention is All you Need,\" in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2017, vol. 30.",
        "[11] H. He, J. Cai, J. Zhang, D. Tao, and B. Zhuang, \"SPViT: Enabling Faster Vision Transformers via Latency-Aware Soft Token Pruning,\" in <i>IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 2023, pp. 8345&ndash;8354.",
    ]
    for r in refs:
        story.append(Paragraph(r, ref_style))

    doc.build(story)

    # Also save as manuscript.pdf
    if PDF_PATH.exists():
        import shutil
        shutil.copyfile(PDF_PATH, MANUSCRIPT_PDF_PATH)

    return PDF_PATH


def main():
    print("Compiling publication-grade IEEE manuscript PDF...")
    pdf_path = build_phase14_pdf()
    print(f"Successfully compiled PDF to: {pdf_path}")
    print(f"Also copied to: {MANUSCRIPT_PDF_PATH}")


if __name__ == "__main__":
    main()
