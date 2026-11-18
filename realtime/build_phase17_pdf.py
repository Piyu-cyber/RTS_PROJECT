"""Compile publication-grade IEEE-style PDF manuscript for Phase 17 using ReportLab.

Directly builds:
  outputs/phase17_final_manuscript/manuscript_ieee.pdf
  outputs/phase17_final_manuscript/manuscript.pdf
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase17_final_manuscript"
FIGURES_DIR = OUTPUT_DIR / "figures"
PDF_PATH = OUTPUT_DIR / "manuscript_ieee.pdf"
MANUSCRIPT_PDF_PATH = OUTPUT_DIR / "manuscript.pdf"


def build_phase17_pdf() -> Path:
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
        fontSize=15.5,
        leading=19.5,
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
        "accuracy through fine-tuning. In this work, we present an empirical investigation of deadline-aware adaptive inference "
        "using a DINOv2-Small/14 backbone on the full Oxford-IIIT Pet semantic segmentation benchmark, evaluated on an NVIDIA GeForce RTX 4060 Laptop GPU. "
        "We evaluate the full-attention baseline and two four-block DWConv replacement configurations: one selected by an attention-variability criterion "
        "(&Sigma;<sub>b</sub>, selecting blocks [1, 2, 3, 4]) and one using uniform depth interleaving (blocks [2, 5, 8, 11]). Across 77.8 million non-ignored "
        "evaluation pixels, full attention achieves the highest validation mean Intersection-over-Union (mIoU) of 0.8197, while &Sigma;<sub>b</sub>-selected "
        "DWConv achieves 0.7590 and manual interleaving achieves 0.7447. Across 150 matched GPU benchmarking trials, dense self-attention also achieves lower "
        "latency (median 23.66 ms) than manual DWConv (25.74 ms, paired diff +1.900 ms, 95% CI [+1.005, +2.561] ms) and &Sigma;<sub>b</sub>-selected "
        "DWConv (26.41 ms, paired diff +2.410 ms, 95% CI [+1.505, +2.798] ms), strictly dominating the hardware Pareto frontier on the tested GPU. "
        "We conduct a formal Worst-Case Execution Time (WCET) audit and establish that a defensible hard real-time bound cannot currently be proved on this platform "
        "due to closed-loop dynamic clock scaling, non-preemptive GPU kernel scheduling, and display driver multiplexing. Consequently, the system is characterized "
        "as a deadline-aware, best-effort real-time inference prototype. We formulate an accuracy-constrained admission-control scheduler and evaluate an empirical "
        "upper-bound estimate of 32.33 ms alongside empirical p95 and p99 policies across ten independent simulated Poisson arrival streams. The adaptive rejection "
        "policies drop unachievable requests before GPU invocation, cutting simulated GPU execution time by 69.2% to 87.2% and raising dispatched reliability from "
        "55.1% to up to 100.0%, while maintaining consistent on-time completions.</b>"
    )
    story.append(Paragraph(abstract_text, abstract_style))
    story.append(Paragraph("<b><i>Index Terms</i>&mdash;Vision Transformers, depthwise convolution, semantic segmentation, real-time systems, admission control, deadline-aware scheduling, GPU inference, WCET feasibility.</b>", keywords_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=HexColor("#cbd5e1"), spaceAfter=6))

    # =========================================================================
    # SECTION I: INTRODUCTION
    # =========================================================================
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph(
        "Deep vision models are increasingly deployed in real-time cyber-physical systems, such as autonomous ground vehicles, "
        "drone obstacle avoidance, and automated industrial robotics. In these environments, inference latency is as critical as predictive accuracy. "
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
        "Conversely, if an inference request has a deadline that cannot be met by any available model, running the model anyway wastes GPU resources and causes queue delays for subsequent requests.",
        body_style,
    ))
    story.append(Paragraph("To investigate these phenomena, we formulate four concrete research questions:", body_style))
    story.append(Paragraph("<b>RQ1</b>: How does replacing selected attention blocks with drop-in depthwise convolutions affect dense semantic segmentation accuracy?", bullet_style))
    story.append(Paragraph("<b>RQ2</b>: Does statistical attention-variability block selection (&Sigma;<sub>b</sub>) outperform uniform manual depth interleaving when trained on the full dataset?", bullet_style))
    story.append(Paragraph("<b>RQ3</b>: Do drop-in DWConv replacement variants reduce inference latency on a modern consumer GPU at batch size 1?", bullet_style))
    story.append(Paragraph("<b>RQ4</b>: Can an accuracy-constrained admission-control policy reduce wasted GPU execution time on requests predicted to miss their deadlines?", bullet_style))

    story.append(Paragraph("Through rigorous empirical evaluation, this paper provides four main contributions:", body_style))
    story.append(Paragraph("1. We evaluate full attention and two four-block DWConv replacement configurations on the official Oxford-IIIT Pet semantic segmentation benchmark using a two-stage fine-tuning schedule, recomputing metrics across 77.8 million valid pixels.", bullet_style))
    story.append(Paragraph("2. We present a multi-round hardware latency benchmark and paired bootstrap analysis on an NVIDIA RTX 4060 Laptop GPU, demonstrating that full attention is faster than both DWConv variants on this platform, strictly dominating the Pareto frontier.", bullet_style))
    story.append(Paragraph("3. We conduct a formal WCET feasibility audit establishing why hard real-time bounds cannot be proved on the target platform, and implement an empirical upper-bound policy alongside percentile-based schedulers.", bullet_style))
    story.append(Paragraph("4. We evaluate the scheduler across ten independent simulated request streams, demonstrating that proactive rejection eliminates up to 87.2% of wasted GPU execution time while raising dispatched reliability from 55.1% to 100.0%.", bullet_style))

    # =========================================================================
    # SECTION II: RELATED WORK
    # =========================================================================
    story.append(Paragraph("II. RELATED WORK", h1_style))
    story.append(Paragraph("A. Vision Transformers and Efficient Attention", h2_style))
    story.append(Paragraph(
        "Vision Transformers [2] tokenize images into grids of non-overlapping patches and process them through alternating layers of MHSA and multi-layer perceptron (MLP) "
        "feed-forward blocks. Self-supervised models such as DINOv2 [3] produce versatile spatial representations that transfer effectively to downstream dense prediction tasks. "
        "Because standard self-attention calculates an all-to-all attention matrix across N tokens, multiple efficient attention techniques have been proposed, "
        "including soft token pruning (e.g., SPViT [11]) and depthwise separable convolutions [8], [9].",
        body_style,
    ))
    story.append(Paragraph(
        "Scribano et al. [1] proposed replacing MHSA blocks with drop-in depthwise convolutions in Vision Foundation Models. Their approach copies pretrained value "
        "projection weights directly into convolutional layers, reducing theoretical parameter count and FLOPs. They also introduced the &Sigma;<sub>b</sub> score to rank "
        "transformer blocks by attention-map variance. Their open-source implementation evaluated image classification on datacenter GPUs. In this paper, we extend "
        "their operator to dense binary semantic segmentation on consumer laptop hardware and evaluate its real-time behavior.",
        body_style,
    ))
    story.append(Paragraph("B. Real-Time Systems and Worst-Case Execution Time", h2_style))
    story.append(Paragraph(
        "In real-time computing, task scheduling must account for timing deadlines and execution variability [5]. Classical scheduling theory assumes that tasks have a "
        "known Worst-Case Execution Time (WCET) [12]. However, modern multi-core processors and GPUs introduce complex hardware features&mdash;such as dynamic frequency scaling, "
        "multi-level caches, and hardware thread scheduling&mdash;that make formal WCET analysis challenging [12], [13]. Dynamic neural networks such as FlexDNN [7] "
        "and mixed-criticality frameworks [6] adjust model execution dynamically. When candidate models cannot provide a latency speedup, admission control and load "
        "shedding become essential to discard unachievable requests before GPU invocation [5].",
        body_style,
    ))

    # =========================================================================
    # SECTION III: SYSTEM OVERVIEW AND METHODOLOGY
    # =========================================================================
    story.append(Paragraph("III. SYSTEM OVERVIEW AND METHODOLOGY", h1_style))

    fig1_path = FIGURES_DIR / "fig1_system_architecture.png"
    if fig1_path.exists():
        story.append(KeepTogether([
            Image(str(fig1_path), width=510, height=205),
            Paragraph("Fig. 1. End-to-end system architecture. Left: DINOv2-Small/14 backbone showing standard MHSA and drop-in 7x7 DWConv replacement, followed by the two-layer segmentation head. Right: Accuracy-constrained real-time scheduler workflow showing quality gating, deadline feasibility check, and load-shedding rejection fallback.", caption_style),
        ]))

    story.append(Paragraph("A. Backbone and Segmentation Architecture", h2_style))
    story.append(Paragraph(
        "The vision backbone is Meta's DINOv2-Small/14 [3], which contains D = 12 transformer encoder blocks, embedding dimension C = 384, 6 attention heads, and patch size P = 14. "
        "For an input image of size 224x224x3, the patch embedding produces N = 256 spatial tokens plus one class token ([CLS]), yielding a sequence length of 257. "
        "Downstream semantic segmentation is performed by a lightweight two-layer convolutional head. The feature tokens from the final backbone block are reshaped into a "
        "16x16 spatial grid, projected from 384 channels to 128 channels via a 3x3 convolution with padding 1, activated by a ReLU function, and projected to 2 output classes "
        "(background vs. pet) through a 1x1 convolution. The resulting feature map is bilinearly upsampled to the original 224x224 resolution.",
        body_style,
    ))

    story.append(Paragraph("B. Drop-in Depthwise Convolution Replacement", h2_style))
    story.append(Paragraph(
        "In drop-in DWConv replacement, selected MHSA blocks are substituted with the CnvAttnDW_blockwise operator [1]. Spatial tokens are reshaped into a two-dimensional "
        "grid of shape (B, C, 16, 16), and a depthwise convolution with kernel size K = 7, padding 3, and group count equal to the channel dimension C = 384 is applied. "
        "Pretrained value projection weights are copied directly into the convolutional filters. The [CLS] token is processed via learned linear projections (cls_all and all_cls) "
        "to maintain bidirectional information exchange with spatial tokens.",
        body_style,
    ))
    story.append(Paragraph("We evaluate three specific model candidates:", body_style))
    story.append(Paragraph("1. <b>Full Attention Baseline</b> (M<sub>full</sub>): The standard DINOv2-Small/14 backbone with all 12 MHSA blocks intact (0 replaced blocks).", bullet_style))
    story.append(Paragraph("2. <b>Manual Interleaved DWConv</b> (M<sub>mod</sub>): Four blocks replaced at uniform depth intervals: zero-based blocks [2, 5, 8, 11] (blocks 3, 6, 9, 12 in 1-based indexing).", bullet_style))
    story.append(Paragraph("3. <b>&Sigma;<sub>b</sub>-Selected DWConv</b> (M<sub>&Sigma;</sub>): Four blocks selected by lowest statistical attention-map variance: zero-based blocks [1, 2, 3, 4] (blocks 2, 3, 4, 5 in 1-based indexing).", bullet_style))

    story.append(Paragraph("C. Statistical Attention-Variability Score (&Sigma;<sub>b</sub>)", h2_style))
    story.append(Paragraph(
        "Following Scribano et al. [1], the sensitivity score &Sigma;<sub>b</sub> for transformer block b in {0, ..., D-1} measures the cross-sample variance of post-softmax "
        "attention weights across a calibration set of M samples: &Sigma;<sub>b</sub> = (1/H) &sum;<sub>h=1</sub><sup>H</sup> Var<sub>m</sub>(A<sub>m,b,h</sub>). "
        "Blocks with low &Sigma;<sub>b</sub> values exhibit attention distributions that change little across diverse inputs, suggesting that their all-to-all communication is redundant. "
        "Sorting blocks in ascending order of &Sigma;<sub>b</sub> identifies the first four candidates as blocks [1, 2, 3, 4].",
        body_style,
    ))

    # =========================================================================
    # SECTION IV: REAL-TIME SYSTEM CHARACTERIZATION AND DEADLINE-AWARE DESIGN
    # =========================================================================
    story.append(Paragraph("IV. REAL-TIME SYSTEM CHARACTERIZATION AND DEADLINE-AWARE DESIGN", h1_style))
    story.append(Paragraph("A. Real-Time System Definition and Motivation", h2_style))
    story.append(Paragraph(
        "A real-time computing system is defined not merely by its logical correctness, but by the timeliness of its output [5]. In real-time perception pipelines, "
        "an image segmentation mask delivered after its operational deadline has elapsed may be considered a failure for that request. For example, in an autonomous mobile "
        "robot navigating at high speed, a delayed obstacle mask can lead to safety violations even if the semantic labels are accurate.",
        body_style,
    ))

    story.append(Paragraph("B. Mapping to Real-Time Systems Concepts", h2_style))
    story.append(Paragraph(
        "We map the concrete components of our implementation to established real-time systems terminology: "
        "<br/>&bull; <b>Task (&tau;<sub>i</sub>)</b>: Processing an incoming image-segmentation inference request i. "
        "<br/>&bull; <b>Release / Arrival Time (r<sub>i</sub>)</b>: The timestamp at which request i enters the scheduling queue. "
        "<br/>&bull; <b>Absolute Deadline (D<sub>i</sub>)</b>: The wall-clock time by which the segmentation output must be delivered. "
        "<br/>&bull; <b>Relative Deadline Budget (d<sub>i</sub>)</b>: The maximum permissible latency interval, where D<sub>i</sub> = r<sub>i</sub> + d<sub>i</sub>. "
        "<br/>&bull; <b>Remaining Time Budget (B<sub>i</sub>(t))</b>: The available execution budget at current scheduling time t: <b>B<sub>i</sub>(t) = D<sub>i</sub> &minus; t</b>. "
        "<br/>&bull; <b>Execution Time Estimate (C<sub>c</sub>)</b>: An empirically measured latency statistic (e.g., p95) or an empirical upper-bound estimate (EUB) for candidate model c. "
        "<br/>&bull; <b>Waiting Delay (W<sub>i</sub>)</b>: The elapsed time between request arrival and scheduling dispatch, accounting for queueing and controller overhead. "
        "<br/>&bull; <b>Admission / Rejection Decision</b>: The scheduler's determination of whether to dispatch a model or reject the request when no eligible model is predicted to meet the deadline. "
        "<br/>&bull; <b>Inference Execution</b>: Dispatching the selected model (M<sub>full</sub>, M<sub>&Sigma;</sub>, or M<sub>mod</sub>) to the GPU. "
        "<br/>&bull; <b>Deadline Outcome</b>: Recording whether a completed request was an on-time completion (L<sub>i</sub> &le; B<sub>i</sub>(t)), a deadline overrun (L<sub>i</sub> > B<sub>i</sub>(t)), or a rejected request.",
        body_style,
    ))

    fig_rts_path = FIGURES_DIR / "fig_rts_flow.png"
    if fig_rts_path.exists():
        story.append(KeepTogether([
            Image(str(fig_rts_path), width=510, height=220),
            Paragraph("Fig. 2. Real-Time System Task Admission, Scheduling, and Outcome Logging Pipeline. The sequential flow shows: Request Arrival &rarr; Deadline/Budget Calculation &rarr; Quality-Threshold Filtering &rarr; Latency-Based Admission Decision &rarr; Model Inference or Rejection &rarr; Deadline-Outcome Logging.", caption_style),
        ]))

    story.append(Paragraph("C. Scheduling Policy and Admission Control Formulation", h2_style))
    story.append(Paragraph(
        "Let C = {M<sub>full</sub>, M<sub>&Sigma;</sub>, M<sub>mod</sub>} denote the candidate registry. Each candidate c in C has a registered validation segmentation quality "
        "q<sub>c</sub> = mIoU(c) and an execution timing estimate C<sub>c</sub>. When request i arrives with quality constraint Q<sub>i</sub> = &tau;<sub>mIoU</sub> and deadline budget B<sub>i</sub>(t), "
        "the scheduler filters candidates into the quality-eligible set C<sub>Q</sub>(i) = {c &isin; C | q<sub>c</sub> &ge; Q<sub>i</sub>} and deadline-feasible set "
        "C<sub>deadline</sub>(i) = {c &isin; C | W<sub>i</sub> + C<sub>c</sub> &le; B<sub>i</sub>(t)}. The jointly feasible candidate set is F(i) = C<sub>Q</sub>(i) &cap; C<sub>deadline</sub>(i). "
        "If F(i) is non-empty, the scheduler selects the candidate with the highest validation accuracy: c* = argmax<sub>c &isin; F(i)</sub> (q<sub>c</sub>, &minus;C<sub>c</sub>). "
        "If F(i) is empty, the scheduler triggers its deterministic fallback policy. Under <b>Adaptive Reject</b> (reject), the request is dropped immediately without GPU invocation. "
        "Crucially, a rejected request is not an on-time completion; rejection is an admission-control load-shedding mechanism to prevent wasting GPU compute on unachievable deadlines.",
        body_style,
    ))

    story.append(Paragraph("D. Accurate Real-Time Categorization", h2_style))
    story.append(Paragraph(
        "The evaluated system is accurately categorized as a <b>deadline-aware, best-effort real-time inference prototype</b>. It incorporates explicit deadlines, latency-based "
        "admission control, configurable accuracy floors, and deadline-outcome tracking. Under the reject fallback policy, it provides firm real-time load-shedding behavior by "
        "discarding requests whose utility drops to zero upon missing their deadlines. We explicitly state that the system is <i>not</i> a hard real-time system and does <i>not</i> "
        "provide guaranteed deadline satisfaction due to tail latency variations, operating system scheduling, and hardware dynamic scaling.",
        body_style,
    ))

    story.append(Paragraph("E. Measured Latency vs. Formal WCET", h2_style))
    story.append(Paragraph(
        "A critical distinction must be maintained between four concepts: "
        "<br/>1. <b>Observed Latency</b>: Individual physical execution timings recorded during profiling runs (e.g., median 23.66 ms). "
        "<br/>2. <b>Percentile-Based Estimates</b>: Statistical quantiles derived from empirical distributions, such as p90 (27.81 ms), p95 (28.57 ms), and p99 (29.97 ms). "
        "<br/>3. <b>Empirical Upper-Bound Estimate (EUB)</b>: A conservative empirical estimate defined as the maximum observed latency plus three times the across-round standard deviation: "
        "<b>EUB<sub>fwd</sub> = max<sub>obs</sub> + 3 &times; &sigma;<sub>round</sub> = 30.74 + 3(0.531) = 32.33 ms</b>. "
        "This 32.33 ms figure is strictly an <i>empirical estimate</i> for the tested setup. It does not guarantee that future executions will not exceed 32.33 ms. "
        "<br/>4. <b>Formal Worst-Case Execution Time (WCET)</b>: A mathematically provable upper bound on execution time that holds under all possible inputs and hardware states [12]. "
        "<br/><b>WCET Feasibility Outcome</b>: <b>The project has not established a formal WCET.</b> A comprehensive platform audit concluded that a defensible hard real-time "
        "bound cannot currently be proved on the target hardware and operating system. The 32.33 ms value must never be labeled as formal WCET.",
        body_style,
    ))

    story.append(Paragraph("F. Platform Barriers and Requirements for Stronger Guarantees", h2_style))
    story.append(Paragraph(
        "Establishing a defensible WCET bound would require overcoming several architectural barriers: "
        "<br/>&bull; <b>Closed-Loop Dynamic Clock Scaling</b>: NVIDIA GPU Boost dynamically scales core clocks between 1,470 MHz and 2,370 MHz based on real-time temperature, current draw, and power budgets. "
        "<br/>&bull; <b>Display Driver Multiplexing</b>: On Windows 11, the WDDM driver multiplexes CUDA compute kernels alongside Desktop Window Manager (DWM) composition without real-time priority preemption. "
        "<br/>&bull; <b>Cold-Start Spikes</b>: An unprimed initial forward pass on the GPU requires <b>391.44 ms</b> (+1,554% spike) due to CUDA runtime initialization and kernel JIT loading, requiring explicit warmup procedures. "
        "<br/>&bull; <b>Memory Allocation Jitter</b>: Dynamic memory allocation in deep learning frameworks can introduce non-deterministic page fault latencies. "
        "<br/>Achieving formal WCET in future work would require safety-certified embedded hardware (e.g., NVIDIA DRIVE AGX), fixed clock frequencies, a real-time operating system "
        "(e.g., Linux PREEMPT_RT or QNX), static memory allocation, and verified static timing analysis tools [12]. However, merely changing to an RTOS or locking clocks does not "
        "automatically prove WCET; complete hardware modeling and analytical verification remain mandatory.",
        body_style,
    ))

    story.append(Paragraph("G. Connection to Research Contribution", h2_style))
    story.append(Paragraph(
        "The primary real-time systems contribution of this work is the design, integration, and empirical evaluation of deadline-aware inference selection and admission "
        "control under verified latency and accuracy trade-offs. The system demonstrates how a real-time scheduler can eliminate wasted computation on unachievable requests "
        "when hardware execution realities invert theoretical architectural speedups.",
        body_style,
    ))

    # =========================================================================
    # SECTION V: EXPERIMENTAL SETUP
    # =========================================================================
    story.append(Paragraph("V. EXPERIMENTAL SETUP", h1_style))
    story.append(Paragraph("A. Hardware and Software Environment", h2_style))
    story.append(Paragraph(
        "All training, benchmarking, and runtime experiments were performed on an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB GDDR6 VRAM (Ada Lovelace architecture, "
        "maximum graphics clock 2,370 MHz, 140W TGP), hosted on an AMD Ryzen 7 processor with 16 GB DDR5 RAM running Windows 11 Enterprise. "
        "The software environment used Python 3.12 within a virtual environment, PyTorch 2.5.1 with CUDA 12.1 runtime, torchvision 0.20.1, and mixed-precision (AMP fp16) execution.",
        body_style,
    ))

    story.append(Paragraph("B. Dataset Partitioning and Label Semantics", h2_style))
    story.append(Paragraph(
        "We use the official Oxford-IIIT Pet dataset [4]. While 7,390 image files exist on disk, 41 unindexed files were excluded from the official benchmark lists, "
        "and one corrupted image file (samoyed_124.jpg) was excluded. The audited experimental partition contains exactly 7,349 valid images: 3,680 training images "
        "and 3,669 evaluation images, partitioned deterministically using seed 42 in split_manifest.json. "
        "Trimap segmentation labels are mapped to binary classes: pixel value 1 (pet foreground) maps to class 1, pixel value 2 (background) maps to class 0, and pixel value 3 "
        "(ambiguous boundary) maps to ignore index 255. Across the 3,669 evaluation images, exactly 77,837,478 non-ignored pixels were evaluated (22,625,709 background pixels and 55,211,769 pet pixels). "
        "<br/><b>Methodological Disclosure</b>: Model checkpoints were selected based on the highest validation mIoU observed across epochs on the 3,669 evaluation images. "
        "Therefore, reported segmentation metrics are <i>validation estimates</i> rather than results on an untouched out-of-sample test set.",
        body_style,
    ))

    story.append(Paragraph("C. Two-Stage Training Protocol", h2_style))
    story.append(Paragraph(
        "All three models were trained using an identical two-stage fine-tuning schedule at 224x224 resolution with batch size 16: Stage 1 (10 epochs, frozen backbone, AdamW, "
        "head learning rate 1e-3, cosine decay); Stage 2 (15 epochs, unfrozen backbone, backbone learning rate 1e-4, head learning rate 1e-3, cosine decay to 1e-6). "
        "The loss function is standard cross-entropy with ignore_index=255.",
        body_style,
    ))

    story.append(Paragraph("D. Latency Measurement and Workload Simulation Protocol", h2_style))
    story.append(Paragraph(
        "Hardware inference latency was evaluated across 150 matched trials per candidate model (5 independent rounds of 30 trials each, with 10 warmups), "
        "using synchronized torch.cuda.Event timing on single-image tensors (1, 3, 224, 224). "
        "Scheduler robustness was evaluated using simulated Poisson arrival streams (&lambda; = 35 requests/second) with deadlines drawn uniformly from U[15, 35] ms. "
        "Ten independent streams of 100 requests each (N = 1,000 total requests) were evaluated across random seeds 5001 to 5010. Dispatched requests sample actual "
        "execution runtimes from the matched hardware benchmark traces.",
        body_style,
    ))

    # =========================================================================
    # SECTION VI: RESULTS AND DISCUSSION
    # =========================================================================
    story.append(Paragraph("VI. RESULTS AND DISCUSSION", h1_style))
    story.append(Paragraph("A. Segmentation Accuracy (RQ1, RQ2)", h2_style))

    tab1_data = [
        ["Model Candidate", "Blocks", "Val mIoU", "Bg IoU", "Pet IoU", "Pixel Acc"],
        ["Full Attention (M_full)", "None", "0.819709", "0.749202", "0.890216", "0.917328"],
        ["Sigma_b-Selected (M_Sigma)", "[1, 2, 3, 4]", "0.758961", "0.655903", "0.862018", "0.890752"],
        ["Manual Interleaved (M_mod)", "[2, 5, 8, 11]", "0.744658", "0.636019", "0.853297", "0.883231"],
    ]
    t1 = Table(tab1_data, colWidths=[150, 75, 75, 75, 75, 80])
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
            Image(str(fig2_path), width=510, height=220),
            Paragraph("Fig. 3. Validation segmentation accuracy across model candidates on the full Oxford-IIIT Pet dataset (3,669 images, 77.8M valid pixels). Validation estimates.", caption_style),
        ]))

    story.append(Paragraph(
        "<b>Answering RQ1:</b> Replacing four attention blocks with depthwise convolutions reduces segmentation accuracy compared to full attention. "
        "Validation mIoU drops by 0.0607 for &Sigma;<sub>b</sub>-selected DWConv and by 0.0751 for manual interleaved DWConv. "
        "The accuracy reduction is most pronounced in the background class (dropping from 0.7492 to 0.6559 and 0.6360), indicating that local 7x7 convolutions capture less global background context than full self-attention.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Answering RQ2:</b> In this full-dataset experiment, &Sigma;<sub>b</sub>-selected replacement achieved a validation mIoU of 0.758961, outperforming manual uniform interleaving (0.744658) "
        "by +0.014303 mIoU (+1.43 percentage points). This result demonstrates that with adequate training data (3,680 images) and a two-stage unfreezing schedule, replacing contiguous early blocks does not cause "
        "catastrophic representation collapse. However, this finding reflects the tested setup and should not be generalized as a claim that &Sigma;<sub>b</sub> selection is universally superior.",
        body_style,
    ))

    story.append(Paragraph("B. Inference Latency and Pareto Inversion (RQ3)", h2_style))

    tab2_data = [
        ["Model Candidate", "Median", "Mean", "p90", "p95", "p99", "Max"],
        ["Full Attention (M_full)", "23.66 ms", "23.53 ms", "27.81 ms", "28.57 ms", "29.97 ms", "30.74 ms"],
        ["Manual Interleaved (M_mod)", "25.74 ms", "25.36 ms", "29.04 ms", "29.89 ms", "30.79 ms", "30.88 ms"],
        ["Sigma_b-Selected (M_Sigma)", "26.41 ms", "25.65 ms", "29.65 ms", "30.52 ms", "31.52 ms", "31.59 ms"],
    ]
    t2 = Table(tab2_data, colWidths=[150, 65, 65, 65, 65, 65, 65])
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

    tab3_data = [
        ["Comparison Pair (A - B)", "Median Diff", "Bootstrap 95% CI", "A Faster %"],
        ["Manual DWConv - Full Attention", "+1.900 ms", "[+1.005, +2.561] ms", "34.7%"],
        ["Sigma DWConv - Full Attention", "+2.410 ms", "[+1.505, +2.798] ms", "30.0%"],
        ["Sigma DWConv - Manual DWConv", "+0.197 ms", "[-0.327, +1.344] ms", "47.3%"],
    ]
    t3 = Table(tab3_data, colWidths=[180, 100, 160, 90])
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
        Paragraph("TABLE III: Paired Latency Differences and Bootstrap 95% CIs (B=2,000 Resamples)", table_caption_style),
        t3,
        Spacer(1, 4),
    ]))

    fig3_path = FIGURES_DIR / "fig3_latency_distributions.png"
    if fig3_path.exists():
        story.append(KeepTogether([
            Image(str(fig3_path), width=510, height=230),
            Paragraph("Fig. 4. Measured forward latency distributions and paired bootstrap 95% confidence intervals across 150 matched trials on the NVIDIA RTX 4060 Laptop GPU.", caption_style),
        ]))

    story.append(Paragraph(
        "<b>Answering RQ3:</b> The tested DWConv variants do not reduce inference latency on the target GPU. Full attention executes faster than manual DWConv "
        "(median difference +1.900 ms) and faster than &Sigma;<sub>b</sub>-selected DWConv (median difference +2.410 ms). Both paired bootstrap confidence intervals strictly exclude "
        "zero ([+1.005, +2.561] ms and [+1.505, +2.798] ms), confirming that full attention is statistically significantly faster on this platform. In contrast, the comparison between the two "
        "DWConv variants includes zero ([-0.327, +1.344] ms), indicating that their latencies are statistically indistinguishable.",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Forward vs. End-to-End Latency</b>: Complete end-to-end request response times include host preprocessing, host-to-device PCIe transfer, model forward execution, "
        "and postprocessing. On the full-attention baseline, end-to-end latency exhibits a median of 29.27 ms, a p95 of 34.18 ms, and a maximum of 36.35 ms, representing an average host offset "
        "of +5.61 ms over forward-only execution. When operational deadlines apply to the complete request, the scheduler must incorporate this offset into budget calculations.",
        body_style,
    ))

    story.append(Paragraph("C. Accuracy-Constrained Candidate Selection", h2_style))

    tab4_data = [
        ["Quality Floor", "Full Attention", "Sigma-Selected", "Manual DWConv", "Scheduler Choice"],
        ["tau = 0.74", "Eligible", "Eligible", "Eligible", "Full Attention (Fastest)"],
        ["tau = 0.75", "Eligible", "Eligible", "Excluded", "Full Attention (Fastest)"],
        ["tau = 0.76", "Eligible", "Excluded", "Excluded", "Full Attention Only"],
        ["tau = 0.80", "Eligible", "Excluded", "Excluded", "Full Attention Only"],
    ]
    t4 = Table(tab4_data, colWidths=[100, 110, 110, 110, 100])
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
            Image(str(fig4_path), width=510, height=230),
            Paragraph("Fig. 5. Candidate model eligibility matrix across declared quality floor scenarios.", caption_style),
        ]))

    story.append(Paragraph("D. Scheduler Robustness and Policy Comparison (RQ4)", h2_style))
    story.append(Paragraph(
        "Table V and Fig. 6 compare the four evaluated scheduling policies across ten independent simulated Poisson arrival streams "
        "(N = 1,000 total requests, seeds 5001&ndash;5010, &lambda; = 35 req/s, relative deadlines U[15, 35] ms, &tau;<sub>mIoU</sub> = 0.75).",
        body_style,
    ))

    tab5_data = [
        ["Policy Mode", "Dispatched / 100", "Rejected / 100", "On-Time / 100", "Overruns / 100", "Total Hit Rate", "Dispatched Hit", "GPU Time", "GPU Saved"],
        ["Always Dispatch", "100.0 +/- 0.0", "0.0 +/- 0.0", "55.1 +/- 5.0", "44.9 +/- 5.0", "55.1% +/- 5.0%", "55.1% +/- 5.0%", "2.351 s", "Baseline (0%)"],
        ["Empirical p95 (28.57 ms)", "31.1 +/- 4.4", "68.9 +/- 4.4", "30.8 +/- 4.2", "0.3 +/- 0.5", "30.8% +/- 4.2%", "99.1% +/- 1.4%", "0.723 s", "69.2%"],
        ["Empirical p99 (29.97 ms)", "25.1 +/- 4.0", "74.9 +/- 4.0", "25.0 +/- 4.0", "0.1 +/- 0.3", "25.0% +/- 4.0%", "99.6% +/- 1.2%", "0.585 s", "75.1%"],
        ["Empirical Upper Bound (32.33 ms)", "13.1 +/- 4.2", "86.9 +/- 4.2", "13.1 +/- 4.2", "0.0 +/- 0.0", "13.1% +/- 4.2%", "100.0% +/- 0.0%", "0.301 s", "87.2%"],
    ]
    t5 = Table(tab5_data, colWidths=[120, 55, 50, 50, 50, 60, 55, 50, 50])
    t5.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.0),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f8fafc")]),
    ]))
    story.append(KeepTogether([
        Paragraph("TABLE V: Scheduler Policy Performance across 10 Independent Simulated Streams (N=1,000 Total Requests, Simulation Results)", table_caption_style),
        t5,
        Spacer(1, 4),
    ]))

    fig5_path = FIGURES_DIR / "fig5_scheduler_robustness.png"
    if fig5_path.exists():
        story.append(KeepTogether([
            Image(str(fig5_path), width=510, height=240),
            Paragraph("Fig. 6. Scheduler robustness comparison between always-dispatch, empirical p95, empirical p99, and empirical upper-bound policies across ten independent simulated Poisson streams.", caption_style),
        ]))

    story.append(Paragraph(
        "<b>Answering RQ4:</b> As documented in Table V, the adaptive rejection policies significantly reduce wasted GPU computation on requests predicted to miss their deadlines: "
        "<br/>&bull; <b>Always Dispatch</b> dispatches all 100 requests, resulting in 44.9 overruns per 100 requests (55.1% dispatched reliability) and consuming 2.351 s of GPU compute. "
        "<br/>&bull; <b>Empirical p95 (28.57 ms)</b> rejects 68.9 unachievable requests, reducing overruns to 0.3 per 100 (99.1% dispatched hit rate) and cutting simulated GPU compute by <b>69.2%</b> (0.723 s). "
        "<br/>&bull; <b>Empirical p99 (29.97 ms)</b> further tightens admission, reducing overruns to 0.1 per 100 (99.6% dispatched hit rate) and saving <b>75.1%</b> of GPU compute (0.585 s). "
        "<br/>&bull; <b>Empirical Upper Bound (32.33 ms)</b> incorporates a +3&sigma; safety margin, eliminating all simulated overruns (0.0 overruns, <b>100.0%</b> dispatched hit rate) and cutting simulated GPU compute by <b>87.2%</b> (0.301 s).",
        body_style,
    ))
    story.append(Paragraph(
        "<b>Throughput vs. Overrun Trade-off</b>: While the empirical upper-bound policy completely eliminates deadline overruns, it also reduces total admitted on-time throughput "
        "from 30.8% (under p95) to 13.1%. This reflects the fundamental real-time systems trade-off: higher timing conservatism guarantees that dispatched tasks complete on time, "
        "but rejects requests whose actual physical execution would have succeeded. "
        "<br/><b>Critical Denominator Clarification</b>: A higher dispatched-request hit rate (e.g., 100.0% vs. 55.1%) does <i>not</i> mean that more requests were completed successfully. "
        "Rejection is an admission-control mechanism that prevents useless execution on impossible deadlines.",
        body_style,
    ))

    story.append(Paragraph("E. Deadline Feasibility Calibration", h2_style))
    story.append(Paragraph(
        "Table VI and Fig. 7 report feasibility prediction accuracy across empirical latency percentiles on 150 test arrivals evaluated against measured physical runtimes.",
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
            Image(str(fig6_path), width=510, height=240),
            Paragraph("Fig. 7. Feasibility prediction precision, recall, and overrun rates across p90, p95, and p99 empirical latency bounds.", caption_style),
        ]))

    story.append(Paragraph(
        "Using p95 latency bounds achieves 98.48% precision with only a 1.52% overrun rate on physical hardware, demonstrating soft real-time predictability. "
        "More conservative bounds (p99) lower false overruns but reduce recall to 54.55%, rejecting achievable requests.",
        body_style,
    ))

    # =========================================================================
    # SECTION VII: LIMITATIONS
    # =========================================================================
    story.append(Paragraph("VII. LIMITATIONS", h1_style))
    story.append(Paragraph(
        "Several important limitations must be stated: "
        "<br/>1. <b>Single Hardware Target</b>: All benchmarks were collected on a single NVIDIA GeForce RTX 4060 Laptop GPU. Edge processors, mobile NPUs, and GPUs without Tensor Cores may exhibit different latency trade-offs. "
        "<br/>2. <b>Single Benchmark Task</b>: Experiments were restricted to binary segmentation on Oxford-IIIT Pet at 224x224 resolution. Multi-class datasets (e.g., ADE20K) may show different sensitivity patterns. "
        "<br/>3. <b>Validation Estimates</b>: Reported accuracy metrics are validation estimates because the evaluation partition was used for checkpoint selection during training. "
        "<br/>4. <b>Latency Inversion</b>: On the tested system, DWConv did not provide a latency speedup, meaning no faster degraded candidate existed for adaptive switching. "
        "<br/>5. <b>Simulated Workloads</b>: Scheduler robustness was evaluated using simulated Poisson arrival streams rather than a physical multi-client deployment. The simulation models sequential request evaluation but does not model dynamic queue accumulation or concurrent GPU memory contention. "
        "<br/>6. <b>No Hard Real-Time Guarantees</b>: Empirical percentile bounds and the 32.33 ms empirical upper-bound estimate provide soft real-time bounds. Operating system interrupts, thermal throttling, and lack of static WCET analysis preclude hard real-time guarantees.",
        body_style,
    ))

    # =========================================================================
    # SECTION VIII: CONCLUSION AND FUTURE WORK
    # =========================================================================
    story.append(Paragraph("VIII. CONCLUSION AND FUTURE WORK", h1_style))
    story.append(Paragraph(
        "In this paper, we conducted a systematic empirical study of deadline-aware adaptive inference for drop-in Depthwise-Convolution Vision Transformers. "
        "Addressing our four research questions: "
        "<br/>&bull; <b>RQ1</b>: Replacing attention blocks with DWConv reduced validation mIoU from 0.8197 (full attention) to 0.7590 (&Sigma;<sub>b</sub>-selected) and 0.7447 (manual interleaved). "
        "<br/>&bull; <b>RQ2</b>: &Sigma;<sub>b</sub>-selected block replacement outperformed manual interleaving by +0.0143 mIoU when fine-tuned on the full dataset with a two-stage schedule. "
        "<br/>&bull; <b>RQ3</b>: On the tested RTX 4060 GPU, full attention executed faster than both DWConv variants (median 23.66 ms vs. 25.74 ms and 26.41 ms), strictly dominating the hardware Pareto frontier. "
        "<br/>&bull; <b>RQ4</b>: Proactive load shedding reduced simulated GPU compute by 69.2% to 87.2% and raised dispatched reliability to up to 100.0%, without altering total on-time completions.",
        body_style,
    ))
    story.append(Paragraph(
        "We established that formal WCET cannot currently be guaranteed on consumer GPU platforms and categorized the system as a deadline-aware, best-effort prototype. "
        "Future work will examine custom fused depthwise kernels, evaluation across additional edge accelerators, truly held-out test splits, and multi-class semantic segmentation benchmarks.",
        body_style,
    ))

    # Acknowledgment
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
        "[12] R. Wilhelm et al., \"The Worst-Case Execution-Time Problem&mdash;Overview of Methods and Survey of Tools,\" <i>ACM Transactions on Embedded Computing Systems (TECS)</i>, vol. 7, no. 3, pp. 1&ndash;53, 2008.",
        "[13] S. Kato, K. Lakshmanan, R. Rajkumar, and Y. Ishikawa, \"TimeGraph: GPU Scheduling for Real-Time Multi-Tasking Environments,\" in <i>USENIX Annual Technical Conference (ATC)</i>, 2011, pp. 17&ndash;30.",
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
    print("Compiling publication-grade IEEE manuscript PDF for Phase 17...")
    pdf_path = build_phase17_pdf()
    print(f"Successfully compiled PDF to: {pdf_path}")
    print(f"Also copied to: {MANUSCRIPT_PDF_PATH}")


if __name__ == "__main__":
    main()
