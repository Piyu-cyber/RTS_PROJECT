"""Generate high-resolution, visually polished publication-grade diagrams for Phase 11.

Diagrams produced:
  1. arch_system_overview.png: High-clarity vector-like diagram contrasting Base Paper vs Our Contribution.
  2. arch_block_replacement.png: Clean schematic of Multi-Head Self-Attention vs Drop-in Depthwise Convolution.
  3. arch_controller_workflow.png: Clear flowchart of the Accuracy-Constrained Real-Time Scheduler.
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase11_manuscript"


def get_font(size: int, bold: bool = False):
    try:
        font_path = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def render_block_replacement_diagram() -> Path:
    w, h = 1100, 420
    img = Image.new("RGB", (w, h), color="#ffffff")
    draw = ImageDraw.Draw(img)

    f_title = get_font(18, bold=True)
    f_sub = get_font(12, bold=False)
    f_box = get_font(13, bold=True)
    f_txt = get_font(11, bold=False)

    # Title Banner
    draw.rectangle([0, 0, w, 40], fill="#0f172a")
    draw.text((25, 10), "TRANSFORMER ATTENTION VS. DROP-IN DEPTHWISE CONVOLUTION REPLACEMENT", fill="#ffffff", font=f_title)

    # Panel 1: Multi-Head Self-Attention (Left)
    p1_x0, p1_y0, p1_x1, p1_y1 = 30, 60, 520, 390
    draw.rectangle([p1_x0, p1_y0, p1_x1, p1_y1], fill="#f8fafc", outline="#cbd5e1", width=2)
    draw.rectangle([p1_x0, p1_y0, p1_x1, p1_y0 + 30], fill="#e2e8f0")
    draw.text((p1_x0 + 15, p1_y0 + 7), "Standard Multi-Head Self-Attention (MHSA)", fill="#1e293b", font=f_box)

    # Input Tokens X
    draw.rectangle([180, 105, 370, 135], fill="#ffffff", outline="#64748b", width=1)
    draw.text((215, 112), "Input Tokens X (N x C)", fill="#0f172a", font=f_txt)

    # Arrow Down to QKV
    draw.line([(275, 135), (275, 155)], fill="#64748b", width=2)

    # Q, K, V Projections
    draw.rectangle([80, 155, 180, 190], fill="#e0f2fe", outline="#0284c7", width=1)
    draw.text((105, 165), "Query Q", fill="#0369a1", font=f_txt)
    draw.rectangle([225, 155, 325, 190], fill="#e0f2fe", outline="#0284c7", width=1)
    draw.text((255, 165), "Key K", fill="#0369a1", font=f_txt)
    draw.rectangle([370, 155, 470, 190], fill="#e0f2fe", outline="#0284c7", width=1)
    draw.text((400, 165), "Value V", fill="#0369a1", font=f_txt)

    # Lines to Attention Matrix
    draw.line([(130, 190), (220, 220)], fill="#64748b", width=2)
    draw.line([(275, 190), (275, 220)], fill="#64748b", width=2)

    # Softmax(QK^T / sqrt(d))
    draw.rectangle([160, 220, 390, 260], fill="#fef3c7", outline="#d97706", width=2)
    draw.text((178, 232), "Attention A = Softmax(QK^T / sqrt(d))", fill="#92400e", font=f_txt)

    # Multiply with V
    draw.line([(275, 260), (275, 290)], fill="#64748b", width=2)
    draw.line([(420, 190), (420, 275), (320, 290)], fill="#64748b", width=2)

    draw.rectangle([180, 290, 370, 325], fill="#e0f2fe", outline="#0284c7", width=1)
    draw.text((220, 300), "Matrix Product: A x V", fill="#0369a1", font=f_txt)

    # Out Projection
    draw.line([(275, 325), (275, 345)], fill="#64748b", width=2)
    draw.rectangle([180, 345, 370, 375], fill="#0284c7")
    draw.text((215, 353), "Linear Projection (W_o)", fill="#ffffff", font=f_box)

    # Complexity Note
    draw.text((60, 395), "Complexity: O(N^2 x C) Global Interaction | Full KV Memory Footprint", fill="#dc2626", font=f_sub)

    # Panel 2: Drop-in Depthwise Convolution (Right)
    p2_x0, p2_y0, p2_x1, p2_y1 = 560, 60, 1070, 390
    draw.rectangle([p2_x0, p2_y0, p2_x1, p2_y1], fill="#f8fafc", outline="#cbd5e1", width=2)
    draw.rectangle([p2_x0, p2_y0, p2_x1, p2_y0 + 30], fill="#dcfce7")
    draw.text((p2_x0 + 15, p2_y0 + 7), "Drop-in Depthwise Convolution (CnvAttnDW_blockwise)", fill="#166534", font=f_box)

    # Input Tokens X
    draw.rectangle([720, 105, 910, 135], fill="#ffffff", outline="#64748b", width=1)
    draw.text((755, 112), "Input Tokens X (N x C)", fill="#0f172a", font=f_txt)

    # Arrow Down
    draw.line([(815, 135), (815, 160)], fill="#64748b", width=2)

    # Value Projection (Pretrained Weight Copy)
    draw.rectangle([700, 160, 930, 195], fill="#dcfce7", outline="#16a34a", width=2)
    draw.text((720, 170), "Linear Projection (V_proj.weight Copy)", fill="#15803d", font=f_txt)

    # Arrow Down
    draw.line([(815, 195), (815, 220)], fill="#64748b", width=2)

    # Reshape to 2D Spatial Grid
    draw.rectangle([710, 220, 920, 255], fill="#fef3c7", outline="#d97706", width=1)
    draw.text((725, 230), "Reshape: (1, N, C) -> (1, C, H, W)", fill="#92400e", font=f_txt)

    # Arrow Down
    draw.line([(815, 255), (815, 280)], fill="#64748b", width=2)

    # 7x7 Depthwise Convolution Kernel
    draw.rectangle([680, 280, 950, 320], fill="#bbf7d0", outline="#15803d", width=2)
    draw.text((700, 292), "7x7 Depthwise Conv (groups = C, stride = 1)", fill="#14532d", font=f_box)

    # Arrow Down & Reshape back
    draw.line([(815, 320), (815, 345)], fill="#64748b", width=2)
    draw.rectangle([720, 345, 910, 375], fill="#16a34a")
    draw.text((745, 353), "Linear Projection (W_o Copy)", fill="#ffffff", font=f_box)

    # Complexity Note
    draw.text((590, 395), "Complexity: O(N x K^2 x C) Local Interaction | Pretrained V Transfer", fill="#16a34a", font=f_sub)

    out_p = OUTPUT_DIR / "arch_block_replacement.png"
    img.save(out_p)
    return out_p


def render_controller_workflow_diagram() -> Path:
    w, h = 1100, 520
    img = Image.new("RGB", (w, h), color="#ffffff")
    draw = ImageDraw.Draw(img)

    f_title = get_font(18, bold=True)
    f_box = get_font(13, bold=True)
    f_txt = get_font(11, bold=False)
    f_sub = get_font(10, bold=False)

    # Header
    draw.rectangle([0, 0, w, 40], fill="#0f172a")
    draw.text((25, 10), "REAL-TIME ACCURACY-CONSTRAINED SCHEDULER: DISPATCH & FALLBACK DECISION LOGIC", fill="#ffffff", font=f_title)

    # Box 1: Request Arrival (Left)
    draw.rectangle([40, 70, 240, 160], fill="#f1f5f9", outline="#475569", width=2)
    draw.text((55, 80), "Incoming Request", fill="#0f172a", font=f_box)
    draw.text((55, 105), "&bull; Image Input X", fill="#334155", font=f_txt)
    draw.text((55, 122), "&bull; Deadline D (ms)", fill="#334155", font=f_txt)
    draw.text((55, 139), "&bull; Quality Floor tau_mIoU", fill="#334155", font=f_txt)

    # Arrow to Candidate Registry
    draw.line([(240, 115), (290, 115)], fill="#475569", width=2)
    draw.polygon([(290, 115), (280, 110), (280, 120)], fill="#475569")

    # Box 2: Candidate Profiles
    draw.rectangle([290, 60, 520, 170], fill="#e0f2fe", outline="#0284c7", width=2)
    draw.text((305, 70), "Audited Candidate Registry", fill="#0369a1", font=f_box)
    draw.text((305, 95), "1. M_full: mIoU 0.7430 | p95 15.65ms", fill="#0f172a", font=f_txt)
    draw.text((305, 115), "2. M_mod: mIoU 0.7036 | p95 16.41ms", fill="#0f172a", font=f_txt)
    draw.text((305, 135), "3. M_sigma: mIoU 0.4801 | p95 17.14ms", fill="#0f172a", font=f_txt)
    draw.text((305, 153), "Latency Metric M in {p90, p95, p99}", fill="#0284c7", font=f_sub)

    # Arrow to Decision Diamond / Feasibility Filter
    draw.line([(520, 115), (580, 115)], fill="#475569", width=2)
    draw.polygon([(580, 115), (570, 110), (570, 120)], fill="#475569")

    # Box 3: Feasibility Pruning & Set Intersection
    draw.rectangle([580, 60, 840, 170], fill="#fef3c7", outline="#d97706", width=2)
    draw.text((600, 70), "Joint Feasibility Filter (F)", fill="#92400e", font=f_box)
    draw.text((600, 95), "C_quality = { c | mIoU(c) >= tau }", fill="#78350f", font=f_txt)
    draw.text((600, 115), "C_deadline = { c | Latency_M(c) <= D }", fill="#78350f", font=f_txt)
    draw.text((600, 138), "Feasible Set: F = C_quality intersect C_deadline", fill="#1e293b", font=f_box)

    # Fork: Feasible (Down-Right) vs Infeasible (Down-Left)
    draw.line([(710, 170), (710, 220)], fill="#475569", width=2)

    # Decision Split
    # Path A: F is Non-Empty (Left branch: Green)
    draw.line([(710, 220), (450, 220), (450, 250)], fill="#16a34a", width=2)
    draw.polygon([(450, 250), (445, 240), (455, 240)], fill="#16a34a")
    draw.text((470, 225), "F is Non-Empty (Feasible Request)", fill="#15803d", font=f_box)

    draw.rectangle([310, 250, 590, 360], fill="#dcfce7", outline="#16a34a", width=2)
    draw.text((325, 260), "OPTIMAL DISPATCH SELECTION", fill="#15803d", font=f_box)
    draw.text((325, 285), "Rule: Select candidate maximizing mIoU:", fill="#166534", font=f_txt)
    draw.text((345, 305), "c* = argmax_{F} ( mIoU(c), -Latency_M(c) )", fill="#0f172a", font=f_box)
    draw.text((325, 335), "Ties broken lexicographically by candidate name", fill="#334155", font=f_sub)

    # Path B: F is Empty (Right branch: Red)
    draw.line([(710, 220), (910, 220), (910, 250)], fill="#dc2626", width=2)
    draw.polygon([(910, 250), (905, 240), (915, 240)], fill="#dc2626")
    draw.text((750, 225), "F is Empty (Infeasible Request)", fill="#b91c1c", font=f_box)

    draw.rectangle([740, 250, 1060, 360], fill="#fee2e2", outline="#ef4444", width=2)
    draw.text((755, 260), "DETERMINISTIC FALLBACK ENGINE", fill="#b91c1c", font=f_box)
    draw.text((755, 285), "Trigger Configured Infeasibility Policy:", fill="#7f1d1d", font=f_txt)
    draw.text((770, 305), "1. reject: Drop request (saves 13.3ms GPU time)", fill="#991b1b", font=f_txt)
    draw.text((770, 325), "2. highest_quality_miss: Dispatch max mIoU", fill="#0369a1", font=f_txt)
    draw.text((770, 343), "3. fastest_miss: Dispatch min latency bound", fill="#d97706", font=f_txt)

    # Bottom Execution Tier: GPU Execution
    draw.line([(450, 360), (450, 400), (600, 400)], fill="#475569", width=2)
    draw.line([(900, 360), (900, 400), (800, 400)], fill="#475569", width=2)
    draw.line([(700, 400), (700, 420)], fill="#475569", width=2)
    draw.polygon([(700, 420), (695, 410), (705, 410)], fill="#475569")

    draw.rectangle([150, 420, 950, 500], fill="#0f172a", outline="#334155", width=2)
    draw.text((170, 430), "GPU EXECUTION & METRIC TRACKING (NVIDIA RTX 4060 Laptop GPU)", fill="#38bdf8", font=f_box)
    draw.text((170, 455), "Controller Dispatch: 0.045 - 0.055 ms (< 0.3% budget) | Preprocessing: 4.88ms | Host-to-Device: 0.28ms | Model: 13.13ms | Post: 0.45ms", fill="#cbd5e1", font=f_txt)
    draw.text((170, 475), "Outcome Tracking: Total Hit Rate (54.0%) vs Dispatched Hit Rate (77.1% under reject) | Quality Compliance Verified", fill="#94a3b8", font=f_sub)

    out_p = OUTPUT_DIR / "arch_controller_workflow.png"
    img.save(out_p)
    return out_p


def main() -> None:
    p1 = render_block_replacement_diagram()
    p2 = render_controller_workflow_diagram()
    print(f"Generated: {p1}")
    print(f"Generated: {p2}")


if __name__ == "__main__":
    main()
