"""Generate Real-Time Systems Decision Flow Diagram for Phase 17 IEEE Paper.

Shows the sequence:
  Request Arrival -> Deadline/Budget Calculation -> Quality-Threshold Filtering
  -> Latency-Based Admission Decision -> Model Inference or Rejection -> Deadline-Outcome Logging
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase17_final_manuscript" / "figures"


def get_font(size: int, bold: bool = False):
    try:
        font_path = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def render_rts_flow_diagram() -> Path:
    w, h = 1180, 360
    img = Image.new("RGB", (w, h), color="#ffffff")
    draw = ImageDraw.Draw(img)

    f_title = get_font(15, bold=True)
    f_box = get_font(11, bold=True)
    f_sub = get_font(9, bold=False)
    f_tag = get_font(8, bold=True)

    # Title
    draw.rectangle([0, 0, w, 36], fill="#0f172a")
    draw.text((20, 9), "REAL-TIME TASK ADMISSION, SCHEDULING, AND OUTCOME LOGGING PIPELINE", fill="#ffffff", font=f_title)

    # 6 sequential stages
    stages = [
        ("1. Request Arrival", ["Input image x_i", "Release time r_i", "Abs deadline D_i", "Quality floor Q_i"], "#e0f2fe", "#0284c7", "#0369a1"),
        ("2. Budget Calculation", ["B_i(t) = D_i - t", "Waiting delay W_i", "Remaining budget", "Scope: forward/e2e"], "#f1f5f9", "#64748b", "#334155"),
        ("3. Quality Filtering", ["C_Q(i) = {c | q_c >= Q_i}", "Exclude degraded", "Check floor violation", "Report if empty"], "#fef3c7", "#d97706", "#92400e"),
        ("4. Admission Decision", ["Check W_i + C_c <= B_i", "Modes: p95, p99, EUB", "Select c* = max(mIoU)", "Or trigger Fallback"], "#f3e8ff", "#9333ea", "#6b21a8"),
        ("5. Inference / Rejection", ["If feasible: Run GPU", "If infeasible & reject:", "  Drop (0 GPU compute)", "Alt: HQ-miss / Fast-miss"], "#ecfdf5", "#059669", "#065f46"),
        ("6. Outcome Logging", ["Record actual latency", "Log Hit vs Miss", "Dispatched Hit Rate", "Total Arrival Hit Rate"], "#f8fafc", "#475569", "#1e293b"),
    ]

    box_w = 170
    box_h = 175
    start_x = 20
    spacing = 24
    y_top = 65

    for idx, (title, bullets, bg_col, border_col, text_col) in enumerate(stages):
        bx0 = start_x + idx * (box_w + spacing)
        bx1 = bx0 + box_w
        by0 = y_top
        by1 = by0 + box_h

        # Stage Box
        draw.rectangle([bx0, by0, bx1, by1], fill=bg_col, outline=border_col, width=2)
        # Header banner
        draw.rectangle([bx0, by0, bx1, by0 + 28], fill=border_col)
        draw.text((bx0 + 8, by0 + 7), title, fill="#ffffff", font=f_box)

        # Bullets
        for b_idx, bullet in enumerate(bullets):
            draw.text((bx0 + 10, by0 + 38 + b_idx * 28), f"• {bullet}", fill=text_col, font=f_sub)

        # Arrow to next stage
        if idx < len(stages) - 1:
            ax0 = bx1 + 2
            ax1 = bx1 + spacing - 2
            ay = by0 + box_h / 2
            draw.line([(ax0, ay), (ax1, ay)], fill="#64748b", width=3)
            # Arrow head
            draw.polygon([(ax1, ay), (ax1 - 6, ay - 5), (ax1 - 6, ay + 5)], fill="#64748b")

    # Bottom Architectural Taxonomy Mapping
    draw.rectangle([20, 260, 1160, 340], fill="#f8fafc", outline="#cbd5e1", width=1)
    draw.text((35, 270), "REAL-TIME SYSTEMS MAPPING:", fill="#0f172a", font=get_font(10, bold=True))
    draw.text((35, 290), "• Real-Time Category: Deadline-aware, best-effort real-time inference prototype (soft real-time / firm real-time with load shedding).", fill="#334155", font=f_sub)
    draw.text((35, 306), "• Schedulability Semantics: Evaluates empirical percentiles (p95, p99) and conservative empirical upper bound (32.33 ms = max + 3*sigma).", fill="#334155", font=f_sub)
    draw.text((35, 322), "• Hard Real-Time Disclaimer: Formal WCET is NOT established; 32.33 ms is an empirical bound, not a mathematically proven upper bound.", fill="#dc2626", font=get_font(9, bold=True))

    out_path = OUTPUT_DIR / "fig_rts_flow.png"
    img.save(out_path)
    return out_path


if __name__ == "__main__":
    p = render_rts_flow_diagram()
    print(f"Generated RTS Flow Diagram: {p}")
