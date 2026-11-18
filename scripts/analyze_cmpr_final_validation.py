"""Margin Distribution, Teacher-Student Flip, and Ablation Analysis Engine.

Processes test logits and predictions across:
- M0_seed42 (DeiT Baseline Teacher)
- M1_seed42 (EViT Baseline Student)
- M2_seed42 (EViT CB-CE Student)
- M6_seed42 (EViT CMPR Student)
- M7_seed42 (EViT CB-CE + CMPR Student, once available)

Generates:
- outputs/cmpr_final_validation/margin_comparison.csv
- outputs/cmpr_final_validation/ablation_results.csv
- outputs/cmpr_final_validation/per_class_metrics.csv
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "cmpr_final_validation")
CM_DIR_MATCHED = os.path.join(PROJECT_ROOT, "outputs", "cmpr_matched_training", "confusion_matrices")
CM_DIR_FINAL = os.path.join(OUTPUT_DIR, "confusion_matrices")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CM_DIR_FINAL, exist_ok=True)

from src.phase23b.bdd_dataset import BDD_CLASSES, NUM_BDD_CLASSES

def load_run_data(run_name: str) -> dict:
    cand1 = os.path.join(CM_DIR_FINAL, f"{run_name}_confusion_matrix.json")
    cand2 = os.path.join(CM_DIR_MATCHED, f"{run_name}_confusion_matrix.json")
    path = cand1 if os.path.isfile(cand1) else cand2
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find confusion matrix for {run_name}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_margin_distribution() -> pd.DataFrame:
    print("[*] Computing margin distributions and flip metrics...", flush=True)
    m0_data = load_run_data("M0_seed42")
    m0_logits = np.array(m0_data["test_logits"], dtype=np.float32)
    m0_preds = np.array(m0_data["test_predictions"], dtype=np.int64)
    targets = np.array(m0_data["test_targets"], dtype=np.int64)

    # Compute M0 margins
    m0_true_logits = m0_logits[np.arange(len(targets)), targets]
    m0_other_logits = np.copy(m0_logits)
    m0_other_logits[np.arange(len(targets)), targets] = -1e9
    m0_max_other = np.max(m0_other_logits, axis=1)
    m0_margins = m0_true_logits - m0_max_other

    students = ["M1_seed42", "M2_seed42", "M6_seed42"]
    if os.path.isfile(os.path.join(CM_DIR_FINAL, "M7_seed42_confusion_matrix.json")) or os.path.isfile(os.path.join(CM_DIR_MATCHED, "M7_seed42_confusion_matrix.json")):
        students.append("M7_seed42")

    rows = []
    for s_name in students:
        s_data = load_run_data(s_name)
        s_logits = np.array(s_data["test_logits"], dtype=np.float32)
        s_preds = np.array(s_data["test_predictions"], dtype=np.int64)

        s_true_logits = s_logits[np.arange(len(targets)), targets]
        s_other_logits = np.copy(s_logits)
        s_other_logits[np.arange(len(targets)), targets] = -1e9
        s_max_other = np.max(s_other_logits, axis=1)
        s_margins = s_true_logits - s_max_other

        # Per-class margin metrics
        for c in range(NUM_BDD_CLASSES):
            c_mask = (targets == c)
            c_supp = int(np.sum(c_mask))
            if c_supp == 0:
                continue

            c_s_margins = s_margins[c_mask]
            c_m0_margins = m0_margins[c_mask]

            # Flips
            t_correct = (m0_preds[c_mask] == c)
            s_correct = (s_preds[c_mask] == c)

            neg_flips = int(np.sum(t_correct & ~s_correct)) # teacher correct, student wrong
            pos_flips = int(np.sum(~t_correct & s_correct)) # teacher wrong, student correct
            both_correct = int(np.sum(t_correct & s_correct))
            both_wrong = int(np.sum(~t_correct & ~s_correct))

            rows.append({
                "student_run": s_name,
                "class_id": c,
                "class_name": BDD_CLASSES[c],
                "support": c_supp,
                "student_recall": round(float(np.mean(s_correct) * 100.0), 2),
                "teacher_recall": round(float(np.mean(t_correct) * 100.0), 2),
                "mean_student_margin": round(float(np.mean(c_s_margins)), 3),
                "median_student_margin": round(float(np.median(c_s_margins)), 3),
                "p10_student_margin": round(float(np.percentile(c_s_margins, 10)), 3),
                "mean_teacher_margin": round(float(np.mean(c_m0_margins)), 3),
                "margin_delta": round(float(np.mean(c_s_margins) - np.mean(c_m0_margins)), 3),
                "negative_flips_count": neg_flips,
                "negative_flip_rate_pct": round(float(neg_flips / max(1, np.sum(t_correct)) * 100.0), 2),
                "positive_flips_count": pos_flips,
                "both_correct_count": both_correct,
                "both_wrong_count": both_wrong,
            })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, "margin_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"[+] Saved margin comparison to: {out_csv}", flush=True)
    return df

def run_margin_ablations() -> pd.DataFrame:
    print("[*] Running margin-loss formulation and gating ablations...", flush=True)
    m0_data = load_run_data("M0_seed42")
    m0_logits = torch.tensor(m0_data["test_logits"], dtype=torch.float32)
    m0_preds = torch.tensor(m0_data["test_predictions"], dtype=torch.long)
    targets = torch.tensor(m0_data["test_targets"], dtype=torch.long)

    # Teacher correctness
    teacher_correct = (m0_preds == targets)

    students = ["M1_seed42", "M2_seed42", "M6_seed42"]
    if os.path.isfile(os.path.join(CM_DIR_FINAL, "M7_seed42_confusion_matrix.json")) or os.path.isfile(os.path.join(CM_DIR_MATCHED, "M7_seed42_confusion_matrix.json")):
        students.append("M7_seed42")

    ablation_rows = []

    for s_name in students:
        s_data = load_run_data(s_name)
        s_logits = torch.tensor(s_data["test_logits"], dtype=torch.float32)

        # Margins
        mask = torch.zeros_like(s_logits, dtype=torch.bool).scatter_(1, targets.unsqueeze(1), True)
        s_true = s_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        t_true = m0_logits.gather(1, targets.unsqueeze(1)).squeeze(1)

        s_other_max = s_logits.masked_fill(mask, -1e4).max(dim=1).values
        t_other_max = m0_logits.masked_fill(mask, -1e4).max(dim=1).values

        m_s = s_true - s_other_max
        m_t = t_true - t_other_max
        diff = m_t - m_s

        # Ablation A & B: Gating enabled vs disabled
        hinge_loss = torch.clamp(diff, min=0.0) ** 2
        gated_loss = (teacher_correct.float() * hinge_loss).sum() / (teacher_correct.sum() + 1e-6)
        ungated_loss = hinge_loss.mean()

        # Ablation C & D: Normalized vs Raw vs Squared Hinge
        raw_diff_loss = torch.clamp(diff, min=0.0).mean()
        norm_factor = m_t.std() + 1e-6
        norm_hinge_loss = ((torch.clamp(diff, min=0.0) / norm_factor) ** 2).mean()

        # Teacher accuracy on rare classes
        for c in range(NUM_BDD_CLASSES):
            c_mask = (targets == c)
            c_t_acc = float(teacher_correct[c_mask].float().mean() * 100.0)
            c_gated = (teacher_correct[c_mask].float() * hinge_loss[c_mask]).sum() / (teacher_correct[c_mask].sum() + 1e-6) if teacher_correct[c_mask].sum() > 0 else 0.0
            c_ungated = hinge_loss[c_mask].mean().item()

            ablation_rows.append({
                "student_model": s_name,
                "class_id": c,
                "class_name": BDD_CLASSES[c],
                "teacher_class_accuracy_pct": round(c_t_acc, 2),
                "teacher_gated_hinge_loss": round(float(c_gated), 4),
                "ungated_hinge_loss": round(float(c_ungated), 4),
                "gating_loss_difference": round(float(c_ungated - c_gated), 4),
                "raw_linear_loss": round(float(torch.clamp(diff[c_mask], min=0.0).mean().item()), 4),
                "normalized_hinge_loss": round(float((((torch.clamp(diff[c_mask], min=0.0) / norm_factor) ** 2).mean()).item()), 4),
            })

    df = pd.DataFrame(ablation_rows)
    out_csv = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"[+] Saved ablation results to: {out_csv}", flush=True)
    return df

if __name__ == "__main__":
    compute_margin_distribution()
    run_margin_ablations()
