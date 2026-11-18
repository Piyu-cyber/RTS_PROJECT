from typing import Tuple, Dict, Any, List
import numpy as np
from scipy import stats


def compute_mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, Any]:
    """Perform McNemar's test on paired classification predictions.
    
    Contingency table:
        b = Model A correct, Model B incorrect
        c = Model A incorrect, Model B correct
        
    Statistic with continuity correction:
        chi2 = (|b - c| - 1)^2 / (b + c)
    """
    y_true = np.asarray(y_true, dtype=int)
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B right

    total_discordant = b + c
    if total_discordant == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "significant": False}

    statistic = ((abs(b - c) - 1.0) ** 2) / float(total_discordant)
    p_value = 1.0 - stats.chi2.cdf(statistic, df=1)

    return {
        "b_a_correct_b_wrong": b,
        "c_a_wrong_b_correct": c,
        "statistic": round(float(statistic), 4),
        "p_value": float(p_value),
        "significant_p05": p_value < 0.05,
    }


def compute_paired_seed_ttest(
    values_a: List[float],
    values_b: List[float],
) -> Dict[str, Any]:
    """Perform exploratory paired t-test across matched training seeds."""
    va = np.asarray(values_a, dtype=float)
    vb = np.asarray(values_b, dtype=float)
    diff = vb - va

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0

    t_stat, p_val = stats.ttest_rel(vb, va)

    return {
        "mean_difference": round(mean_diff, 4),
        "std_difference": round(std_diff, 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_val), 4),
        "degrees_of_freedom": len(values_a) - 1,
        "significant_p05": p_val < 0.05,
    }
