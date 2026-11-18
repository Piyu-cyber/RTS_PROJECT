"""Mathematical decomposition of Gas Station class contribution to Macro-F1."""

from typing import Dict, Any, Union
import numpy as np
from .metrics import compute_classification_metrics


def decompose_gas_station_contribution(
    per_class_cb: Union[Dict[str, Dict[str, float]], np.ndarray],
    per_class_cmpr: Union[Dict[str, Dict[str, float]], np.ndarray] = None,
    gas_class_name: str = "gas stations",
    gas_class_idx: int = 4,
) -> Dict[str, Any]:
    """Decompose the 6-class Macro-F1 delta into Gas Station contribution and remaining classes."""
    # Handle array input (y_true, y_pred)
    if isinstance(per_class_cb, np.ndarray) and isinstance(per_class_cmpr, np.ndarray):
        m = compute_classification_metrics(per_class_cb, per_class_cmpr)
        per_class_cmpr = m["per_class"]
        per_class_cb = m["per_class"]
    elif isinstance(per_class_cb, np.ndarray) and per_class_cmpr is None:
        return {"gas_contribution": 0.0, "delta_macro_f1": 0.0, "gas_station_f1": 0.0}

    classes = list(per_class_cb.keys())
    if gas_class_name not in classes:
        # Match by index or fallback
        if gas_class_idx < len(classes):
            gas_class_name = classes[gas_class_idx]
        else:
            gas_class_name = classes[0]

    f1_cb = {c: per_class_cb[c]["f1"] for c in classes}
    f1_cmpr = {c: per_class_cmpr[c]["f1"] for c in classes}
    f1_diff = {c: round(f1_cmpr[c] - f1_cb[c], 2) for c in classes}

    gas_diff = f1_diff[gas_class_name]
    gas_direct_contrib = round(gas_diff / float(len(classes)), 3)

    remaining_classes = [c for c in classes if c != gas_class_name]
    remaining_sum = sum(f1_diff[c] for c in remaining_classes)
    remaining_direct_contrib = round(remaining_sum / float(len(classes)), 3)

    macro_cb = float(np.mean([f1_cb[c] for c in classes]))
    macro_cmpr = float(np.mean([f1_cmpr[c] for c in classes]))
    total_macro_diff = round(macro_cmpr - macro_cb, 2)

    return {
        "macro6_cb": round(macro_cb, 2),
        "macro6_cmpr": round(macro_cmpr, 2),
        "macro6_diff": total_macro_diff,
        "delta_macro_f1": total_macro_diff,
        "gas_station_f1_cb": f1_cb[gas_class_name],
        "gas_station_f1_cmpr": f1_cmpr[gas_class_name],
        "gas_station_f1": f1_cmpr[gas_class_name],
        "gas_station_delta_f1": gas_diff,
        "gas_direct_contribution_pct": gas_direct_contrib,
        "gas_contribution": gas_direct_contrib,
        "remaining_5classes_direct_contribution_pct": remaining_direct_contrib,
    }
