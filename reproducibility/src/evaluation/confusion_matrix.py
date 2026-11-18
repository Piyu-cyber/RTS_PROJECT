from typing import List, Dict, Any
import numpy as np
import pandas as pd
from .metrics import BDD_CLASSES


def format_confusion_matrix_dataframe(
    cm_array: List[List[int]],
    class_names: List[str] = BDD_CLASSES,
) -> pd.DataFrame:
    """Format raw confusion matrix array into a readable pandas DataFrame."""
    df = pd.DataFrame(cm_array, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
    return df


def compute_normalized_confusion_matrices(
    cm_array: List[List[int]],
) -> Dict[str, np.ndarray]:
    """Compute row-normalized (recall) and column-normalized (precision) confusion matrices."""
    cm = np.array(cm_array, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    col_sums = cm.sum(axis=0, keepdims=True)

    cm_row_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0
    cm_col_norm = np.divide(cm, col_sums, out=np.zeros_like(cm), where=col_sums != 0) * 100.0

    return {
        "raw": cm,
        "row_normalized_recall_pct": np.round(cm_row_norm, 2),
        "column_normalized_precision_pct": np.round(cm_col_norm, 2),
    }
