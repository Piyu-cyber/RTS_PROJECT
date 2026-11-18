"""Self-contained pure-NumPy classification metrics calculation."""

from typing import Dict, Any, List, Tuple
import numpy as np

BDD_CLASSES = [
    "city street",
    "highway",
    "residential",
    "parking lot",
    "gas stations",
    "tunnel",
]


def precision_recall_fscore_support(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    zero_division: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure NumPy implementation of multi-class precision, recall, F1, and support."""
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for c in labels:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        supp = int(np.sum(y_true == c))

        p = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        r = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        f = (2.0 * p * r) / (p + r) if (p + r) > 0 else zero_division

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        supports.append(supp)

    return (
        np.array(precisions, dtype=np.float64),
        np.array(recalls, dtype=np.float64),
        np.array(f1s, dtype=np.float64),
        np.array(supports, dtype=np.int64),
    )


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int] = None,
) -> np.ndarray:
    """Pure NumPy confusion matrix."""
    if labels is None:
        num_classes = max(int(np.max(y_true)), int(np.max(y_pred))) + 1
        labels = list(range(num_classes))
    else:
        num_classes = len(labels)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            cm[t, p] += 1
    return cm


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    num_classes: int = None,
) -> Dict[str, Any]:
    """Compute complete classification metrics including 6-class and 5-class Macro-F1."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if class_names is None:
        if num_classes is not None:
            class_names = [f"class_{i}" for i in range(num_classes)]
        else:
            class_names = BDD_CLASSES

    num_classes = len(class_names)
    labels = list(range(num_classes))

    correct = int(np.sum(y_true == y_pred))
    total = len(y_true)
    top1_accuracy = (correct / total * 100.0) if total > 0 else 0.0

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0.0
    )

    macro_f1_6class = float(np.mean(f1) * 100.0)
    total_supp = float(np.sum(support))
    weighted_f1 = float(np.sum(f1 * (support / total_supp)) * 100.0) if total_supp > 0 else 0.0

    gas_idx = class_names.index("gas stations") if "gas stations" in class_names else 4
    non_gas_indices = [i for i in range(num_classes) if i != gas_idx]
    macro_f1_5class = float(np.mean([f1[i] for i in non_gas_indices]) * 100.0)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(precision[i] * 100.0), 2),
            "recall": round(float(recall[i] * 100.0), 2),
            "f1": round(float(f1[i] * 100.0), 2),
            "support": int(support[i]),
        }

    return {
        "top1_accuracy": round(top1_accuracy, 2),
        "macro_f1": round(macro_f1_6class, 2),
        "macro_f1_6class": round(macro_f1_6class, 2),
        "macro_f1_5class": round(macro_f1_5class, 2),
        "weighted_f1": round(weighted_f1, 2),
        "total_samples": total,
        "correct_predictions": correct,
        "per_class": per_class,
    }


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Empirical percentile bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    N = len(y_true)
    top1_boot = []
    macro_boot = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        b_true = y_true[idx]
        b_pred = y_pred[idx]

        m = compute_classification_metrics(b_true, b_pred)
        top1_boot.append(m["top1_accuracy"])
        macro_boot.append(m["macro_f1"])

    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0

    return {
        "top1_accuracy_ci": (
            round(float(np.percentile(top1_boot, low_pct)), 2),
            round(float(np.percentile(top1_boot, high_pct)), 2),
        ),
        "macro_f1_ci": (
            round(float(np.percentile(macro_boot, low_pct)), 2),
            round(float(np.percentile(macro_boot, high_pct)), 2),
        ),
    }


# Backward compatibility alias
compute_bootstrap_ci = compute_bootstrap_confidence_intervals
