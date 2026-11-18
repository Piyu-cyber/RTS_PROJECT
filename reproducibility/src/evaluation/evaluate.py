from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .metrics import compute_classification_metrics, BDD_CLASSES


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = None,
    class_names: list = BDD_CLASSES,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader, returning full metrics, true labels, and predictions.
    
    Args:
        model: PyTorch model.
        dataloader: PyTorch DataLoader for evaluation split.
        device: Device to run evaluation on.
        class_names: List of class strings.
        
    Returns:
        Tuple of (metrics_dict, y_true_array, y_pred_array).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)

    metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    return metrics, y_true, y_pred
