"""Phase 23B package for official EViT reproduction and BDD100K driving scene classification."""

from .models import build_model, load_official_checkpoint, MODEL_CONFIGS
from .bdd_dataset import get_bdd_datasets, BDD_CLASSES, NUM_BDD_CLASSES, get_bdd_transforms, BDD100KClassificationDataset
from .benchmark_suite import run_multi_run_benchmark
from .bdd_finetuning import finetune_bdd_model, evaluate_bdd_split, compute_metrics

__all__ = [
    "build_model",
    "load_official_checkpoint",
    "MODEL_CONFIGS",
    "get_bdd_datasets",
    "BDD_CLASSES",
    "NUM_BDD_CLASSES",
    "get_bdd_transforms",
    "BDD100KClassificationDataset",
    "run_multi_run_benchmark",
    "finetune_bdd_model",
    "evaluate_bdd_split",
    "compute_metrics",
]
