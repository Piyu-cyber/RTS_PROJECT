import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..utils.seed import set_seed
from ..utils.logging import setup_logger, save_json
from ..data.dataset import BDD100KDataset, BDD_CLASSES
from ..models.deit import build_deit_small
from ..models.evit import build_evit_small
from ..losses.class_balanced_ce import ClassBalancedCrossEntropyLoss, VERIFIED_BDD_COUNTS
from ..losses.cmpr import CompositeTrainingObjective
from .teacher import load_frozen_teacher
from .trainer import Trainer


def get_parameter_groups(model: nn.Module, head_lr: float = 3e-4, backbone_lr: float = 2e-5, weight_decay: float = 1e-4):
    """Separate classifier head parameters from backbone parameters for differential learning rates."""
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


def run_training_experiment(
    model_type: str = "k07",          # 'deit', 'k09', 'k07', 'k05'
    loss_type: str = "cb_cmpr",       # 'ce', 'cb', 'cmpr', 'cb_cmpr'
    seed: int = 42,
    train_manifest: str = None,
    val_manifest: str = None,
    images_dir: str = None,
    teacher_checkpoint: str = None,
    pretrained_weights: str = None,
    output_dir: str = "./outputs/training",
    epochs: int = 3,
    batch_size: int = 64,
    head_lr: float = 3e-4,
    backbone_lr: float = 2e-5,
    weight_decay: float = 1e-4,
    device: str = None,
) -> dict:
    """Configurable entrypoint to train a single experimental model."""
    set_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(output_dir, exist_ok=True)
    run_name = f"{model_type.upper()}-{loss_type.upper().replace('_', '-')}_seed{seed}"

    logger = setup_logger(run_name, log_file=os.path.join(output_dir, f"{run_name}.log"))

    # Load manifests
    with open(train_manifest, "r", encoding="utf-8") as f:
        train_items = json.load(f)
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_items = json.load(f)

    train_samples = [(it["name"], it["label"]) for it in train_items]
    val_samples = [(it["name"], it["label"]) for it in val_items]

    train_dataset = BDD100KDataset(train_samples, images_dir=images_dir, is_train=True)
    val_dataset = BDD100KDataset(val_samples, images_dir=images_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2 if os.name == "nt" else 4, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2 if os.name == "nt" else 4, pin_memory=torch.cuda.is_available(),
    )

    # Build model
    if model_type.lower() == "deit":
        model, _ = build_deit_small(num_classes=6, pretrained_path=pretrained_weights, device=dev)
    else:
        model, _ = build_evit_small(keep_rate_variant=model_type.lower(), num_classes=6, pretrained_path=pretrained_weights, device=dev)

    # Configure objective
    if "cb" in loss_type.lower():
        base_criterion = ClassBalancedCrossEntropyLoss(class_counts=VERIFIED_BDD_COUNTS)
    else:
        base_criterion = nn.CrossEntropyLoss()

    use_cmpr = "cmpr" in loss_type.lower()
    teacher_model = None
    if use_cmpr:
        if not teacher_checkpoint or not os.path.isfile(teacher_checkpoint):
            raise FileNotFoundError(f"Teacher checkpoint required for CMPR: {teacher_checkpoint}")
        teacher_model = load_frozen_teacher(teacher_checkpoint, device=dev, num_classes=6)

    criterion = CompositeTrainingObjective(
        base_criterion=base_criterion,
        use_cmpr=use_cmpr,
        lambda_cmpr=0.5,
        gamma=0.0,
        teacher_gate=True,
    )

    # Optimizer and scheduler
    param_groups = get_parameter_groups(model, head_lr=head_lr, backbone_lr=backbone_lr, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.5 * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=dev,
        teacher_model=teacher_model,
        use_amp=True,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        run_name=run_name,
        config={"model_type": model_type, "loss_type": loss_type, "seed": seed, "epochs": epochs},
        logger=logger,
    )

    results = trainer.train(num_epochs=epochs)
    save_json(results["history"], os.path.join(output_dir, f"{run_name}_history.json"))
    return results
