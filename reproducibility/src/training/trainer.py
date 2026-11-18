import os
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..evaluation.evaluate import evaluate_model
from .checkpoint import save_training_checkpoint


class Trainer:
    """Orchestrates model training, validation checkpointing, and metric logging."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        teacher_model: Optional[nn.Module] = None,
        use_amp: bool = True,
        checkpoint_dir: str = "./checkpoints",
        run_name: str = "experiment",
        config: Dict[str, Any] = None,
        logger: Any = None,
    ):
        self.model = model.to(device)
        self.teacher_model = teacher_model.to(device) if teacher_model else None
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.config = config or {}
        self.logger = logger

        self.best_metric = -float("inf")
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, f"{run_name}_best.pth")
        self.final_checkpoint_path = os.path.join(self.checkpoint_dir, f"{run_name}_final.pth")
        self.history = []

    def log(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one complete training epoch."""
        self.model.train()
        total_loss = 0.0
        total_base_loss = 0.0
        total_cmpr_loss = 0.0
        total_samples = 0
        total_teacher_correct = 0

        for step, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits_s = self.model(images)
                
                logits_t = None
                if self.teacher_model is not None:
                    with torch.no_grad():
                        logits_t = self.teacher_model(images)

                loss, base_loss, cmpr_loss, n_correct = self.criterion(
                    logits_s=logits_s,
                    target=targets,
                    logits_t=logits_t,
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_base_loss += base_loss.item() * bs
            total_cmpr_loss += cmpr_loss.item() * bs
            total_samples += bs
            total_teacher_correct += n_correct

        avg_loss = total_loss / total_samples
        avg_base = total_base_loss / total_samples
        avg_cmpr = total_cmpr_loss / total_samples

        return {
            "epoch": epoch,
            "loss": avg_loss,
            "base_loss": avg_base,
            "cmpr_loss": avg_cmpr,
            "teacher_correct_samples": total_teacher_correct,
        }

    def train(self, num_epochs: int) -> Dict[str, Any]:
        """Execute full training across num_epochs with validation at each epoch."""
        self.log(f"Starting training run '{self.run_name}' for {num_epochs} epochs on {self.device}")
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            t_epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - t_epoch_start

            # Evaluate on validation split
            val_metrics, _, _ = evaluate_model(self.model, self.val_loader, device=self.device)
            val_top1 = val_metrics["top1_accuracy"]
            val_macro = val_metrics["macro_f1_6class"]

            log_entry = {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 4),
                "val_top1": val_top1,
                "val_macro_f1": val_macro,
                "val_weighted_f1": val_metrics["weighted_f1"],
                "epoch_time_sec": round(epoch_time, 2),
            }
            self.history.append(log_entry)
            self.log(
                f"Epoch {epoch}/{num_epochs} | Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Top-1: {val_top1:.2f}% | Val Macro-F1: {val_macro:.2f}% | ({epoch_time:.1f}s)"
            )

            # Checkpoint selection based on validation Macro-F1
            if val_macro > self.best_metric:
                self.best_metric = val_macro
                best_hash = save_training_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    step=epoch * len(self.train_loader),
                    val_metric=val_macro,
                    config=self.config,
                    file_path=self.best_checkpoint_path,
                )
                self.log(f"  --> Saved new best checkpoint (Val Macro-F1: {val_macro:.2f}%, SHA256: {best_hash[:8]}...)")

        # Save final checkpoint
        final_hash = save_training_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=num_epochs,
            step=num_epochs * len(self.train_loader),
            val_metric=self.history[-1]["val_macro_f1"],
            config=self.config,
            file_path=self.final_checkpoint_path,
        )

        total_time = time.time() - start_time
        self.log(f"Training completed in {total_time/60:.2f} mins. Best Val Macro-F1: {self.best_metric:.2f}%")

        return {
            "run_name": self.run_name,
            "best_metric": self.best_metric,
            "best_checkpoint": self.best_checkpoint_path,
            "final_checkpoint": self.final_checkpoint_path,
            "history": self.history,
            "total_time_seconds": round(total_time, 2),
        }
