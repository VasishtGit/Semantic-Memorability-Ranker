"""Training loop and checkpointing utilities for the memorability ranker."""

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainer.losses import get_loss
from trainer.metrics import regression_metrics


class Trainer:
    """High-level training loop with checkpointing, validation, and early stopping."""

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr=1e-4,
        weight_decay=1e-2,
        loss_name="huber",
        device="cuda",
        checkpoint_dir="checkpoints",
        scheduler=None,
        patience=8,
    ):

        # Move the model to the selected device and create the optimizer.
        self.device = torch.device(device)

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_fn = get_loss(loss_name)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = scheduler

        # Enable mixed-precision training on CUDA for faster updates.
        self.scaler = torch.amp.GradScaler(
            "cuda",
        )

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Track the best validation RMSE and the metrics that produced it.
        self.best_rmse = float("inf")
        self.best_metrics = None

        ##################################################
        # Early Stopping
        ##################################################

        self.patience = patience
        self.wait = 0

    ####################################################################
    # Train One Epoch
    ####################################################################

    def train_epoch(self):
        """Run one full pass over the training loader and return the mean loss."""

        self.model.train()

        running_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc="Training",
        )

        for batch in progress:

            input_ids = batch["input_ids"].to(
                self.device,
            )

            attention_mask = batch[
                "attention_mask"
            ].to(
                self.device,
            )

            labels = batch["labels"].to(
                self.device,
            )

            self.optimizer.zero_grad(
                set_to_none=True,
            )

            with torch.amp.autocast(
                "cuda",
            ):

                prediction = self.model(
                    input_ids,
                    attention_mask,
                )

                loss = self.loss_fn(
                    prediction,
                    labels,
                )

            self.scaler.scale(
                loss,
            ).backward()

            self.scaler.unscale_(
                self.optimizer,
            )

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            self.scaler.step(
                self.optimizer,
            )

            self.scaler.update()

            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
            )

        return running_loss / len(
            self.train_loader,
        )

    ####################################################################
    # Validation
    ####################################################################

    def validate_epoch(self):
        """Run validation and compute regression metrics on the held-out data."""

        self.model.eval()

        predictions = []
        labels = []

        running_loss = 0.0

        with torch.no_grad():

            progress = tqdm(
                self.val_loader,
                desc="Validation",
            )

            for batch in progress:

                input_ids = batch["input_ids"].to(
                    self.device,
                )

                attention_mask = batch[
                    "attention_mask"
                ].to(
                    self.device,
                )

                target = batch["labels"].to(
                    self.device,
                )

                prediction = self.model(
                    input_ids,
                    attention_mask,
                )

                loss = self.loss_fn(
                    prediction,
                    target,
                )

                running_loss += loss.item()

                predictions.extend(
                    prediction.cpu().tolist()
                )

                labels.extend(
                    target.cpu().tolist()
                )

        metrics = regression_metrics(
            predictions,
            labels,
        )

        metrics["loss"] = (
            running_loss / len(self.val_loader)
        )

        return metrics

    ####################################################################
    # Save Model
    ####################################################################

    def save_checkpoint(
        self,
        filename,
    ):
        """Persist the current model weights to disk."""

        torch.save(
            self.model.state_dict(),
            self.checkpoint_dir / filename,
        )

    ####################################################################
    # Fit
    ####################################################################

    def fit(
        self,
        epochs,
    ):
        """Train the model for the requested number of epochs."""

        for epoch in range(
            1,
            epochs + 1,
        ):

            print(
                f"\nEpoch {epoch}/{epochs}"
            )

            train_loss = self.train_epoch()

            metrics = self.validate_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Train Loss : {train_loss:.4f}"
            )

            print(
                f"Val Loss   : {metrics['loss']:.4f}"
            )

            print(
                f"MAE        : {metrics['mae']:.4f}"
            )

            print(
                f"RMSE       : {metrics['rmse']:.4f}"
            )

            print(
                f"Pearson    : {metrics['pearson']:.4f}"
            )

            print(
                f"Spearman   : {metrics['spearman']:.4f}"
            )

            self.save_checkpoint(
                "latest.pt",
            )

            ##################################################
            # Best Model
            ##################################################

            if metrics["rmse"] < self.best_rmse:

                self.best_rmse = metrics["rmse"]

                self.best_metrics = metrics.copy()

                self.wait = 0

                self.save_checkpoint(
                    "best.pt",
                )

                print(
                    "✓ Best model updated."
                )

            else:

                self.wait += 1

                print(
                    f"No improvement ({self.wait}/{self.patience})"
                )

            ##################################################
            # Early Stopping
            ##################################################

            if self.wait >= self.patience:

                print("\nEarly stopping triggered.")

                break

        ##################################################
        # Load Best Model
        ##################################################

        self.model.load_state_dict(
            torch.load(
                self.checkpoint_dir / "best.pt",
                map_location=self.device,
            )
        )

        return self.best_metrics