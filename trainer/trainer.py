"""Training loop and checkpointing utilities for the memorability ranker.

This module provides a production-grade training manager featuring mixed-precision
acceleration, gradient clipping, within-story pairwise accuracy evaluation, cosine
learning rate scheduling, best-checkpoint persistence, and early stopping.
"""

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainer.losses import get_loss
from trainer.metrics import regression_metrics


def pairwise_accuracy(
    predictions,
    targets,
    story_ids,
):
    """Calculate within-story pairwise ranking accuracy.

    For every pair of clauses within the same story where target scores differ,
    checks whether the predicted relative order matches the target order.

    Args:
        predictions: Predicted memorability tensor of shape (N,).
        targets: Ground-truth target tensor of shape (N,).
        story_ids: Story identifier tensor of shape (N,).

    Returns:
        Fraction of correctly ordered pairs as a float in [0.0, 1.0].
    """

    correct = 0
    total = 0

    # Identify all distinct story IDs present in the evaluation tensors
    unique_stories = torch.unique(
        story_ids
    )

    # Evaluate pairwise concordance story by story
    for story_id in unique_stories:

        mask = story_ids == story_id

        story_predictions = predictions[
            mask
        ]

        story_targets = targets[
            mask
        ]

        # Require at least two clauses in a story to evaluate pairwise comparisons
        if len(story_predictions) < 2:
            continue

        # Form matrix of pairwise differences for predictions
        prediction_diff = (
            story_predictions.unsqueeze(1)
            - story_predictions.unsqueeze(0)
        )

        # Form matrix of pairwise differences for ground-truth targets
        target_diff = (
            story_targets.unsqueeze(1)
            - story_targets.unsqueeze(0)
        )

        # Only compare pairs with different target scores.
        pair_mask = target_diff != 0

        # Skip if no pairs have distinct targets
        if not pair_mask.any():
            continue

        # Count pairs where predicted difference direction matches target direction
        correct += (
            (
                torch.sign(
                    prediction_diff[pair_mask]
                )
                == torch.sign(
                    target_diff[pair_mask]
                )
            )
            .sum()
            .item()
        )

        # Increment total count of compared valid pairs
        total += pair_mask.sum().item()

    # Return zero if no valid pairs were found
    if total == 0:
        return 0.0

    return correct / total


class Trainer:
    """High-level training loop with checkpointing, validation, and early stopping."""

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr=1e-4,
        weight_decay=1e-2,
        loss_name="ranking",
        device="cuda",
        checkpoint_dir="checkpoints",
        scheduler=None,
        patience=10,
    ):
        """Initialize the training manager.

        Args:
            model: PyTorch neural network module to train.
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            lr: Initial learning rate for AdamW.
            weight_decay: Weight decay coefficient for AdamW regularization.
            loss_name: Name of the loss criterion ('ranking', 'mse', 'mae', 'huber').
            device: Target device string ('cuda' or 'cpu').
            checkpoint_dir: Directory path where model checkpoints are saved.
            scheduler: Optional learning rate scheduler.
            patience: Number of epochs to wait without improvement before early stopping.
        """
        # Resolve the requested device.
        #
        # If CUDA was requested but is unavailable, fall back to CPU.
        if (
            device == "cuda"
            and not torch.cuda.is_available()
        ):
            print(
                "CUDA is unavailable. "
                "Falling back to CPU."
            )
            device = "cpu"

        self.device = torch.device(
            device
        )

        # Move model to the selected device.
        self.model = model.to(
            self.device
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Ranking loss.
        self.loss_fn = get_loss(
            loss_name
        )

        # Optimizer.
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = scheduler

        # Mixed precision is enabled only on CUDA.
        self.use_amp = (
            self.device.type == "cuda"
        )

        # Initialize gradient scaler for automatic mixed precision on CUDA
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(
                "cuda"
            )
        else:
            self.scaler = None

        # Checkpoint directory.
        self.checkpoint_dir = Path(
            checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Track the best validation pairwise accuracy.
        self.best_pairwise_accuracy = (
            float("-inf")
        )

        self.best_metrics = None

        # Early stopping.
        self.patience = patience
        self.wait = 0

    # Train One Epoch

    def train_epoch(self):
        """Run one full pass over the training loader."""

        # Set model to training mode
        self.model.train()

        running_loss = 0.0

        # Progress bar over training batches
        progress = tqdm(
            self.train_loader,
            desc="Training",
        )

        for batch in progress:

            # Transfer token IDs to active compute device
            input_ids = batch[
                "input_ids"
            ].to(
                self.device
            )

            # Transfer attention mask to active compute device
            attention_mask = batch[
                "attention_mask"
            ].to(
                self.device
            )

            # Mask identifying target-clause tokens.
            #
            # 1 = target clause
            # 0 = paragraph/context/special/padding token
            clause_mask = batch[
                "clause_mask"
            ].to(
                self.device
            )

            # Transfer target labels to active compute device
            labels = batch[
                "labels"
            ].to(
                self.device
            )

            # Transfer story IDs to active compute device
            story_ids = batch[
                "story_ids"
            ].to(
                self.device
            )

            # Clear parameter gradients from previous step
            self.optimizer.zero_grad(
                set_to_none=True
            )

            # CUDA uses mixed precision.
            # CPU uses normal float32 execution.
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                # Forward pass through the memorability ranker
                prediction = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    clause_mask=clause_mask,
                )

                # Compute composite regression and pairwise ranking loss
                loss = self.loss_fn(
                    prediction,
                    labels,
                    story_ids,
                )

            # Backpropagation and optimization step with AMP scaling
            if self.use_amp:

                # Scale loss and backpropagate gradients
                self.scaler.scale(
                    loss
                ).backward()

                # Unscale gradients prior to gradient clipping
                self.scaler.unscale_(
                    self.optimizer
                )

                # Clip gradient norms to stabilize training
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                # Step optimizer and update scaler
                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

            else:

                # Standard backpropagation on CPU
                loss.backward()

                # Clip gradient norms
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                # Step optimizer parameters
                self.optimizer.step()

            running_loss += loss.item()

            # Update progress bar display
            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        # Return mean loss across all batches
        return (
            running_loss
            / len(self.train_loader)
        )

    # Validation

    def validate_epoch(self):
        """Run validation and compute regression/ranking metrics."""

        # Set model to evaluation mode
        self.model.eval()

        predictions = []
        labels = []
        story_ids = []

        running_loss = 0.0

        # Disable gradient computation during validation
        with torch.no_grad():

            progress = tqdm(
                self.val_loader,
                desc="Validation",
            )

            for batch in progress:

                # Transfer batch tensors to compute device
                input_ids = batch[
                    "input_ids"
                ].to(
                    self.device
                )

                attention_mask = batch[
                    "attention_mask"
                ].to(
                    self.device
                )

                # Target-clause token mask.
                clause_mask = batch[
                    "clause_mask"
                ].to(
                    self.device
                )

                target = batch[
                    "labels"
                ].to(
                    self.device
                )

                batch_story_ids = batch[
                    "story_ids"
                ].to(
                    self.device
                )

                # Execute forward evaluation under autocast
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                ):
                    prediction = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        clause_mask=clause_mask,
                    )

                    loss = self.loss_fn(
                        prediction,
                        target,
                        batch_story_ids,
                    )

                running_loss += loss.item()

                # Collect batch predictions and targets for global evaluation
                predictions.extend(
                    prediction.cpu().tolist()
                )

                labels.extend(
                    target.cpu().tolist()
                )

                story_ids.extend(
                    batch_story_ids.cpu().tolist()
                )

        # Compute regression metrics (MAE, RMSE, Pearson, Spearman)
        metrics = regression_metrics(
            predictions,
            labels,
        )

        # Calculate pairwise accuracy across the
        # complete validation set, grouped by story.
        prediction_tensor = torch.tensor(
            predictions,
            dtype=torch.float32,
        )

        label_tensor = torch.tensor(
            labels,
            dtype=torch.float32,
        )

        story_id_tensor = torch.tensor(
            story_ids,
            dtype=torch.long,
        )

        metrics[
            "pairwise_accuracy"
        ] = pairwise_accuracy(
            prediction_tensor,
            label_tensor,
            story_id_tensor,
        )

        metrics["loss"] = (
            running_loss
            / len(self.val_loader)
        )

        return metrics

    # Save Model

    def save_checkpoint(
        self,
        filename,
    ):
        """Persist the current model weights to disk."""

        # Move a detached copy of the model weights to CPU before
        # serialization. This keeps the saved checkpoint device-neutral.
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.model.state_dict().items()
        }

        # Save model state dictionary to target checkpoint path
        torch.save(
            state_dict,
            self.checkpoint_dir / filename,
        )

        del state_dict

    # Fit

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

            # Execute training pass
            train_loss = (
                self.train_epoch()
            )

            # Execute validation pass
            metrics = (
                self.validate_epoch()
            )

            # Step the learning rate scheduler if configured
            if self.scheduler is not None:
                self.scheduler.step()

            # Log epoch metrics to console
            print(
                f"Train Loss : "
                f"{train_loss:.4f}"
            )

            print(
                f"Val Loss   : "
                f"{metrics['loss']:.4f}"
            )

            print(
                f"MAE        : "
                f"{metrics['mae']:.4f}"
            )

            print(
                f"RMSE       : "
                f"{metrics['rmse']:.4f}"
            )

            print(
                f"Pearson    : "
                f"{metrics['pearson']:.4f}"
            )

            print(
                f"Spearman   : "
                f"{metrics['spearman']:.4f}"
            )

            print(
                f"Pairwise Acc : "
                f"{metrics['pairwise_accuracy']:.4f}"
            )

            # Best Model

            # Save when validation pairwise accuracy improves.
            if (
                metrics["pairwise_accuracy"]
                > self.best_pairwise_accuracy
            ):

                self.best_pairwise_accuracy = (
                    metrics[
                        "pairwise_accuracy"
                    ]
                )

                self.best_metrics = (
                    metrics.copy()
                )

                self.wait = 0

                # Persist the new best checkpoint
                self.save_checkpoint(
                    "best.pt"
                )

                print(
                    "✓ Best model updated."
                )

            else:

                self.wait += 1

                print(
                    f"No improvement "
                    f"({self.wait}/{self.patience})"
                )


            # Early Stopping

            # Check if patience threshold has been reached
            if (
                self.wait
                >= self.patience
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break

        # Load Best Model

        best_checkpoint = (
            self.checkpoint_dir
            / "best.pt"
        )

        # Verify that a checkpoint file was successfully created
        if not best_checkpoint.exists():

            raise RuntimeError(
                "No best checkpoint was created. "
                "Validation may have failed before the first "
                "checkpoint could be saved."
            )

        # Restore model parameters from the best recorded checkpoint
        self.model.load_state_dict(
            torch.load(
                best_checkpoint,
                map_location=self.device,
            )
        )

        return self.best_metrics
