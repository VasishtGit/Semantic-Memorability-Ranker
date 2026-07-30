"""Run k-fold cross-validation for the memorability ranking model."""

import json
import random

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from dataset.dataset import NeuroDaptDataset
from models.memorability_ranker import NeuroDapt
from trainer.collate import NeuroDaptCollator
from trainer.trainer import Trainer

# Ensure deterministic behavior for repeated validation runs.

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Configure batch size, epochs, and the number of folds for evaluation.

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 2e-5
NUM_FOLDS = 5

# Load the full processed dataset once for cross-validation splits.

dataset = NeuroDaptDataset(
    "data/processed/train.jsonl",
)

# Create a shared collator for all folds.

collator = NeuroDaptCollator()

# Iterate through each fold, train a model, and collect validation metrics.

kf = KFold(
    n_splits=NUM_FOLDS,
    shuffle=True,
    random_state=SEED,
)

results = []

for fold, (train_idx, val_idx) in enumerate(
    kf.split(dataset),
    start=1,
):

    print("\n" + "=" * 60)
    print(f"Fold {fold}/{NUM_FOLDS}")
    print("=" * 60)

    # Extract train and validation indices for this fold.

    train_dataset = Subset(
        dataset,
        train_idx,
    )

    val_dataset = Subset(
        dataset,
        val_idx,
    )

    # Build fold-specific loaders with the same batching logic as training.

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    # Create a fresh model for each fold to avoid cross-fold leakage.

    model = NeuroDapt(
        unfreeze_last_n_layers=2,
    )

    # Configure the trainer with a fold-specific checkpoint directory.

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=LEARNING_RATE,
        checkpoint_dir=f"checkpoints/fold_{fold}",
        patience=8,
    )

    # Attach the same cosine schedule used in single-train runs.

    trainer.scheduler = CosineAnnealingLR(
        trainer.optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )


    # Fit the model for this fold and store the resulting metrics.

    metrics = trainer.fit(
        epochs=EPOCHS,
    )

    results.append(metrics)

    # Print the key metrics for the completed fold.

    print("\nFold Summary")

    for key, value in metrics.items():
        print(
            f"{key.upper():10s}: {value:.4f}"
        )

# Persist the collected fold metrics to disk for later inspection.

with open(
    "cross_validation_results.json",
    "w",
) as f:

    json.dump(
        results,
        f,
        indent=4,
    )


# Report the mean and standard deviation of the main evaluation metrics.

print("\n")
print("=" * 60)
print("Cross Validation Results")
print("=" * 60)

for metric in [
    "mae",
    "rmse",
    "pearson",
    "spearman",
]:

    values = [
        r[metric]
        for r in results
    ]

    mean = np.mean(values)
    std = np.std(values)

    print(
        f"{metric.upper():10s}: {mean:.4f} ± {std:.4f}"
    )