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

##################################################
# Reproducibility
##################################################

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

##################################################
# Hyperparameters
##################################################

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 2e-5
NUM_FOLDS = 5

##################################################
# Dataset
##################################################

dataset = NeuroDaptDataset(
    "data/processed/train.jsonl",
)

##################################################
# Collator
##################################################

collator = NeuroDaptCollator()

##################################################
# Cross Validation
##################################################

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

    ##################################################
    # Dataset Split
    ##################################################

    train_dataset = Subset(
        dataset,
        train_idx,
    )

    val_dataset = Subset(
        dataset,
        val_idx,
    )

    ##################################################
    # DataLoaders
    ##################################################

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

    ##################################################
    # Model
    ##################################################

    model = NeuroDapt(
        unfreeze_last_n_layers=2,
    )

    ##################################################
    # Trainer
    ##################################################

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=LEARNING_RATE,
        checkpoint_dir=f"checkpoints/fold_{fold}",
        patience=8,
    )

    ##################################################
    # Scheduler
    ##################################################

    trainer.scheduler = CosineAnnealingLR(
        trainer.optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    ##################################################
    # Train
    ##################################################

    metrics = trainer.fit(
        epochs=EPOCHS,
    )

    results.append(metrics)

    ##################################################
    # Fold Summary
    ##################################################

    print("\nFold Summary")

    for key, value in metrics.items():
        print(
            f"{key.upper():10s}: {value:.4f}"
        )

##################################################
# Save Results
##################################################

with open(
    "cross_validation_results.json",
    "w",
) as f:

    json.dump(
        results,
        f,
        indent=4,
    )

##################################################
# Overall Summary
##################################################

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