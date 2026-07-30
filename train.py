"""Training entry point for the memorability ranking model."""

import random

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset.dataset import NeuroDaptDataset
from models.memorability_ranker import NeuroDapt
from trainer.collate import NeuroDaptCollator
from trainer.trainer import Trainer

# Keep runs stable by fixing the random seed across Python, NumPy, and PyTorch.

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Basic training configuration for the memorability regressor.

BATCH_SIZE = 16

EPOCHS = 50

LEARNING_RATE = 2e-5

# Load the processed training data from disk.

dataset = NeuroDaptDataset(
    "data/processed/train.jsonl",
)

train_size = int(
    0.8 * len(dataset)
)

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)


# Use a collator to batch text samples into model-ready tensors.

collator = NeuroDaptCollator()


# Split the dataset into train and validation loaders for iterative training.

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
)



# Instantiate the core ranking model with a lightly fine-tuned backbone.

model = NeuroDapt(
    unfreeze_last_n_layers=2,
)

# Wrap the model and loaders in the custom training runtime.

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=LEARNING_RATE,
)

# Use a cosine schedule to gradually decay the learning rate during training.

trainer.scheduler = CosineAnnealingLR(
    trainer.optimizer,
    T_max=EPOCHS,
    eta_min=1e-6,
)

# Start the optimization loop for the configured number of epochs.

trainer.fit(
    epochs=EPOCHS,
)