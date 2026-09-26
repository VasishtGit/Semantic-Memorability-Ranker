"""Training entry point for the memorability ranking model.

This script manages the end-to-end training procedure: setting random seeds for reproducibility,
loading the JSONL dataset, performing a leak-free story-level train/validation split,
organizing samples into story-grouped batches, configuring the optimizer and cosine
learning rate scheduler, and executing the training loop with validation and early stopping.
"""

import random

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import Subset

from dataset.dataset import MemorabilityDataset
from models.memorability_ranker import SemanticMemorabilityRanker
from trainer.collate import MemorabilityCollator
from trainer.trainer import Trainer


# Set fixed random seed for reproducible data splitting and weight initialization
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# Training hyperparameters
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4


class StoryBatchSampler(Sampler):
    """Yield batches containing clauses from a single story."""

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
    ):
        """Group dataset sample indices by story identifier.

        Args:
            dataset: Dataset or Subset instance providing sample records.
            batch_size: Maximum number of clauses to include per batch.
            shuffle: Whether to randomize story order and clause order within stories.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        story_to_indices = {}

        # Map each story ID to its list of dataset indices
        for local_index in range(
            len(dataset)
        ):
            sample = dataset[local_index]

            story_id = sample[
                "narrative_id"
            ]

            if story_id not in story_to_indices:
                story_to_indices[story_id] = []

            story_to_indices[
                story_id
            ].append(local_index)

        self.story_to_indices = story_to_indices

    def __iter__(self):
        """Iterate over batches of clause indices grouped by story."""
        story_ids = list(
            self.story_to_indices.keys()
        )

        # Shuffle story presentation order if requested
        if self.shuffle:
            random.shuffle(story_ids)

        for story_id in story_ids:
            indices = list(
                self.story_to_indices[story_id]
            )

            # Shuffle clauses within the story if requested
            if self.shuffle:
                random.shuffle(indices)

            # Chunk indices into batches of size batch_size
            for start in range(
                0,
                len(indices),
                self.batch_size,
            ):
                yield indices[
                    start:start + self.batch_size
                ]

    def __len__(self):
        """Compute the total number of batches across all stories."""
        total_batches = 0

        for indices in self.story_to_indices.values():
            total_batches += (
                len(indices)
                + self.batch_size
                - 1
            ) // self.batch_size

        return total_batches


# Load the preprocessed training dataset
dataset = MemorabilityDataset(
    "data/processed/breithaupt_training.jsonl",
)

# Story-level train/validation split

# Extract all unique narrative story identifiers
story_ids = sorted(
    {
        sample["narrative_id"]
        for sample in dataset.samples
    }
)

# Shuffle story IDs deterministically
random.Random(SEED).shuffle(
    story_ids
)

# Split stories 80% train, 20% validation
train_story_count = int(
    0.8 * len(story_ids)
)

train_story_ids = set(
    story_ids[:train_story_count]
)

val_story_ids = set(
    story_ids[train_story_count:]
)


# Partition dataset sample indices based on story split to prevent leakage
train_indices = [
    index
    for index, sample in enumerate(
        dataset.samples
    )
    if sample["narrative_id"]
    in train_story_ids
]

val_indices = [
    index
    for index, sample in enumerate(
        dataset.samples
    )
    if sample["narrative_id"]
    in val_story_ids
]


# Create PyTorch Subset views for training and validation splits
train_dataset = Subset(
    dataset,
    train_indices,
)

val_dataset = Subset(
    dataset,
    val_indices,
)


# Log dataset split statistics
print(
    f"Total stories: "
    f"{len(story_ids)}"
)

print(
    f"Train stories: "
    f"{len(train_story_ids)}"
)

print(
    f"Validation stories: "
    f"{len(val_story_ids)}"
)

print(
    f"Train examples: "
    f"{len(train_dataset)}"
)

print(
    f"Validation examples: "
    f"{len(val_dataset)}"
)

print(
    f"Story leakage: "
    f"{bool(train_story_ids & val_story_ids)}"
)


# Initialize batch collator with tokenizer
collator = MemorabilityCollator()


# Configure story batch samplers for train and validation
train_batch_sampler = StoryBatchSampler(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_batch_sampler = StoryBatchSampler(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# Create DataLoaders using the story batch samplers
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    collate_fn=collator,
)

val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_batch_sampler,
    collate_fn=collator,
)


# Initialize the ranker model with the top 2 ModernBERT layers unfrozen
model = SemanticMemorabilityRanker(
    unfreeze_last_n_layers=2,
)


# Configure high-level training manager
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=LEARNING_RATE,
)


# Attach cosine annealing learning rate scheduler
trainer.scheduler = CosineAnnealingLR(
    trainer.optimizer,
    T_max=EPOCHS,
    eta_min=1e-6,
)


# Execute training when run as the main script
if __name__ == "__main__":
    trainer.fit(
        epochs=EPOCHS,
    )
