"""Training entry point for the memorability ranking model."""

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


SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


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
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        story_to_indices = {}

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
        story_ids = list(
            self.story_to_indices.keys()
        )

        if self.shuffle:
            random.shuffle(story_ids)

        for story_id in story_ids:
            indices = list(
                self.story_to_indices[story_id]
            )

            if self.shuffle:
                random.shuffle(indices)

            for start in range(
                0,
                len(indices),
                self.batch_size,
            ):
                yield indices[
                    start:start + self.batch_size
                ]

    def __len__(self):
        total_batches = 0

        for indices in self.story_to_indices.values():
            total_batches += (
                len(indices)
                + self.batch_size
                - 1
            ) // self.batch_size

        return total_batches


dataset = MemorabilityDataset(
    "data/processed/breithaupt_training.jsonl",
)


# ------------------------------------------------------------------
# Story-level train/validation split
# ------------------------------------------------------------------

story_ids = sorted(
    {
        sample["narrative_id"]
        for sample in dataset.samples
    }
)

random.Random(SEED).shuffle(
    story_ids
)

train_story_count = int(
    0.8 * len(story_ids)
)

train_story_ids = set(
    story_ids[:train_story_count]
)

val_story_ids = set(
    story_ids[train_story_count:]
)


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


train_dataset = Subset(
    dataset,
    train_indices,
)

val_dataset = Subset(
    dataset,
    val_indices,
)


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


collator = MemorabilityCollator()


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


model = SemanticMemorabilityRanker(
    unfreeze_last_n_layers=2,
)


trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    lr=LEARNING_RATE,
)


trainer.scheduler = CosineAnnealingLR(
    trainer.optimizer,
    T_max=EPOCHS,
    eta_min=1e-6,
)


if __name__ == "__main__":
    trainer.fit(
        epochs=EPOCHS,
    )