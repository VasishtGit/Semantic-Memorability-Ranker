"""Training utilities and runtime components for the memorability ranker.

This package provides the core training pipeline components, including batch collation
with clause mask generation, composite regression and ranking loss functions, evaluation
metrics (MAE, RMSE, Pearson, Spearman, and pairwise accuracy), and the high-level Trainer
loop with mixed precision and checkpointing support.
"""
