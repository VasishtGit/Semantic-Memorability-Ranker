"""Loss factory for selecting the regression objective."""

import torch.nn as nn


def get_loss(name="huber"):
    """Return the requested PyTorch loss module for regression training."""

    name = name.lower()

    if name == "mse":
        return nn.MSELoss()

    if name == "mae":
        return nn.L1Loss()

    if name == "huber":
        return nn.SmoothL1Loss(beta=0.1)

    raise ValueError(f"Unknown loss: {name}")