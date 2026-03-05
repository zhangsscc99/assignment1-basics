# Data loading and checkpointing.
from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, IO

import numpy as np
import torch
from torch import Tensor


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """Sample (input, target) batches from a 1D array of token IDs."""
    n = len(dataset)
    if n < context_length + 1:
        raise ValueError("Dataset too short for context_length")
    max_start = n - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])
    x_t = torch.from_numpy(x).long().to(device)
    y_t = torch.from_numpy(y).long().to(device)
    return x_t, y_t


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    state = torch.load(src, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]
