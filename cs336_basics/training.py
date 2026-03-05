# 数据加载与 checkpoint 保存/加载。
# 对应文档: docs/training.md
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
    """从一维 token ID 数组中随机采样 batch：x 为输入序列，y 为对应下一个 token（右移 1）。"""
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
    """将 model、optimizer 的 state_dict 与当前 iteration 写入 out（文件路径或文件对象）。"""
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
    """从 src 加载 checkpoint，恢复 model 与 optimizer，返回保存的 iteration。"""
    state = torch.load(src, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]
