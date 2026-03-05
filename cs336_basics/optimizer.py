# AdamW 与学习率调度（cosine + warmup）。
# 对应文档: docs/optimizer.md
from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch.optim import Optimizer


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Warmup 阶段线性增到 max，再 cosine 衰减到 min，超过 cycle 后保持 min。"""
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("Betas must be in [0, 1)")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        """一步 AdamW：更新 m/v，用偏差修正后的步长更新参数，再做 weight decay。"""
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]
                g = p.grad.data
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).add_(g.pow(2), alpha=1 - beta2)
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data.addcdiv_(m, v.sqrt().add(eps), value=-alpha_t)
                p.data.mul_(1 - lr * wd)
        return loss
