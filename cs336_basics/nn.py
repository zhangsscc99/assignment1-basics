# NN 基础模块：Linear, Embedding, RMSNorm, SiLU, softmax, cross_entropy, gradient_clipping.
# 对应文档: docs/nn.md
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    """沿 dim 做 softmax；先减该维最大值再 exp，数值稳定。"""
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """平均交叉熵：inputs (..., vocab_size)，targets (...) 为类别 id；先 log_softmax 再 gather 取负。"""
    dim = -1
    x = inputs - inputs.max(dim=dim, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(x).sum(dim=dim, keepdim=True) + 1e-12)
    log_prob = inputs - log_sum_exp
    nll = -torch.gather(log_prob, dim, targets.unsqueeze(dim)).squeeze(dim)
    return nll.mean()


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """将所有参数的梯度视为一个向量，若 L2 范数 > max_l2_norm 则整体缩放（原地修改 grad）。"""
    params = list(parameters)
    total_norm_sq = sum(p.grad.data.pow(2).sum().item() for p in params if p.grad is not None)
    total_norm = total_norm_sq ** 0.5
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(scale)


class Linear(nn.Module):
    """无 bias 线性层：y = x @ weight.T，weight 形状 (out_features, in_features)。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=(2.0 / (in_features + out_features)) ** 0.5, a=-3, b=3)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    """词嵌入：token_ids 索引 weight，得到 (..., embedding_dim)。weight 形状 (num_embeddings, embedding_dim)。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """RMSNorm：对最后一维 x 做 x / sqrt(mean(x^2)+eps) * weight；先转 float32 再算，避免溢出。"""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.float()
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).pow(0.5)
        out = x / rms * self.weight
        return out.to(in_dtype)


def silu(x: Tensor) -> Tensor:
    """SiLU (Swish): x * sigmoid(x)。"""
    return x * torch.sigmoid(x)
