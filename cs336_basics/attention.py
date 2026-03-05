# Scaled dot-product attention, RoPE, multi-head causal self-attention.
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch import Tensor

from .nn import Linear, softmax


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    # Q, K: (..., seq_q, d_k), V: (..., seq_k, d_v). mask: (..., seq_q, seq_k), True = attend.
    d_k = Q.size(-1)
    scale = d_k ** -0.5
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return einsum(attn, V, "... q k, ... k d -> ... q d")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # Precompute cos/sin for positions 0..max_seq_len-1. Shape (max_seq_len, d_k/2) each.
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # x: (..., seq_len, d_k), token_positions: (..., seq_len)
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x1_ = x1 * cos - x2 * sin
        x2_ = x1 * sin + x2 * cos
        return torch.stack([x1_, x2_], dim=-1).flatten(-2)


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)
        if token_positions is None:
            token_positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        causal = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        causal = causal.expand(B, self.num_heads, S, S)
        out = scaled_dot_product_attention(q, k, v, mask=causal)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.output_proj(out)
