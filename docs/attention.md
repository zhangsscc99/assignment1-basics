# 注意力与 RoPE：知识与代码说明

对应代码文件：`cs336_basics/attention.py`

---

## 1. 背景知识

### 1.1 Scaled Dot-Product Attention (SDPA)

- 公式：**Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V**
- **Q**: (seq_q, d_k)，查询；**K**: (seq_k, d_k)，键；**V**: (seq_k, d_v)，值。
- 除以 **sqrt(d_k)** 防止点积过大导致 softmax 梯度变小。
- **Mask**：在 softmax 前把不能 attend 的位置置为 -inf（True 表示可 attend）。

### 1.2 Multi-Head Self-Attention (MHA)

- 把 d_model 拆成 **num_heads** 个头，每个头独立做 SDPA，再拼起来过一层 **output 投影**。
- **Q、K、V** 各由一个线性层从 x 得到；本实现中 d_k = d_v = d_model / num_heads。
- **Causal mask**：位置 i 只能看 0..i，保证自回归生成时不会“看到未来”。

### 1.3 Rotary Position Embedding (RoPE)

- 不显式加位置向量，而是对 **Q、K** 按位置做旋转，使注意力分数只依赖 **相对位置**。
- 把 d_k 维分成 d_k/2 对，每对 (q_{2k-1}, q_{2k}) 按位置 i 旋转角度 i * theta^{-2k/d_k}；K 同样旋转。
- 效果：内积 (RoPE(q_i), RoPE(k_j)) 只与 i-j 有关。无额外可学习参数，可预计算 cos/sin 表。

### 1.4 本文件结构

- **scaled_dot_product_attention**：通用 SDPA，支持任意 batch 维和 mask。
- **RotaryPositionalEmbedding**：预计算 cos/sin，forward 时按 token_positions 取表并施加旋转。
- **MultiheadSelfAttention**：Q/K/V 投影 → 拆头 → 对 Q、K 做 RoPE → causal SDPA → 合并头 → output 投影。

---

## 2. 代码结构概览

```
attention.py
├── scaled_dot_product_attention(Q, K, V, mask=None)
├── class RotaryPositionalEmbedding   # 预计算 cos/sin，forward 对 x 按位置旋转
└── class MultiheadSelfAttention     # 含 Q/K/V/out 投影、RoPE、causal SDPA
```

---

## 3. 关键实现说明

### 3.1 SDPA

- 用 einsum 算 `Q K^T`，乘 scale=1/sqrt(d_k)；mask 为 False 处填 -inf；对最后一维 softmax 后与 V 乘。

### 3.2 RoPE

- **inv_freq**: 维度 d_k/2，值为 1/theta^(2k/d_k)。
- **freqs**: (max_seq_len, d_k/2)，位置 × inv_freq；cos/sin 缓存在 buffer。
- **forward**：按 token_positions 取 cos、sin；x 的奇数/偶数字维度配对做旋转：x1'=x1*cos-x2*sin, x2'=x1*sin+x2*cos。

### 3.3 MultiheadSelfAttention

- 先线性得到 Q、K、V，再 `rearrange` 成 (B, num_heads, S, d_k)。
- 对 Q、K 应用同一 RoPE（token_positions 可为 None，则用 0..S-1）。
- 构造下三角 causal mask，调用 SDPA，再合并头并过 output_proj。

---

## 4. 与作业的对应关系

- **§3.5.4** SDPA 与 mask。
- **§3.5.3** RoPE 公式与实现。
- **§3.5.5** Causal multi-head self-attention，RoPE 只加在 Q、K 上。

---

## 5. 学习建议

1. 画一张图：从 x 到 Q/K/V，到每个头的 attention，到 concat 和 output。
2. 用 2×2 旋转矩阵理解 RoPE 对一对维度的变换。
3. 试一个小序列，手算一次 causal mask 下某一行的 attention 权重。
