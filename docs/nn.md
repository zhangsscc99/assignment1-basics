# NN 基础模块：知识与代码说明

对应代码文件：`cs336_basics/nn.py`

---

## 1. 背景知识

### 1.1 Linear（线性层）

- 变换：**y = x W^T**（或写作 y = x @ W.T），其中 x 形状 `(..., in_features)`，W 形状 `(out_features, in_features)`。
- 作业要求：**无 bias**，与多数现代 LLM 一致；参数用 **truncated normal** 初始化，方差与 in+out 维度相关，避免梯度爆炸/消失。

### 1.2 Embedding（词嵌入）

- 把 **token id**（整数）映射成 **连续向量**：查表 `weight[id]`，形状 `(num_embeddings, embedding_dim)`。
- 前向就是索引：`output = weight[token_ids]`，支持任意 batch/序列维度。
- 初始化：truncated normal，方差 1。

### 1.3 RMSNorm（Root Mean Square Layer Normalization）

- 对最后一维做归一化：先算 **RMS(a) = sqrt(mean(a^2) + eps)**，再 **a / RMS(a) * gain**，其中 gain 是可学习向量（与 d_model 同长）。
- 与 LayerNorm 区别：不做减均值，只做缩放；计算更简单，很多 LLM 用 RMSNorm。
- 数值：先转 float32 再算 RMS，最后转回原 dtype，避免溢出。

### 1.4 SiLU（Swish）

- 激活函数：**SiLU(x) = x * sigmoid(x)**。
- 在 SwiGLU 里会用到；作业允许这里用 `torch.sigmoid` 保证数值稳定。

### 1.5 Softmax

- 把某维变成概率分布：**exp(x_i) / sum(exp(x_j))**。
- **数值稳定**：先减该维最大值再 exp，再归一化，避免 exp 溢出。

### 1.6 Cross-Entropy Loss

- 分类损失：**-log(softmax(logits)[target]**。
- 实现时用 **log_softmax** 再 gather target 位置取负，最后对 batch 求平均；同样先减最大值再 log-sum-exp，避免溢出。

### 1.7 Gradient Clipping

- 若所有参数梯度的 **L2 范数** 超过阈值 M，则整体缩放梯度使范数 = M，防止梯度爆炸。
- 对每个参数 **原地** 修改 `grad`。

---

## 2. 代码结构概览

```
nn.py
├── softmax(x, dim)              # 沿 dim 做 softmax，减最大值稳定
├── cross_entropy(inputs, targets)
├── gradient_clipping(parameters, max_l2_norm, eps=1e-6)
├── class Linear                  # 无 bias 线性层，weight (out, in)
├── class Embedding               # id -> 向量，weight (vocab_size, d_model)
├── class RMSNorm                 # 最后一维 RMS 归一化 + 可学习 gain
└── def silu(x)                   # x * sigmoid(x)
```

---

## 3. 关键实现说明

### 3.1 Linear

- `weight`: `nn.Parameter`，形状 `(out_features, in_features)`，满足 **y = x @ weight.T**。
- 初始化：`trunc_normal_(std=sqrt(2/(in+out)), a=-3, b=3)`。

### 3.2 RMSNorm

- `forward`: 先转 float32 → 算 `rms = sqrt(mean(x^2) + eps)` → `x / rms * self.weight` → 转回原 dtype。

### 3.3 cross_entropy

- `inputs`: `(..., vocab_size)`，`targets`: `(...)` 整数。
- 先减最大值、算 log_sum_exp，得到 log_prob；再 `-gather(log_prob, targets).mean()`。

### 3.4 gradient_clipping

- 先算所有参数梯度的 L2 范数；若 > max_l2_norm，则每个 `grad *= max_l2_norm / (norm + eps)`。

---

## 4. 与作业的对应关系

- **§3.4** Linear / Embedding 及初始化。
- **§3.5.1** RMSNorm 公式与实现。
- **§3.5.2** SiLU 在 SwiGLU 中的使用。
- **§3.5.4** softmax（数值稳定）用于 attention。
- **§4.1** cross_entropy 与 perplexity 关系。
- **§4.5** gradient clipping。

---

## 5. 学习建议

1. 手写一维 softmax（减 max 再 exp、归一化），再对照代码。
2. 用小 tensor 试一次 `cross_entropy`，和 `F.cross_entropy` 对比。
3. 理解 Linear 的 `(out, in)` 与 `x @ W.T` 的维度对应关系。
