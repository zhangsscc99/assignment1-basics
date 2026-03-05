# Transformer LM：SwiGLU、Block 与整体结构

对应代码文件：`cs336_basics/transformer.py`

---

## 1. 背景知识

### 1.1 SwiGLU（门控前馈）

- 公式：**FFN(x) = W2 ( SiLU(W1 x) ⊙ W3 x )**
- 即：一路 W1→SiLU，一路 W3，两路逐元相乘后再用 W2 投影回 d_model。
- d_ff 通常取约 8/3 * d_model，并向上取整到 64 的倍数（利于 GPU）。
- 比单纯 MLP(W1, ReLU, W2) 更常用在现代 LLM 中。

### 1.2 Pre-Norm Transformer Block

- 每个 block 两子层：**MHA** 和 **FFN**。
- **Pre-norm**：先 norm 再子层，再残差。即  
  - `x = x + Attn(RMSNorm(x))`  
  - `x = x + FFN(RMSNorm(x))`
- 最后一层 block 之后再做一次 RMSNorm，再进 LM head。

### 1.3 整体 LM 结构

- **Token embedding**：id → d_model 向量。
- **N 个 Transformer block**（每块含 MHA + FFN，均带 RoPE 与 causal mask）。
- **Final RMSNorm** → **LM head**（线性层 d_model → vocab_size）得到 logits。
- 训练时对 logits 做 cross-entropy；推理时取最后一位置 logits 做采样生成。

---

## 2. 代码结构概览

```
transformer.py
├── class SwiGLU           # W1,W3 上投影，SiLU 与门控乘，W2 下投影
├── class TransformerBlock # ln1 → attn → 残差 → ln2 → ffn → 残差
└── class TransformerLM    # embed → N×Block → ln_final → lm_head
```

---

## 3. 关键实现说明

### 3.1 SwiGLU

- w1: (d_model → d_ff)，w2: (d_ff → d_model)，w3: (d_model → d_ff)。
- forward: `w2(silu(w1(x)) * w3(x))`。

### 3.2 TransformerBlock

- 先 `x + attn(ln1(x))`，再 `x + ffn(ln2(x))`；attn 和 ffn 内部不再做 norm。

### 3.3 TransformerLM

- token_embeddings → 逐层 Block（传入 token_positions 用于 RoPE）→ ln_final → lm_head。
- 若未传 token_positions，则用 0..seq_len-1。

---

## 4. 与作业的对应关系

- **§3.5.2** SwiGLU 与 d_ff 取法。
- **§3.5** Pre-norm block 结构。
- **§3.1** 整体 LM：embedding → blocks → final norm → logits。

---

## 5. 学习建议

1. 画一张从 token id 到 logits 的数据流图。
2. 对照论文/作业图确认 pre-norm 与残差的位置。
3. 理解为什么 RoPE 只在 Q、K 上、而 V 不需要位置信息。
