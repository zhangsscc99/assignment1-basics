# 代码与文档索引

每个 Markdown 文件对应 `cs336_basics/` 下的一个 Python 文件，介绍相关背景知识和代码结构。

| 文档 | 对应代码 | 内容概要 |
|------|----------|----------|
| [bpe.md](bpe.md) | `bpe.py` | BPE 训练、Tokenizer 编码/解码、预分词与特殊 token |
| [nn.md](nn.md) | `nn.py` | Linear、Embedding、RMSNorm、SiLU、softmax、cross_entropy、gradient_clipping |
| [attention.md](attention.md) | `attention.py` | SDPA、RoPE、Causal Multi-Head Self-Attention |
| [transformer.md](transformer.md) | `transformer.py` | SwiGLU、Transformer Block、完整 Transformer LM |
| [optimizer.md](optimizer.md) | `optimizer.py` | AdamW、Cosine + Warmup 学习率调度 |
| [training.md](training.md) | `training.py` | get_batch、save/load_checkpoint |

**建议阅读顺序**（按依赖）：nn → attention → transformer → optimizer → training → bpe（或先 bpe 再其他）。

代码文件中已在文件头注明「对应文档: docs/xxx.md」，并在关键处加了中文注释。
