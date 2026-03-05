# CS336 Assignment 1 完整 Walkthrough

## 1. 环境与测试

### 安装与运行测试

```bash
# 在 assignment1-basics 目录下
cd assignment1-basics

# 使用 uv（推荐）
uv run pytest

# 或使用 pip + pytest
pip install -e .
pytest
```

所有实现通过 `tests/adapters.py` 调用 `cs336_basics` 中的代码；跑通测试即说明作业要求的接口已实现。

---

## 2. 数据准备

### 2.1 下载数据（见 README）

```bash
mkdir -p data
cd data
# TinyStories
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
# OpenWebText 子集（可选）
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
cd ..
```

### 2.2 训练 BPE 分词器

```bash
# TinyStories，vocab_size=10000，特殊词 <|endoftext|>
uv run python scripts/train_bpe.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab_size 10000 \
  --special_tokens "<|endoftext|>" \
  --out_dir data
```

会生成 `data/vocab.json` 和 `data/merges.txt`。

### 2.3 将语料转成 token 序列（uint16）

```bash
# 训练集
uv run python scripts/tokenize_data.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab data/vocab.json \
  --merges data/merges.txt \
  --output data/train_tokens.npy

# 验证集
uv run python scripts/tokenize_data.py \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --vocab data/vocab.json \
  --merges data/merges.txt \
  --output data/valid_tokens.npy
```

---

## 3. 训练有哪些、怎么跑

### 3.1 训练类型概览

| 训练 | 数据 | 说明 |
|------|------|------|
| **TinyStories 主实验** | TinyStories train | 作业主流程：约 327M tokens，调学习率、batch size 等 |
| **低资源 / 调试** | TinyStories train | 少步数、小 batch（如 40M tokens）便于在 CPU/MPS 上跑 |
| **OpenWebText** | OWT train | 用同一模型配置在 OWT 上训练，交 leaderboard |

### 3.2 训练脚本用法

```bash
# 基础：TinyStories，默认步数 5000
uv run python scripts/train.py \
  --data data/train_tokens.npy \
  --valid_data data/valid_tokens.npy \
  --config configs/tinystories.json \
  --out_dir checkpoints \
  --device cuda \
  --batch_size 32 \
  --steps 5000 \
  --lr_max 1e-3 \
  --lr_min 1e-5 \
  --warmup_iters 100 \
  --eval_every 500 \
  --save_every 1000
```

**常用参数：**

- `--device`: `cuda` / `cpu` / `mps`（Apple Silicon）
- `--batch_size`: 显存/内存允许下尽量大（如 64、128）
- `--steps`: 总步数；作业建议约 327M tokens ≈ `batch_size * steps * context_length`
- `--lr_max`, `--lr_min`, `--warmup_iters`: 学习率与 warmup
- `--resume checkpoints/ckpt_1000.pt`: 从该 checkpoint 继续训练

**低资源示例（约 40M tokens）：**

```bash
uv run python scripts/train.py \
  --data data/train_tokens.npy \
  --valid_data data/valid_tokens.npy \
  --config configs/tinystories.json \
  --out_dir checkpoints \
  --device cpu \
  --batch_size 32 \
  --steps 5000 \
  --eval_every 500 \
  --save_every 1000
```

（`configs/tinystories.json` 里 `context_length` 为 256，32×5000×256 = 40,960,000 tokens。）

---

## 4. 生成与评估

### 4.1 生成文本

```bash
uv run python scripts/generate.py \
  --checkpoint checkpoints/ckpt_final.pt \
  --config configs/tinystories.json \
  --vocab data/vocab.json \
  --merges data/merges.txt \
  --prompt "Once upon a time" \
  --max_tokens 256 \
  --temperature 0.8 \
  --top_p 0.9
```

- `--temperature`: 越小越确定，越大越随机。
- `--top_p`: nucleus sampling 的 p；1.0 表示不做 top-p。

### 4.2 Perplexity

验证集上算 loss，再算 perplexity：

- 在训练脚本里已经按 `eval_every` 打印了 `valid loss`。
- 公式：`perplexity = exp(valid_loss)`（loss 为 per-token 交叉熵）。

---

## 5. 作业流程小结

1. **实现**：BPE、Tokenizer、Transformer LM、损失/优化器/数据/checkpoint 等（已写在 `cs336_basics/` 并通过 adapters 接测试）。
2. **数据**：下载 → 训 BPE → 生成 `train_tokens.npy` / `valid_tokens.npy`。
3. **训练**：用 `scripts/train.py` 跑 TinyStories（及可选 OWT），调学习率、batch size、步数。
4. **生成与报告**：用 `scripts/generate.py` 生成样例，记录 valid loss / perplexity 与实验设置，写 writeup。

---

## 6. 目录结构速览

```
assignment1-basics/
├── cs336_basics/       # 你的实现
│   ├── bpe.py          # BPE 训练 + Tokenizer
│   ├── nn.py           # Linear, Embedding, RMSNorm, SiLU, softmax, cross_entropy, gradient_clipping
│   ├── attention.py    # SDPA, RoPE, MultiheadSelfAttention
│   ├── transformer.py  # SwiGLU, TransformerBlock, TransformerLM
│   ├── optimizer.py    # AdamW, get_lr_cosine_schedule
│   └── training.py     # get_batch, save/load_checkpoint
├── tests/
│   └── adapters.py     # 调用 cs336_basics 的适配层
├── scripts/
│   ├── train_bpe.py    # 训 BPE → vocab.json, merges.txt
│   ├── tokenize_data.py # 文本 → train/valid_tokens.npy
│   ├── train.py        # 训练 LM
│   └── generate.py    # 从 checkpoint 生成文本
├── configs/
│   └── tinystories.json # 模型超参
└── data/               # 数据与分词结果（需自行下载与生成）
```

按上述顺序：**数据 → BPE → tokenize → train → generate** 即可完成作业主流程。
