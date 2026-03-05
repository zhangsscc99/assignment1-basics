# BPE 分词器：知识与代码说明

对应代码文件：`cs336_basics/bpe.py`

---

## 1. 背景知识

### 1.1 为什么要分词？

- 语言模型按 **token** 处理文本，不能直接吃字符或整句。
- **词表** 有限（如 1 万），每个 token 一个整数 ID；训练/推理时用 ID 序列。

### 1.2 三种粒度

| 粒度 | 优点 | 缺点 |
|------|------|------|
| **词 (word)** | 语义单位清晰 | 词表巨大、OOV 多 |
| **字/字节 (char/byte)** | 无 OOV、词表小 (256) | 序列很长、难学 |
| **子词 (subword, BPE)** | 折中：压缩序列、控制词表 | 需要先“训练”词表 |

### 1.3 Byte-level BPE（本作业）

- 文本先转成 **UTF-8 字节序列**（0–255），再在字节序列上做 BPE。
- **初始词表**：特殊 token（如 `<|endoftext|>`）+ 256 个单字节。
- **合并规则**：反复把“出现最多的相邻字节对”合并成一个新 token，直到词表达到目标大小（如 10000）。
- **好处**：任何 Unicode 都能表示，不会 OOV；合并后常见子词一个 ID，序列变短。

### 1.4 预分词 (Pre-tokenization)

- 不能直接在整篇文本上数“相邻字节对”，否则会把不同词之间的边界也合并掉。
- 做法：先用**预分词**（如按空格+正则）切成“词/片段”，**只在每个片段内部**数相邻对并合并。
- 本作业使用 GPT-2 风格的正则（`PAT`），把文本切成类似单词、数字、标点的片段。

### 1.5 编码与解码

- **编码 (encode)**：文本 → 按预分词切段 → 每段内按 merge 顺序应用合并 → 得到 ID 序列。
- **解码 (decode)**：ID 序列 → 查词表得到字节串 → 拼成字节再 UTF-8 解码成字符串。
- **特殊 token**：如 `<|endoftext|>` 在编码时不被拆开，始终对应一个 ID；训练时用其分隔文档，避免跨文档合并。

---

## 2. 代码结构概览

```
bpe.py
├── PAT                    # 预分词正则（GPT-2 风格）
├── train_bpe()            # 训练 BPE，得到 vocab + merges
├── _replace_pair()        # 辅助：在词内把某一对 id 替换成新 id
├── class Tokenizer        # 编码/解码器
│   ├── __init__           # 用 vocab + merges + special_tokens 构造
│   ├── from_files()       # 从 vocab.json + merges.txt 加载
│   ├── encode()           # 文本 → id 列表（含特殊 token 处理）
│   ├── _encode_segment()  # 无特殊 token 的一段文本的编码
│   ├── _bpe_encode_bytes()# 单段字节序列应用 merges 得到 id 列表
│   ├── encode_iterable() # 流式编码大文件（省内存）
│   └── decode()           # id 列表 → 文本
├── _load_vocab_from_file()   # 从 JSON 加载词表（支持本格式与 GPT-2 格式）
└── _load_merges_from_file()  # 从 merges 文件加载合并规则
```

---

## 3. 关键逻辑说明

### 3.1 `train_bpe`

1. **按特殊 token 切分语料**：保证不会跨文档合并。
2. **预分词 + 统计**：对每段用 `PAT` 得到“词”，每个词用 UTF-8 转成字节元组，统计 `word_counts`。
3. **初始词表**：`special_tokens` 的 ID 先占位，再 0–255 各一个 ID；得到 `byte_to_id`。
4. **把词表成 ID 序列**：每个“词”变成一串 id，得到 `id_word_counts`（词 → 出现次数）。
5. **循环直到词表满**：
   - 统计所有“相邻 id 对”的出现次数；
   - 选出现最多的一对（相同则按字节序取更大）；
   - 把该对记入 `merges`，新 token 加入 `vocab`；
   - 在所有词里把该 id 对替换成新 id，更新 `id_word_counts`。
6. 返回 `vocab`（id → bytes）和 `merges`（(left_bytes, right_bytes) 列表）。

### 3.2 `Tokenizer.encode`

- 若没有 special_tokens：整段用 `_encode_segment`（预分词 + 每段 `_bpe_encode_bytes`）。
- 若有：用正则把文本按 special_tokens 拆成 `[段1, 特殊1, 段2, 特殊2, ...]`，奇下标是普通段（编码），偶下标是特殊 token（直接查 id）。

### 3.3 `_bpe_encode_bytes`

- 输入是一段字节（一个预分词单元）。
- 初始化为单字节列表，然后**按 merges 顺序**从左到右扫描，能合并就合并成一整块。
- 最后每块在词表里查 id，得到 id 列表。

### 3.4 `decode`

- 每个 id 用 `vocab` 查成 bytes，拼成一个大字节串，再 `decode("utf-8", errors="replace")` 成字符串。
- 非法 UTF-8 会用替换字符，避免报错。

---

## 4. 与作业的对应关系

- **§2.4 BPE 训练**：`train_bpe` 对应；初始词表、预分词、按对合并、 tie-break 按字节序。
- **§2.6 编码/解码**：`encode` / `decode` / `encode_iterable`；特殊 token、流式大文件。
- **§2.7 实验**：用本 Tokenizer 做压缩比、吞吐、tokenize 整份数据等。

---

## 5. 学习建议

1. 先用小语料、小 `vocab_size` 跑一次 `train_bpe`，打印前几条 `merges`，看合并的是哪些字节对。
2. 对一句简单英文做 `encode`，再 `decode`，确认 round-trip；再试带 `<|endoftext|>` 的字符串。
3. 理解“为什么必须按特殊 token 切分”“为什么预分词只在段内合并”。
