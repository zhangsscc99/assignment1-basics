# 数据加载与 Checkpoint

对应代码文件：`cs336_basics/training.py`

---

## 1. 背景知识

### 1.1 语言模型训练数据形式

- 语料被 tokenize 成**一维 token ID 序列**（可能多篇文档用特殊 token 拼接）。
- 训练时每次取**一段连续子序列**作为输入，**右移一位**作为目标（预测下一个 token）。
- 因此从位置 s 取长度为 L 的序列：input = [s..s+L-1]，target = [s+1..s+L]。

### 1.2 get_batch

- 从一维数组里**随机**采样多个起始位置（不重复也可，作业里是随机），每个起始位置取一段长度为 context_length 的连续子序列。
- 返回两个张量：input 和 target，形状均为 (batch_size, context_length)，且 target[i, j] = input[i, j+1] 的“下一个 token”（即 target 整段相对 input 右移 1）。
- 数据可以来自 **np.memmap**，这样大文件不会一次性进内存。

### 1.3 Checkpoint

- **保存**：model.state_dict()、optimizer.state_dict()、当前 iteration，用 torch.save 写入文件。
- **加载**：torch.load 后分别 load_state_dict 到 model 和 optimizer，并返回 iteration，便于恢复训练和学习率调度。

---

## 2. 代码结构概览

```
training.py
├── get_batch(dataset, batch_size, context_length, device)
├── save_checkpoint(model, optimizer, iteration, out)
└── load_checkpoint(src, model, optimizer) -> iteration
```

---

## 3. 关键实现说明

### 3.1 get_batch

- 若数据集长度 n < context_length+1 则报错。
- max_start = n - context_length - 1；在 [0, max_start] 上均匀随机采样 batch_size 个起点。
- x[s:s+L]，y[s+1:s+L+1]；转为 LongTensor 并放到指定 device。

### 3.2 save_checkpoint / load_checkpoint

- 字典含 "model"、"optimizer"、"iteration"；load 时用 map_location="cpu" 以便在任意设备上加载后再 to(device)。

---

## 4. 与作业的对应关系

- **§5.1** 数据加载形式与 get_batch 接口。
- **§5.2** checkpoint 内容与保存/恢复方式。

---

## 5. 学习建议

1. 用一个小数组（如 arange(100)）和 context_length=5 试一次 get_batch，检查 x 与 y 是否差 1。
2. 理解为什么大语料要用 memmap：训练脚本里用 np.load(..., mmap_mode="r") 或 np.memmap 传入 get_batch。
