# 优化器与学习率调度

对应代码文件：`cs336_basics/optimizer.py`

---

## 1. 背景知识

### 1.1 AdamW

- **Adam**：一阶矩 m、二阶矩 v，用梯度 g 更新：m←β1*m+(1-β1)*g，v←β2*v+(1-β2)*g²；步长用偏差修正后的 m/(√v+ε)。
- **AdamW**：在 Adam 的参数更新之后，再对参数做 **weight decay**：θ ← θ - α·λ·θ（与梯度更新解耦）。
- 超参：lr α，β1/β2（常用 0.9/0.999），eps（如 1e-8），weight_decay λ。

### 1.2 Cosine 学习率 + Warmup

- **Warmup**：前 Tw 步线性从 0 增到 α_max。
- **Cosine**：Tw 到 Tc 步按余弦从 α_max 降到 α_min；t > Tc 后保持 α_min。
- 公式：progress = (t - Tw) / (Tc - Tw)，lr = α_min + 0.5*(1+cos(π*progress))*(α_max - α_min)。

---

## 2. 代码结构概览

```
optimizer.py
├── get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
└── class AdamW(Optimizer)   # 继承 torch.optim.Optimizer，维护 m/v 与 weight decay
```

---

## 3. 关键实现说明

### 3.1 get_lr_cosine_schedule

- it < warmup：线性增加。
- it > cosine_cycle_iters：返回 min_lr。
- 否则按余弦公式计算。

### 3.2 AdamW.step

- 对每个参数：若无梯度则跳过；否则更新 m、v，算偏差修正系数，用 m/(√v+ε) 更新参数，再乘 (1 - lr*wd) 做 weight decay。

---

## 4. 与作业的对应关系

- **§4.3** AdamW 算法。
- **§4.4** Cosine + warmup 学习率调度。

---

## 5. 学习建议

1. 对照作业 Algorithm 1 逐步看 step() 里每一行。
2. 画一条 lr 随 step 变化的曲线（warmup + cosine + 平台）。
