#!/usr/bin/env python3
"""
Train Transformer LM on tokenized data.
Usage (example):
  uv run python scripts/train.py --data data/train_tokens.npy --config configs/tinystories.json --out_dir checkpoints
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# Add project root so we can import cs336_basics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.nn import cross_entropy, gradient_clipping
from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.training import get_batch, save_checkpoint, load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to tokenized train data (.npy, uint16)")
    p.add_argument("--valid_data", default=None, help="Path to tokenized valid data (.npy)")
    p.add_argument("--config", required=True, help="Path to model config JSON")
    p.add_argument("--out_dir", default="checkpoints", help="Checkpoint directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr_max", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = p.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    vocab_size = config["vocab_size"]
    context_length = config["context_length"]
    d_model = config["d_model"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    d_ff = config.get("d_ff", (8 * d_model // 3 + 63) // 64 * 64)
    rope_theta = config.get("rope_theta", 10000.0)

    device = torch.device(args.device)
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr_max, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    # Load data (memmap for large files)
    data = np.load(args.data, mmap_mode="r")
    if data.dtype != np.uint16:
        data = data.astype(np.uint16)
    valid_data = None
    if args.valid_data and os.path.exists(args.valid_data):
        valid_data = np.load(args.valid_data, mmap_mode="r").astype(np.uint16)

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.perf_counter()
    for step in range(start_iter, args.steps):
        lr = get_lr_cosine_schedule(step, args.lr_max, args.lr_min, args.warmup_iters, args.steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = get_batch(data, args.batch_size, context_length, str(device))
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"step {step+1} loss {loss.item():.4f} lr {lr:.2e}")
        if valid_data is not None and (step + 1) % args.eval_every == 0:
            with torch.no_grad():
                vx, vy = get_batch(valid_data, args.batch_size, context_length, str(device))
                vlogits = model(vx)
                vloss = cross_entropy(vlogits.view(-1, vocab_size), vy.view(-1))
            print(f"  valid loss {vloss.item():.4f}")
        if (step + 1) % args.save_every == 0:
            path = os.path.join(args.out_dir, f"ckpt_{step+1}.pt")
            save_checkpoint(model, optimizer, step + 1, path)
            print(f"  saved {path}")
    elapsed = time.perf_counter() - t0
    print(f"Done. {args.steps} steps in {elapsed:.1f}s")
    save_checkpoint(model, optimizer, args.steps, os.path.join(args.out_dir, "ckpt_final.pt"))


if __name__ == "__main__":
    main()
