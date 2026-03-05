#!/usr/bin/env python3
"""
Generate text from a trained Transformer LM checkpoint.
Usage (example):
  uv run python scripts/generate.py --checkpoint checkpoints/ckpt_final.pt --tokenizer_dir data --prompt "Once upon a time" --max_tokens 256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.transformer import TransformerLM
from cs336_basics.bpe import Tokenizer
from cs336_basics.nn import softmax


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > p
    sorted_probs = sorted_probs.masked_fill(mask, 0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(sorted_probs, 1)
    return sorted_idx.gather(-1, idx).squeeze(-1)


def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    device: torch.device | None = None,
) -> str:
    if device is None:
        device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]
    context_length = model.context_length
    for _ in range(max_tokens - 1):
        x = torch.tensor([ids[-context_length:]], device=device, dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        logits = logits[0, -1, :] / temperature
        probs = softmax(logits.unsqueeze(0), dim=-1)[0]
        if top_p < 1.0:
            next_id = sample_top_p(probs.unsqueeze(0), top_p).item()
        else:
            next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    return tokenizer.decode(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    p.add_argument("--config", required=True, help="Path to model config JSON")
    p.add_argument("--vocab", required=True, help="Path to vocab JSON")
    p.add_argument("--merges", required=True, help="Path to merges file")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    device = torch.device(args.device)
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config.get("d_ff", (8 * config["d_model"] // 3 + 63) // 64 * 64),
        rope_theta=config.get("rope_theta", 10000.0),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    eos_id = tokenizer._bytes_to_id.get(b"<|endoftext|>")
    text = generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=eos_id,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
