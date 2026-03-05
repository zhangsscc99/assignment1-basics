#!/usr/bin/env python3
"""Train BPE tokenizer on a text corpus and save vocab + merges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.bpe import train_bpe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to raw text file (e.g. TinyStories train)")
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--special_tokens", default=["<|endoftext|>"], nargs="+")
    p.add_argument("--out_dir", default="data", help="Directory to write vocab.json and merges.txt")
    args = p.parse_args()
    vocab, merges = train_bpe(
        args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_json = {str(i): list(b) for i, b in vocab.items()}
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f)
    with open(out_dir / "merges.txt", "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(",".join(str(b) for b in left) + " " + ",".join(str(b) for b in right) + "\n")
    print(f"Saved vocab ({len(vocab)} entries) and {len(merges)} merges to {out_dir}")


if __name__ == "__main__":
    main()
