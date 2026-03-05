#!/usr/bin/env python3
"""Tokenize a text file using trained BPE and save as uint16 numpy array."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.bpe import Tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to raw text file")
    p.add_argument("--vocab", required=True, help="Path to vocab.json (from train_bpe)")
    p.add_argument("--merges", required=True, help="Path to merges.txt")
    p.add_argument("--output", required=True, help="Output .npy path (uint16)")
    p.add_argument("--special_tokens", default=["<|endoftext|>"], nargs="+")
    args = p.parse_args()
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)
    ids = []
    with open(args.input, encoding="utf-8", errors="replace") as f:
        for tid in tokenizer.encode_iterable(f):
            ids.append(tid)
    arr = np.array(ids, dtype=np.uint16)
    np.save(args.output, arr)
    print(f"Saved {len(arr)} tokens to {args.output}")


if __name__ == "__main__":
    main()
