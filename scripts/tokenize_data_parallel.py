#!/usr/bin/env python3
"""并行分词：多进程处理大文件"""
import argparse
import multiprocessing as mp
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cs336_basics.bpe import Tokenizer


def process_chunk(args):
    """处理文本块"""
    chunk_text, vocab_path, merges_path, special_tokens = args
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    return tokenizer.encode(chunk_text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--merges", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--special_tokens", default=["<|endoftext|>"], nargs="+")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk_size", type=int, default=5_000_000, help="每块字符数")
    args = p.parse_args()

    print(f"读取文件: {args.input}")
    with open(args.input, encoding="utf-8", errors="replace") as f:
        text = f.read()

    print(f"文件大小: {len(text):,} 字符")
    print(f"使用 {args.workers} 个进程并行处理...")

    # 分块
    chunks = []
    for i in range(0, len(text), args.chunk_size):
        chunks.append((text[i:i+args.chunk_size], args.vocab, args.merges, args.special_tokens))

    print(f"分成 {len(chunks)} 块")

    # 并行处理
    with mp.Pool(args.workers) as pool:
        results = pool.map(process_chunk, chunks)

    # 合并结果
    ids = []
    for r in results:
        ids.extend(r)

    arr = np.array(ids, dtype=np.uint16)
    print(f"写入 {args.output}...")
    np.save(args.output, arr)
    print(f"完成。保存了 {len(arr):,} tokens")


if __name__ == "__main__":
    main()
