# Byte-pair encoding: training and tokenizer (encode/decode).
# 对应文档: docs/bpe.md
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import regex as re_module

# GPT-2 风格预分词：按单词/数字/标点等切分，不在整句上直接合并字节对
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs: Any,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train byte-level BPE. Returns vocab: id -> bytes, merges: list of (left, right) in order.
    """
    path = Path(input_path)
    corpus = path.read_text(encoding="utf-8", errors="replace")

    # 按特殊 token 切分，避免跨文档合并（例如 <|endoftext|> 分隔的文档之间不合并）
    if special_tokens:
        pattern = "|".join(re.escape(s) for s in special_tokens)
        segments = re.split(pattern, corpus)
    else:
        segments = [corpus]

    # 预分词：每段用正则切出“词”，每个词转成 UTF-8 字节元组并统计出现次数
    word_counts: Counter[tuple[int, ...]] = Counter()
    for seg in segments:
        for m in re_module.finditer(PAT, seg):
            token = m.group(0)
            word_counts[tuple(token.encode("utf-8"))] += 1

    # 初始词表：先放特殊 token，再放 256 个单字节；id -> bytes
    vocab: dict[int, bytes] = {}
    next_id = 0
    for s in special_tokens:
        vocab[next_id] = s.encode("utf-8")
        next_id += 1
    num_special = next_id
    for b in range(256):
        vocab[next_id] = bytes([b])
        next_id += 1
    # 字节 b 的 id = num_special + b
    byte_to_id = {b: num_special + b for b in range(256)}

    # 把每个“词”表示成当前词表下的 id 元组，便于后续统计相邻对
    id_word_counts: Counter[tuple[int, ...]] = Counter()
    for word_bytes, count in word_counts.items():
        ids = tuple(byte_to_id[b] for b in word_bytes)
        if ids:
            id_word_counts[ids] += count

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        # 统计所有“相邻 id 对”的出现次数（按词加权）
        pair_counts: Counter[tuple[int, int]] = Counter()
        for word, count in id_word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += count

        if not pair_counts:
            break

        # 选出现次数最多的对；次数相同则按 (left_bytes, right_bytes) 字典序取更大
        def key(item: tuple[tuple[int, int], int]) -> tuple[int, bytes, bytes]:
            (left_id, right_id), cnt = item
            left_bytes = vocab[left_id]
            right_bytes = vocab[right_id]
            return (cnt, left_bytes, right_bytes)

        (left_id, right_id), _ = max(pair_counts.items(), key=key)
        left_bytes = vocab[left_id]
        right_bytes = vocab[right_id]
        merges.append((left_bytes, right_bytes))
        new_bytes = left_bytes + right_bytes
        new_id = next_id
        next_id += 1
        vocab[new_id] = new_bytes

        # 在所有词中，把连续的 (left_id, right_id) 替换成 new_id，得到新的词表表示
        new_id_word_counts: Counter[tuple[int, ...]] = Counter()
        for word, count in id_word_counts.items():
            new_word = _replace_pair(word, left_id, right_id, new_id)
            if new_word:
                new_id_word_counts[new_word] += count
        id_word_counts = new_id_word_counts

    return vocab, merges


def _replace_pair(
    word: tuple[int, ...],
    left_id: int,
    right_id: int,
    new_id: int,
) -> tuple[int, ...]:
    """将 word 中所有连续的 (left_id, right_id) 替换为 new_id，返回新 id 元组。"""
    if len(word) < 2:
        return word
    out: list[int] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == left_id and word[i + 1] == right_id:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class Tokenizer:
    """BPE 分词器：支持 encode/decode，以及可选的 special_tokens（编码时保持为单 token）。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self._vocab = dict(vocab)
        self._merges = list(merges)
        self._special_tokens = list(special_tokens or [])
        # 反向映射：bytes -> id，用于编码时查 id；若有重复 bytes 只保留第一个 id
        self._bytes_to_id: dict[bytes, int] = {}
        for i, b in self._vocab.items():
            if b not in self._bytes_to_id:
                self._bytes_to_id[b] = i
        # 若 special_tokens 不在词表中，则追加到词表末尾
        for s in self._special_tokens:
            b = s.encode("utf-8")
            if b not in self._bytes_to_id:
                new_id = max(self._vocab.keys(), default=-1) + 1
                self._vocab[new_id] = b
                self._bytes_to_id[b] = new_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        vocab = _load_vocab_from_file(vocab_filepath)
        merges = _load_merges_from_file(merges_filepath, vocab)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """文本 -> token id 列表。若设置了 special_tokens，会按它们切分并保留为单 token。"""
        if not text:
            return []
        if not self._special_tokens:
            return self._encode_segment(text)
        # 按特殊 token 切分并保留分隔符：得到 [段1, 特殊1, 段2, ...]
        pattern = "|".join(re.escape(s) for s in self._special_tokens)
        parts = re.split("(" + pattern + ")", text)
        ids = []
        for i in range(len(parts)):
            if i % 2 == 0:
                if parts[i]:
                    ids.extend(self._encode_segment(parts[i]))
            else:
                # delimiter
                ids.append(self._bytes_to_id[parts[i].encode("utf-8")])
        return ids

    def _encode_segment(self, text: str) -> list[int]:
        """对不含特殊 token 的一段文本编码：预分词后对每段应用 BPE merges。"""
        if not text:
            return []
        result: list[int] = []
        for m in re_module.finditer(PAT, text):
            token = m.group(0)
            result.extend(self._bpe_encode_bytes(token.encode("utf-8")))
        return result

    def _bpe_encode_bytes(self, b: bytes) -> list[int]:
        """对单个预分词单元（一段 bytes）应用 BPE：按 merges 顺序合并，再查 id。"""
        if not b:
            return []
        # 初始为单字节列表，然后按 merges 顺序从左到右合并
        tokens: list[bytes] = [bytes([x]) for x in b]
        for left, right in self._merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == left and tokens[i + 1] == right:
                    tokens = tokens[:i] + [left + right] + tokens[i + 2 :]
                else:
                    i += 1
        return [self._bytes_to_id[t] for t in tokens]

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        """流式编码：从字符串迭代器（如文件）逐块读入，按块/行产出 token id，适合大文件省内存。"""
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            # 按 special_tokens 切分并输出完整段；边界未完整 token 留在 buffer
            if self._special_tokens:
                pattern = "|".join(re.escape(s) for s in self._special_tokens)
                parts = re.split("(" + pattern + ")", buffer)
                # Last part might be incomplete (no trailing newline). Emit all but last.
                if len(parts) > 1 or (len(parts) == 1 and "\n" in buffer or buffer.endswith(tuple(self._special_tokens))):
                    for i in range(0, len(parts) - 1, 2):
                        seg = parts[i]
                        if seg:
                            for id_ in self._encode_segment(seg):
                                yield id_
                        if i + 1 < len(parts) and parts[i + 1]:
                            yield self._bytes_to_id[parts[i + 1].encode("utf-8")]
                    buffer = parts[-1] if len(parts) % 2 == 1 else ""
            else:
                # No special: emit full lines or large chunks to avoid boundary issues.
                while "\n" in buffer or len(buffer) > 8192:
                    if "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        for id_ in self._encode_segment(line + "\n"):
                            yield id_
                    else:
                        cut = buffer[:8192]
                        buffer = buffer[8192:]
                        for id_ in self._encode_segment(cut):
                            yield id_
        if buffer:
            for id_ in self._encode_segment(buffer):
                yield id_

    def decode(self, ids: list[int]) -> str:
        """id 列表 -> 查词表拼成字节串 -> UTF-8 解码；非法字节用替换字符。"""
        b = b"".join(self._vocab.get(i, b"") for i in ids)
        return b.decode("utf-8", errors="replace")


def _load_vocab_from_file(path: str | os.PathLike) -> dict[int, bytes]:
    """从 JSON 加载词表。支持本仓库格式 {"0": [97,98], ...} 或 GPT-2 风格 {"token": id}。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # 本仓库格式：键为 id 字符串，值为字节列表
    vocab: dict[int, bytes] = {}
    for k, v in data.items():
        try:
            idx = int(k)
            if isinstance(v, list):
                vocab[idx] = bytes(v)
            elif isinstance(v, str):
                vocab[idx] = v.encode("utf-8")
            else:
                vocab[idx] = v
        except (TypeError, ValueError):
            pass
    if vocab:
        return vocab
    # GPT-2 风格：键为 token 字符串，值为 id
    for k, v in data.items():
        idx = int(v)
        if k == "<|endoftext|>" or len(k) > 1:
            vocab[idx] = k.encode("utf-8")
        else:
            vocab[idx] = k.encode("utf-8")
    return vocab


def _load_merges_from_file(
    path: str | os.PathLike,
    vocab: dict[int, bytes] | None = None,
) -> list[tuple[bytes, bytes]]:
    """从 merges 文件加载合并规则。支持本格式（逗号分隔的字节整数）或 GPT-2 风格字符串。"""
    merges = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line or " " not in line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                a, b = parts
                # 本仓库格式：每行 "left_byte0,left_byte1 right_byte0,right_byte1"
                if "," in a or "," in b:
                    left = bytes(int(x) for x in a.split(","))
                    right = bytes(int(x) for x in b.split(","))
                    merges.append((left, right))
                else:
                    # GPT-2 style: tokens as in vocab (single chars). Decode via UTF-8.
                    merges.append((a.encode("utf-8"), b.encode("utf-8")))
    return merges
