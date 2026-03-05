# Byte-pair encoding: training and tokenizer (encode/decode).
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import regex as re_module

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

    # Split on special tokens so we never merge across them.
    if special_tokens:
        pattern = "|".join(re.escape(s) for s in special_tokens)
        segments = re.split(pattern, corpus)
    else:
        segments = [corpus]

    # Pre-tokenize each segment and get word -> count. Each word is tuple of byte values (0-255).
    word_counts: Counter[tuple[int, ...]] = Counter()
    for seg in segments:
        for m in re_module.finditer(PAT, seg):
            token = m.group(0)
            word_counts[tuple(token.encode("utf-8"))] += 1

    # Initial vocab: special tokens then bytes 0..255. id -> bytes.
    vocab: dict[int, bytes] = {}
    next_id = 0
    for s in special_tokens:
        vocab[next_id] = s.encode("utf-8")
        next_id += 1
    num_special = next_id
    for b in range(256):
        vocab[next_id] = bytes([b])
        next_id += 1
    # Byte b has id = num_special + b
    byte_to_id = {b: num_special + b for b in range(256)}

    # Represent each word as tuple of token ids.
    id_word_counts: Counter[tuple[int, ...]] = Counter()
    for word_bytes, count in word_counts.items():
        ids = tuple(byte_to_id[b] for b in word_bytes)
        if ids:
            id_word_counts[ids] += count

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        # Count pairs (consecutive ids in all words).
        pair_counts: Counter[tuple[int, int]] = Counter()
        for word, count in id_word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += count

        if not pair_counts:
            break

        # Best pair: max count, then lexicographically greater (by bytes).
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

        # Replace every (left_id, right_id) with new_id in all words.
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
    """Replace all consecutive (left_id, right_id) in word with new_id."""
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
    """BPE tokenizer with encode/decode and optional special tokens."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self._vocab = dict(vocab)
        self._merges = list(merges)
        self._special_tokens = list(special_tokens or [])
        # bytes -> id (for decode). Prefer first id if duplicate.
        self._bytes_to_id: dict[bytes, int] = {}
        for i, b in self._vocab.items():
            if b not in self._bytes_to_id:
                self._bytes_to_id[b] = i
        # Add special tokens to vocab if not present.
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
        if not text:
            return []
        if not self._special_tokens:
            return self._encode_segment(text)
        # Split keeping delimiters: ["a", "<|endoftext|>", "b"]
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
        if not text:
            return []
        result: list[int] = []
        for m in re_module.finditer(PAT, text):
            token = m.group(0)
            result.extend(self._bpe_encode_bytes(token.encode("utf-8")))
        return result

    def _bpe_encode_bytes(self, b: bytes) -> list[int]:
        """Encode a single pre-token (bytes) to list of ids using merges."""
        if not b:
            return []
        # Start with single-byte tokens (ids).
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
        """Lazily yield token ids from an iterable of strings (e.g. file line by line)."""
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            # Split on special tokens and emit full segments; keep incomplete at boundary.
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
        b = b"".join(self._vocab.get(i, b"") for i in ids)
        return b.decode("utf-8", errors="replace")


def _load_vocab_from_file(path: str | os.PathLike) -> dict[int, bytes]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Our format: {"0": [97, 98], "1": [99]} for id -> list of byte values.
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
    # GPT-2 style: keys are token strings (single char or <|endoftext|>), values are ids.
    for k, v in data.items():
        idx = int(v)
        if k == "<|endoftext|>" or len(k) > 1:
            vocab[idx] = k.encode("utf-8")
        else:
            # Single char in GPT-2 byte repr; need byte decoder from tiktoken/gpt2.
            vocab[idx] = k.encode("utf-8")
    return vocab


def _load_merges_from_file(
    path: str | os.PathLike,
    vocab: dict[int, bytes] | None = None,
) -> list[tuple[bytes, bytes]]:
    merges = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line or " " not in line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                a, b = parts
                # Our format: "32,116 104,101" (comma-separated byte ints)
                if "," in a or "," in b:
                    left = bytes(int(x) for x in a.split(","))
                    right = bytes(int(x) for x in b.split(","))
                    merges.append((left, right))
                else:
                    # GPT-2 style: tokens as in vocab (single chars). Decode via UTF-8.
                    merges.append((a.encode("utf-8"), b.encode("utf-8")))
    return merges
