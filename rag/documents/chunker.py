"""Sentence-aware chunker with sliding overlap.

We avoid tiktoken / a tokenizer dependency here because chunking is purely
a *retrieval* concern; we use approximate character lengths instead.
"""
from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}", re.UNICODE)


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, target_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    """Greedy sentence packing with overlap. Returns non-empty chunk strings."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for sent in sentences:
        s_len = len(sent) + 1
        if buf and buf_len + s_len > target_chars:
            chunks.append(" ".join(buf).strip())
            # build overlap
            tail: list[str] = []
            tail_len = 0
            for s in reversed(buf):
                if tail_len + len(s) + 1 > overlap_chars:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            buf = tail[:]
            buf_len = tail_len
        buf.append(sent)
        buf_len += s_len

    if buf:
        chunks.append(" ".join(buf).strip())
    # Hard-cut overly long single sentences as a safety net
    out: list[str] = []
    for c in chunks:
        if len(c) <= target_chars * 1.5:
            out.append(c)
        else:
            for i in range(0, len(c), target_chars):
                out.append(c[i : i + target_chars])
    return [c for c in out if c]
