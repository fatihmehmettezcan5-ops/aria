"""Text preprocessing + byte ↔ unicode mapping (GPT-2 style)."""
from __future__ import annotations

from functools import lru_cache


# Unicode-aware GPT-2 style pretokeniser pattern. We use the `regex` module
# (not stdlib `re`) because it supports `\p{L}` etc.
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Reversible mapping from bytes 0..255 to printable unicode.

    Mirrors the trick used in GPT-2: every byte gets a printable unicode
    character so we can do BPE on strings and round-trip exactly.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    return {v: k for k, v in bytes_to_unicode().items()}


def get_pairs(symbols: tuple[str, ...]) -> set[tuple[str, str]]:
    """Adjacent symbol pairs in a token's symbol sequence."""
    return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}
