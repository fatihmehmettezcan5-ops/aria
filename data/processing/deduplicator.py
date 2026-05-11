"""Basic exact + near-duplicate filter using shingled MinHash-lite."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _hash_shingles(text: str, k: int = 5, n: int = 64) -> set[int]:
    toks = _TOKEN.findall(text.lower())
    if len(toks) < k:
        return {hash(text)}
    out = set()
    for i in range(len(toks) - k + 1):
        h = hashlib.md5(" ".join(toks[i : i + k]).encode()).digest()
        out.add(int.from_bytes(h[:4], "little"))
    # subsample to n smallest hashes — cheap MinHash signature
    return set(sorted(out)[:n])


def deduplicate(texts: Iterable[str], threshold: float = 0.8) -> Iterator[str]:
    """Yield texts that are not near-duplicates of any prior one."""
    seen_sigs: list[set[int]] = []
    seen_exact: set[str] = set()
    for t in texts:
        h = hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()
        if h in seen_exact:
            continue
        sig = _hash_shingles(t)
        is_dup = False
        for s in seen_sigs:
            inter = len(sig & s)
            if inter == 0:
                continue
            jacc = inter / len(sig | s)
            if jacc >= threshold:
                is_dup = True
                break
        if is_dup:
            continue
        seen_exact.add(h)
        seen_sigs.append(sig)
        yield t
