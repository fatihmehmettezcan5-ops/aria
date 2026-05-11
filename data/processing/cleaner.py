"""Generic text cleaning helpers."""
from __future__ import annotations

import re
import unicodedata

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_NL = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _CTRL.sub("", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def looks_low_quality(text: str, min_chars: int = 200) -> bool:
    """Cheap heuristics: too short, mostly non-letters, very repetitive."""
    if len(text) < min_chars:
        return True
    letters = sum(1 for c in text if c.isalpha())
    if letters / max(1, len(text)) < 0.5:
        return True
    # repetition: top-3 lines should not exceed 30% of total lines
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    from collections import Counter
    top = Counter(lines).most_common(3)
    if sum(c for _, c in top) / len(lines) > 0.3:
        return True
    return False
