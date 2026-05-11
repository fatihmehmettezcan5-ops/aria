"""Parse <TOOL_CALL>{...}</TOOL_CALL> spans out of generated text."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from tokenizer.special_tokens import SpecialTokens

_S = SpecialTokens
_TOOL_RE = re.compile(
    re.escape(_S.TOOL_CALL) + r"\s*(.*?)\s*" + re.escape(_S.TOOL_CALL_END),
    re.DOTALL,
)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict
    raw: str
    span: tuple[int, int]   # (start, end) char offsets in the source string


def find_tool_calls(text: str) -> list[ParsedToolCall]:
    out: list[ParsedToolCall] = []
    for m in _TOOL_RE.finditer(text):
        body = m.group(1).strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            # Try a forgiving fallback: find the first {...} block.
            i = body.find("{")
            j = body.rfind("}")
            if i != -1 and j > i:
                try:
                    obj = json.loads(body[i : j + 1])
                except json.JSONDecodeError:
                    continue
            else:
                continue
        if not isinstance(obj, dict):
            continue
        out.append(ParsedToolCall(
            name=str(obj.get("name", "")),
            arguments=obj.get("arguments") or obj.get("args") or {},
            raw=body,
            span=m.span(),
        ))
    return out


def first_tool_call(text: str) -> ParsedToolCall | None:
    """Return the *first complete* tool call in `text` (used during streaming)."""
    calls = find_tool_calls(text)
    return calls[0] if calls else None


def has_open_tool_call(text: str) -> bool:
    """True if a <TOOL_CALL> opener exists without a matching closer (still streaming)."""
    n_open = text.count(_S.TOOL_CALL)
    n_close = text.count(_S.TOOL_CALL_END)
    return n_open > n_close
