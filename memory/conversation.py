"""Conversation memory: in-memory turn buffer + token-budget windowing.

The DB is the source of truth for persisted history (see backend/database).
This class manages the *active* in-context window passed to the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Turn:
    role: Role
    content: str
    metadata: dict = field(default_factory=dict)


class Conversation:
    def __init__(self, system_prompt: str | None = None) -> None:
        self.turns: list[Turn] = []
        if system_prompt:
            self.turns.append(Turn("system", system_prompt))

    def add(self, role: Role, content: str, **meta) -> None:
        self.turns.append(Turn(role, content, metadata=meta))

    def as_pairs(self) -> list[tuple[str, str]]:
        return [(t.role, t.content) for t in self.turns]

    def trim_to_budget(self, count_tokens, max_tokens: int) -> None:
        """Drop oldest non-system turns until total tokens <= max_tokens."""
        def total() -> int:
            return sum(count_tokens(t.content) for t in self.turns)
        while total() > max_tokens and len(self.turns) > 2:
            # Keep the system turn (index 0) and most recent user/assistant.
            for i in range(1, len(self.turns) - 2):
                if self.turns[i].role != "system":
                    del self.turns[i]
                    break
            else:
                break
