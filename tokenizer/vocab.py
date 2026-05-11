"""Bidirectional vocabulary."""
from __future__ import annotations


class Vocab:
    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []

    def add(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.id_to_token)
        self.token_to_id[token] = idx
        self.id_to_token.append(token)
        return idx

    def __len__(self) -> int:
        return len(self.id_to_token)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id
