"""Stopping criteria for autoregressive generation."""
from __future__ import annotations

from collections.abc import Iterable

import torch


class StopCondition:
    """Combine multiple stopping rules. Active per batch row."""

    def __init__(
        self,
        stop_token_ids: Iterable[int] | None = None,
        max_new_tokens: int = 256,
        stop_strings: Iterable[str] | None = None,
        decode_fn=None,
    ) -> None:
        self.stop_ids = set(stop_token_ids or [])
        self.max_new_tokens = max_new_tokens
        self.stop_strings = list(stop_strings or [])
        self.decode_fn = decode_fn  # callable(list[int]) -> str

    def should_stop(self, generated: torch.Tensor, step: int) -> torch.Tensor:
        """generated: (B, T_gen). Returns bool tensor of shape (B,)."""
        B = generated.size(0)
        flags = torch.zeros(B, dtype=torch.bool, device=generated.device)
        if step >= self.max_new_tokens:
            flags[:] = True
            return flags
        if self.stop_ids:
            last = generated[:, -1].tolist()
            for i, t in enumerate(last):
                if t in self.stop_ids:
                    flags[i] = True
        if self.stop_strings and self.decode_fn is not None:
            for i in range(B):
                txt = self.decode_fn(generated[i].tolist())
                for s in self.stop_strings:
                    if s in txt:
                        flags[i] = True
                        break
        return flags
