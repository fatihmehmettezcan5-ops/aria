"""Per-layer KV cache for autoregressive inference."""
from __future__ import annotations

import torch


class LayerKVCache:
    """Stores K, V tensors of shape (B, H_kv, T, D_head). Grows with each step."""

    __slots__ = ("k", "v")

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=-2)
            self.v = torch.cat([self.v, v_new], dim=-2)
        return self.k, self.v

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[-2]


class KVCache:
    """Container of LayerKVCache, one per transformer block."""

    def __init__(self, n_layers: int) -> None:
        self.layers: list[LayerKVCache] = [LayerKVCache() for _ in range(n_layers)]

    def __getitem__(self, i: int) -> LayerKVCache:
        return self.layers[i]

    @property
    def length(self) -> int:
        return self.layers[0].length if self.layers else 0

    def reset(self) -> None:
        for layer in self.layers:
            layer.k = None
            layer.v = None
