"""Single decoder block: pre-norm → attention → pre-norm → MLP, with residuals."""
from __future__ import annotations

import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feedforward import SwiGLU
from model.kv_cache import LayerKVCache
from model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = MultiHeadAttention(d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache=cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x
