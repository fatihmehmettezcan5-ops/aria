"""Multi-head (and grouped-query) self-attention with causal masking + RoPE.

Implemented from scratch with explicit matmul / softmax — no calls to
F.scaled_dot_product_attention or flash attention helpers.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.kv_cache import LayerKVCache
from model.rope import apply_rope


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        n_kv_heads = n_kv_heads or n_heads
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads (GQA)"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.dropout = dropout

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """(B, H_kv, T, D) -> (B, H_kv*n_rep, T, D), without a real copy via expand+reshape."""
        if n_rep == 1:
            return x
        b, h_kv, t, d = x.shape
        return (
            x[:, :, None, :, :]
            .expand(b, h_kv, n_rep, t, d)
            .reshape(b, h_kv * n_rep, t, d)
        )

    def forward(
        self,
        x: torch.Tensor,                # (B, T, d_model)
        cos: torch.Tensor,              # RoPE caches
        sin: torch.Tensor,
        cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, H_kv, D = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.wq(x).view(B, T, H, D).transpose(1, 2)        # (B,H,T,D)
        k = self.wk(x).view(B, T, H_kv, D).transpose(1, 2)     # (B,H_kv,T,D)
        v = self.wv(x).view(B, T, H_kv, D).transpose(1, 2)

        # RoPE: offset by previous cache length so positions are absolute.
        offset = cache.length if cache is not None else 0
        q, k = apply_rope(q, k, cos, sin, offset=offset)

        if cache is not None:
            k, v = cache.append(k, v)
        # Expand KV heads up to query heads (GQA).
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        # Scaled dot-product attention (manual).
        # q: (B,H,T,D)  k: (B,H,Tk,D)  -> attn: (B,H,T,Tk)
        scale = 1.0 / math.sqrt(D)
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale

        Tk = k.shape[-2]
        # Causal mask: token i (in current chunk, absolute pos = offset+i)
        # may attend to keys 0..offset+i.
        if cache is None:
            # Pure training: square causal mask
            mask = torch.full((T, Tk), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)
        else:
            # During incremental decode T is small (often 1) and Tk = offset+T.
            # row i corresponds to absolute pos offset+i; allow keys up to that.
            row_idx = torch.arange(T, device=x.device).unsqueeze(1) + offset  # (T,1)
            col_idx = torch.arange(Tk, device=x.device).unsqueeze(0)          # (1,Tk)
            mask = torch.where(
                col_idx <= row_idx,
                torch.zeros((), device=x.device),
                torch.full((), float("-inf"), device=x.device),
            )
        attn = attn + mask  # broadcast over (B,H)

        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v)               # (B,H,T,D)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.wo(out)
