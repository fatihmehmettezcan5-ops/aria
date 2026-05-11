"""Rotary Position Embedding (RoPE).

Given a head-dim vector x split into pairs (x_2i, x_2i+1), RoPE rotates each
pair by an angle θ_i * pos. We use the (real-domain) interleaved formulation:

    x'_2i   = x_2i cos(θ) − x_2i+1 sin(θ)
    x'_2i+1 = x_2i sin(θ) + x_2i+1 cos(θ)

with θ_i = base ** (-2i / d). All from scratch, no flash-attention helpers.
"""
from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) caches of shape (seq_len, head_dim)."""
    assert head_dim % 2 == 0, "RoPE needs even head_dim"
    half = head_dim // 2
    # θ_i = base^(-2i/d) for i = 0..half-1
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    # angles per (position, freq)
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)             # (seq_len, half)
    # Duplicate so we have one entry per dim (interleaved pairs).
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(dtype)  # (seq_len, head_dim)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(dtype)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate by 90°: pair (x_a, x_b) -> (-x_b, x_a) on the first/second half."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,            # (B, H, T, D)
    k: torch.Tensor,            # (B, H_kv, T, D)
    cos: torch.Tensor,          # (T_offset+T, D) or (T, D)
    sin: torch.Tensor,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to q and k. Supports KV-cache via `offset`."""
    T = q.shape[-2]
    cos = cos[offset : offset + T]      # (T, D)
    sin = sin[offset : offset + T]
    # broadcast over batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)
