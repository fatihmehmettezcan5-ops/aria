import torch

from model.rope import apply_rope, build_rope_cache


def test_rope_shapes():
    cos, sin = build_rope_cache(seq_len=16, head_dim=8)
    assert cos.shape == (16, 8) and sin.shape == (16, 8)


def test_rope_offset_consistency():
    """Applying RoPE in one shot vs. with cache offsets should match."""
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 6, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    cos, sin = build_rope_cache(T, D)

    q_full, k_full = apply_rope(q, k, cos, sin, offset=0)

    # Now do it in two halves, the second with offset.
    q1, k1 = apply_rope(q[:, :, :3], k[:, :, :3], cos, sin, offset=0)
    q2, k2 = apply_rope(q[:, :, 3:], k[:, :, 3:], cos, sin, offset=3)
    q_split = torch.cat([q1, q2], dim=-2)
    k_split = torch.cat([k1, k2], dim=-2)
    assert torch.allclose(q_full, q_split, atol=1e-6)
    assert torch.allclose(k_full, k_split, atol=1e-6)
