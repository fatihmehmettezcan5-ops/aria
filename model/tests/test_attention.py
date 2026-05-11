import torch

from model.attention import MultiHeadAttention
from model.kv_cache import LayerKVCache
from model.rope import build_rope_cache


def test_attention_shape_and_causal():
    torch.manual_seed(0)
    d_model, H, T = 32, 4, 7
    attn = MultiHeadAttention(d_model, n_heads=H, n_kv_heads=2)
    cos, sin = build_rope_cache(T, d_model // H)
    x = torch.randn(2, T, d_model)
    out = attn(x, cos, sin)
    assert out.shape == (2, T, d_model)


def test_kv_cache_matches_full_pass():
    """Step-by-step decode with KV cache should match a full forward pass."""
    torch.manual_seed(0)
    d_model, H, T = 16, 4, 5
    attn = MultiHeadAttention(d_model, n_heads=H, n_kv_heads=2)
    attn.eval()
    cos, sin = build_rope_cache(T + 4, d_model // H)
    x = torch.randn(1, T, d_model)

    with torch.no_grad():
        full = attn(x, cos, sin)

        cache = LayerKVCache()
        outs = []
        for t in range(T):
            outs.append(attn(x[:, t : t + 1], cos, sin, cache=cache))
        step = torch.cat(outs, dim=1)

    assert torch.allclose(full, step, atol=1e-5), (full - step).abs().max()
