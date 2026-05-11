import torch

from model.config import MODEL_PRESETS, ModelConfig
from model.kv_cache import KVCache
from model.transformer import Transformer


def _smoke_config(vocab=128) -> ModelConfig:
    cfg = MODEL_PRESETS["smoke"]
    return ModelConfig(**{**cfg.to_dict(), "vocab_size": vocab})


def test_forward_shape():
    cfg = _smoke_config()
    m = Transformer(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits = m(ids)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_param_count_reasonable():
    cfg = _smoke_config(vocab=256)
    m = Transformer(cfg)
    n = m.num_parameters()
    assert n > 1000  # sanity


def test_save_load_roundtrip(tmp_path):
    cfg = _smoke_config()
    m = Transformer(cfg)
    p = tmp_path / "ckpt.pt"
    m.save(p)
    m2 = Transformer.load(p)
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_cached_decode_matches_full():
    cfg = _smoke_config()
    m = Transformer(cfg).eval()
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        full = m(ids)
        cache = KVCache(cfg.n_layers)
        outs = []
        for t in range(ids.shape[1]):
            outs.append(m(ids[:, t : t + 1], cache=cache))
        step = torch.cat(outs, dim=1)
    assert torch.allclose(full, step, atol=1e-4), (full - step).abs().max()
