import torch

from inference.generator import GenerationConfig, Generator
from inference.sampler import Sampler, SamplerConfig
from model.config import MODEL_PRESETS, ModelConfig
from model.transformer import Transformer
from tokenizer.bpe_trainer import train_bpe
from tokenizer.tokenizer import Tokenizer


def _tiny_model_and_tok(tmp_path):
    (tmp_path / "c.txt").write_text("hello world. " * 200, encoding="utf-8")
    vocab, merges = train_bpe([str(tmp_path)], vocab_size=300, min_pair_freq=1, verbose=False)
    tok = Tokenizer(vocab, merges)
    cfg = ModelConfig(**{**MODEL_PRESETS["smoke"].to_dict(), "vocab_size": tok.vocab_size})
    return Transformer(cfg).eval(), tok


def test_greedy_runs(tmp_path):
    m, tok = _tiny_model_and_tok(tmp_path)
    g = Generator(m, tok, device="cpu")
    out = g.generate_text("hello", GenerationConfig(max_new_tokens=10, temperature=0.0))
    assert isinstance(out, str)


def test_streaming_yields_ints(tmp_path):
    m, tok = _tiny_model_and_tok(tmp_path)
    g = Generator(m, tok, device="cpu")
    ids = tok.encode("hello", add_bos=True)
    n = 0
    for tid in g.stream(ids, GenerationConfig(max_new_tokens=5, temperature=0.0)):
        assert isinstance(tid, int)
        n += 1
        if n >= 5:
            break
    assert n >= 1


def test_sampler_top_p_doesnt_crash():
    s = Sampler(SamplerConfig(temperature=0.7, top_k=5, top_p=0.9))
    logits = torch.randn(2, 50)
    out = s.sample(logits)
    assert out.shape == (2,)
