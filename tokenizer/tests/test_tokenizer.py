"""Round-trip + special-token tests for the from-scratch BPE tokenizer."""
from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer.bpe_trainer import train_bpe
from tokenizer.special_tokens import SpecialTokens
from tokenizer.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tiny_tok(tmp_path_factory):
    corpus_dir = tmp_path_factory.mktemp("corpus")
    text = (
        "Hello, world!\n"
        "Merhaba dünya. Yapay zekâ asistanı.\n"
        "The quick brown fox jumps over the lazy dog. " * 50
        + "İstanbul'da hava güneşli. " * 30
        + "def add(a, b):\n    return a + b\n" * 20
    )
    (corpus_dir / "a.txt").write_text(text, encoding="utf-8")
    vocab, merges = train_bpe([str(corpus_dir)], vocab_size=600, min_pair_freq=1, verbose=False)
    return Tokenizer(vocab, merges)


def test_roundtrip_ascii(tiny_tok):
    s = "Hello, world!"
    ids = tiny_tok.encode(s)
    assert tiny_tok.decode(ids) == s


def test_roundtrip_unicode_turkish(tiny_tok):
    s = "Merhaba dünya — İstanbul'da yapay zekâ."
    ids = tiny_tok.encode(s)
    assert tiny_tok.decode(ids) == s


def test_roundtrip_emoji(tiny_tok):
    s = "🤖✨ hello 💡"
    ids = tiny_tok.encode(s)
    assert tiny_tok.decode(ids) == s


def test_special_tokens_single_id(tiny_tok):
    S = SpecialTokens
    text = f"{S.USER}hi{S.END}"
    ids = tiny_tok.encode(text)
    assert tiny_tok.token_to_id[S.USER] in ids
    assert tiny_tok.token_to_id[S.END] in ids
    # decode preserves them (default keeps specials)
    assert tiny_tok.decode(ids) == text


def test_skip_special_tokens(tiny_tok):
    S = SpecialTokens
    ids = tiny_tok.encode(f"{S.USER}hi{S.END}")
    assert tiny_tok.decode(ids, skip_special=True) == "hi"


def test_save_and_load(tiny_tok, tmp_path):
    p = tmp_path / "tok.json"
    tiny_tok.save(p)
    other = Tokenizer.from_file(p)
    s = "round-trip after reload — Türkçe & emoji 🎉"
    assert other.decode(other.encode(s)) == s
