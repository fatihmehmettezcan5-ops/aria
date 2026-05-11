import pytest

from agent.prompt_builder import build_prompt_ids, system_prompt
from tokenizer.bpe_trainer import train_bpe
from tokenizer.tokenizer import Tokenizer
from tools.registry import get_default_registry


def test_system_prompt_includes_tools():
    reg = get_default_registry()
    sp = system_prompt("en", reg.schemas())
    assert "calculator" in sp and "web_search" in sp
    sp_tr = system_prompt("tr", reg.schemas())
    assert "asistan" in sp_tr.lower()


def test_build_prompt_ids_smoke(tmp_path):
    (tmp_path / "c.txt").write_text("hello world. " * 200, encoding="utf-8")
    vocab, merges = train_bpe([str(tmp_path)], vocab_size=400, min_pair_freq=1, verbose=False)
    tok = Tokenizer(vocab, merges)
    reg = get_default_registry()
    ids = build_prompt_ids(
        tok, language="en", tool_specs=reg.schemas(),
        history=[("user", "hi"), ("assistant", "hello!")],
        user_message="what is 2+2?",
    )
    assert isinstance(ids, list) and len(ids) > 5
    # Must end with assistant role marker
    from tokenizer.special_tokens import SpecialTokens
    assert ids[-1] == tok.token_to_id[SpecialTokens.ASSISTANT]
