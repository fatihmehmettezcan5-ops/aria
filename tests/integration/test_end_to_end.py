"""End-to-end smoke: train a tiny model, generate text, run the agent."""
import asyncio
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_full_pipeline(tmp_path):
    # 1. Train a tiny tokenizer
    from tokenizer.bpe_trainer import train_bpe
    from tokenizer.tokenizer import Tokenizer

    corpus = tmp_path / "c.txt"
    corpus.write_text(
        "Hello world. " * 100
        + "Question: What is 2 + 2?\nAnswer: 4.\n" * 30
        + "Merhaba dünya. " * 50,
        encoding="utf-8",
    )
    vocab, merges = train_bpe([str(corpus)], vocab_size=400, min_pair_freq=1, verbose=False)
    tok = Tokenizer(vocab, merges)
    assert tok.vocab_size > 250

    # 2. Build a tiny model
    from model.config import MODEL_PRESETS, ModelConfig
    from model.transformer import Transformer
    cfg = ModelConfig(**{**MODEL_PRESETS["smoke"].to_dict(), "vocab_size": tok.vocab_size})
    model = Transformer(cfg).eval()

    # 3. Generate something
    from inference.generator import GenerationConfig, Generator
    gen = Generator(model, tok, device="cpu")
    out = gen.generate_text("hello", GenerationConfig(max_new_tokens=10, temperature=0.0))
    assert isinstance(out, str)

    # 4. Agent loop runs and emits done event
    from agent.orchestrator import AgentOrchestrator
    from tools.registry import get_default_registry
    from tools.schema import ToolContext

    orch = AgentOrchestrator(gen, get_default_registry(),
                             gen_cfg=GenerationConfig(max_new_tokens=8, temperature=0.0))
    events = []
    async for ev in orch.run(language="en", history=[], user_message="hi", ctx=ToolContext()):
        events.append(ev.type)
    assert "done" in events
