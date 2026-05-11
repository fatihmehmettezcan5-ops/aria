"""Conversation summarisation using *our* model when context grows too long."""
from __future__ import annotations

from inference.generator import GenerationConfig, Generator


SUMMARY_PROMPT = (
    "Summarise the following conversation in 3-5 bullet points, preserving "
    "facts the assistant should remember. Be concise.\n\n"
    "Conversation:\n{conv}\n\nSummary:"
)


def summarise_conversation(generator: Generator, transcript: str, max_new_tokens: int = 256) -> str:
    prompt = SUMMARY_PROMPT.format(conv=transcript[:6000])
    return generator.generate_text(
        prompt,
        GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.3, top_p=0.9),
    ).strip()
