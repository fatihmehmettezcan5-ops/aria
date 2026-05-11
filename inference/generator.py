"""Streaming, KV-cached, autoregressive generation."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from inference.sampler import Sampler, SamplerConfig
from inference.stopping import StopCondition
from model.kv_cache import KVCache
from model.transformer import Transformer
from tokenizer.tokenizer import Tokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    repetition_penalty: float = 1.1
    stop_strings: list[str] | None = None
    extra_stop_token_ids: list[int] | None = None


class Generator:
    """Wraps a Transformer + Tokenizer for chat / text generation.

    Single-stream (batch=1) by default, but `generate_batch` is also provided
    for offline batch jobs.
    """

    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    @torch.no_grad()
    def stream(
        self,
        prompt_ids: list[int],
        cfg: GenerationConfig,
    ) -> Iterator[int]:
        """Yield generated token IDs one at a time."""
        sampler = Sampler(SamplerConfig(
            temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
        ))
        stop_ids = [self.tokenizer.eos_id, self.tokenizer.end_id]
        if cfg.extra_stop_token_ids:
            stop_ids.extend(cfg.extra_stop_token_ids)
        stopper = StopCondition(
            stop_token_ids=stop_ids,
            max_new_tokens=cfg.max_new_tokens,
            stop_strings=cfg.stop_strings,
            decode_fn=lambda ids: self.tokenizer.decode(ids, skip_special=False),
        )

        # Truncate prompt to model context (leave room for new tokens).
        max_ctx = self.model.config.max_seq_len - 1
        room = max_ctx - cfg.max_new_tokens
        if len(prompt_ids) > room and room > 0:
            prompt_ids = prompt_ids[-room:]

        cache = KVCache(self.model.config.n_layers)
        ids_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # 1) Prefill — feed the whole prompt in one shot.
        logits = self.model(ids_tensor, cache=cache)
        last_logits = logits[:, -1, :]

        generated = torch.empty((1, 0), dtype=torch.long, device=self.device)
        for step in range(cfg.max_new_tokens):
            next_id = sampler.sample(last_logits, history=torch.cat([ids_tensor, generated], dim=1))
            generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
            yield int(next_id.item())

            if stopper.should_stop(generated, step + 1).any():
                return

            # 2) Decode step — feed only the new token; cache holds the rest.
            logits = self.model(next_id.unsqueeze(1), cache=cache)
            last_logits = logits[:, -1, :]

    def generate(self, prompt_ids: list[int], cfg: GenerationConfig) -> list[int]:
        return list(self.stream(prompt_ids, cfg))

    def generate_text(self, prompt: str, cfg: GenerationConfig | None = None) -> str:
        cfg = cfg or GenerationConfig()
        ids = self.tokenizer.encode(prompt, add_bos=True)
        out_ids = self.generate(ids, cfg)
        return self.tokenizer.decode(out_ids, skip_special=True)
