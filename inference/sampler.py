"""Token sampling: greedy / temperature / top-k / top-p / repetition penalty."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplerConfig:
    temperature: float = 1.0
    top_k: int | None = 50
    top_p: float | None = 0.95
    repetition_penalty: float = 1.0
    repetition_window: int = 128


class Sampler:
    def __init__(self, cfg: SamplerConfig) -> None:
        self.cfg = cfg

    def sample(self, logits: torch.Tensor, history: torch.Tensor | None = None) -> torch.Tensor:
        """Sample one token from final-timestep logits.

        logits:  (B, V)   — last position's logits
        history: (B, T)   — already-generated IDs (for repetition penalty)
        Returns: (B,)     — sampled token IDs
        """
        cfg = self.cfg
        logits = logits.float().clone()

        # Repetition penalty: divide logits of recently-seen tokens by `penalty`.
        if cfg.repetition_penalty != 1.0 and history is not None and history.numel() > 0:
            window = history[:, -cfg.repetition_window :]
            for i in range(logits.size(0)):
                ids = window[i].unique()
                pos = logits[i, ids] >= 0
                logits[i, ids[pos]] /= cfg.repetition_penalty
                logits[i, ids[~pos]] *= cfg.repetition_penalty

        if cfg.temperature <= 0:
            return logits.argmax(dim=-1)

        logits = logits / cfg.temperature

        # top-k
        if cfg.top_k is not None and cfg.top_k > 0 and cfg.top_k < logits.size(-1):
            top_vals, top_idx = torch.topk(logits, cfg.top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, top_idx, top_vals)
            logits = mask

        # top-p (nucleus)
        if cfg.top_p is not None and 0.0 < cfg.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumprob = sorted_probs.cumsum(dim=-1)
            # mask tokens beyond cumulative prob top_p (keep at least 1)
            mask = cumprob - sorted_probs > cfg.top_p
            sorted_logits[mask] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        # safety: replace nan rows (could happen if all -inf)
        if torch.isnan(probs).any():
            probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
            row_sum = probs.sum(-1, keepdim=True)
            probs = torch.where(row_sum > 0, probs / row_sum, torch.ones_like(probs) / probs.size(-1))
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
