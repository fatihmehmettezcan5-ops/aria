"""Optional neural sentence embedder using *our* trained Transformer.

Mean-pools the last-layer hidden states (skipping pad tokens) and L2-normalises.
This is a viable alternative to TF-IDF once you have a fine-tuned checkpoint;
see `rag.embeddings.trainer` for a contrastive-learning recipe.
"""
from __future__ import annotations

import numpy as np
import torch

from model.transformer import Transformer
from tokenizer.tokenizer import Tokenizer


class NeuralEncoder:
    def __init__(self, model: Transformer, tokenizer: Tokenizer, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.device = torch.device(device)
        self.dim = model.config.d_model

    @torch.no_grad()
    def encode_one(self, text: str, max_len: int = 256) -> np.ndarray:
        ids = self.tok.encode(text, add_bos=True)[:max_len]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        # We use the embeddings only — call up to final_norm.
        h = self.model.tok_emb(x)
        cos, sin = self.model.rope_cos, self.model.rope_sin
        for block in self.model.blocks:
            h = block(h, cos, sin)
        h = self.model.final_norm(h)              # (1, T, D)
        vec = h.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode_one(t) for t in texts]
