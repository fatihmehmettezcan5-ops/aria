"""Optional: contrastive sentence-embedder fine-tune over our model.

This is a recipe (in code) — call `train_contrastive(model, tok, pairs)` with
a list of (anchor, positive) string pairs. Loss is in-batch InfoNCE.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rag.embeddings.model import NeuralEncoder


class PairsDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[str, str]:
        return self.pairs[i]


def _embed_batch(enc: NeuralEncoder, texts: list[str], max_len: int = 128) -> torch.Tensor:
    ids_list = [enc.tok.encode(t, add_bos=True)[:max_len] for t in texts]
    L = max(len(x) for x in ids_list)
    pad = enc.tok.pad_id
    ids = torch.tensor(
        [x + [pad] * (L - len(x)) for x in ids_list],
        dtype=torch.long, device=enc.device,
    )
    h = enc.model.tok_emb(ids)
    cos, sin = enc.model.rope_cos, enc.model.rope_sin
    for block in enc.model.blocks:
        h = block(h, cos, sin)
    h = enc.model.final_norm(h)
    mask = (ids != pad).unsqueeze(-1).float()
    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
    return F.normalize(pooled, dim=-1)


def train_contrastive(
    encoder: NeuralEncoder,
    pairs: list[tuple[str, str]],
    *,
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 5e-5,
    temperature: float = 0.05,
) -> None:
    encoder.model.train()
    opt = torch.optim.AdamW(encoder.model.parameters(), lr=lr)
    loader = DataLoader(PairsDataset(pairs), batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        total = 0.0
        n = 0
        for batch in loader:
            anchors, positives = list(batch[0]), list(batch[1])
            za = _embed_batch(encoder, anchors)
            zp = _embed_batch(encoder, positives)
            logits = (za @ zp.T) / temperature
            target = torch.arange(za.size(0), device=za.device)
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) * 0.5
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        print(f"[contrastive] epoch {ep+1}: loss={total/max(1,n):.4f}")
    encoder.model.eval()
