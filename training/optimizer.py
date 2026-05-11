"""Build an AdamW optimizer with separate weight-decay groups.

Parameters that should NOT be decayed: 1-D tensors (norms, biases) and
embeddings. Everything else (linear weights, attention projections) gets
the full weight_decay value.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "tok_emb" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available() and torch.__version__ >= "2.0"
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, fused=fused)
