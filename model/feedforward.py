"""SwiGLU feed-forward network."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """FFN(x) = (Swish(W1 x) ⊙ (W3 x)) W2"""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # gate
        self.w3 = nn.Linear(d_model, d_ff, bias=False)   # up
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
