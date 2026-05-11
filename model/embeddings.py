"""Re-export of the token embedding (kept as separate file per the spec).

The actual nn.Embedding is constructed inside Transformer; this module
exposes a thin wrapper so external code can use TokenEmbedding directly
in unit tests / experiments.
"""
from __future__ import annotations

import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    """Plain token embedding. Kept as a separate name for clarity."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__(vocab_size, d_model)
