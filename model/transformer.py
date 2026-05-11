"""The full decoder-only Transformer language model."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.kv_cache import KVCache
from model.rmsnorm import RMSNorm
from model.rope import build_rope_cache
from model.transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    n_kv_heads=config.n_kv_heads,
                    dropout=config.dropout,
                    norm_eps=config.norm_eps,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Pre-build RoPE caches up to max_seq_len; will be moved to device on first call.
        head_dim = config.d_model // config.n_heads
        cos, sin = build_rope_cache(config.max_seq_len, head_dim, base=config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Gradient checkpointing toggle
        self._grad_ckpt = False

        self.apply(self._init_weights)

    # ---- init ----
    def _init_weights(self, m: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)

    # ---- features ----
    def enable_gradient_checkpointing(self, flag: bool = True) -> None:
        self._grad_ckpt = flag

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.tok_emb.weight.numel()
            if not self.config.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    # ---- forward ----
    def forward(
        self,
        input_ids: torch.Tensor,            # (B, T)
        cache: KVCache | None = None,
    ) -> torch.Tensor:                      # logits (B, T, V)
        B, T = input_ids.shape
        if cache is None and T > self.config.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}")

        x = self.tok_emb(input_ids)
        cos = self.rope_cos
        sin = self.rope_sin

        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            if self._grad_ckpt and self.training and layer_cache is None:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, layer_cache, use_reentrant=False
                )
            else:
                x = block(x, cos, sin, cache=layer_cache)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    # ---- persistence ----
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.to_dict(),
                "model_state": self.state_dict(),
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "Transformer":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ModelConfig.from_dict(ckpt["config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state"])
        return model
