"""Model configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int | None = None  # None → MHA; <n_heads → GQA
    d_model: int = 512
    d_ff: int = 1376  # ≈ 2.67 * d_model rounded for SwiGLU
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    init_std: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        # ignore unknown keys for forward-compat
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


# Named presets (vocab_size to be overridden by tokenizer at load time).
MODEL_PRESETS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(
        n_layers=6, n_heads=4, n_kv_heads=4,
        d_model=256, d_ff=688, max_seq_len=512,
    ),
    "small": ModelConfig(
        n_layers=12, n_heads=8, n_kv_heads=4,
        d_model=512, d_ff=1376, max_seq_len=1024,
    ),
    "medium": ModelConfig(
        n_layers=16, n_heads=12, n_kv_heads=4,
        d_model=768, d_ff=2048, max_seq_len=2048,
    ),
    "base": ModelConfig(
        n_layers=24, n_heads=16, n_kv_heads=4,
        d_model=1024, d_ff=2730, max_seq_len=4096,
    ),
    # Used by smoke tests / CI
    "smoke": ModelConfig(
        n_layers=2, n_heads=4, n_kv_heads=2,
        d_model=128, d_ff=256, max_seq_len=256,
    ),
}
