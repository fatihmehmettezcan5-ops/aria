"""Singleton wrapper around our model + tokenizer + generator.

Loaded lazily on first use; logs whether a real checkpoint was found or
whether we fall back to a tiny untrained model (so the app starts even
without `make smoke-train`).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import torch

from backend.config import get_settings
from inference.generator import GenerationConfig, Generator
from model.config import MODEL_PRESETS, ModelConfig
from model.transformer import Transformer
from tokenizer.bpe_trainer import train_bpe
from tokenizer.tokenizer import Tokenizer
from training.checkpointing import load_checkpoint


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


class ModelService:
    _instance: "ModelService | None" = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "ModelService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = ModelService()
            return cls._instance

    def __init__(self) -> None:
        s = get_settings()
        self.device = _resolve_device(s.aria_device)
        self.checkpoint_path = s.aria_checkpoint
        self.tokenizer_path = s.aria_tokenizer
        self.fallback = False
        self._load()

    def _load(self) -> None:
        if Path(self.tokenizer_path).exists() and Path(self.checkpoint_path).exists():
            print(f"[model] loading {self.checkpoint_path} on {self.device}")
            self.model, _ = load_checkpoint(self.checkpoint_path, map_location=self.device)
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        else:
            print(f"[model] WARNING: no checkpoint at {self.checkpoint_path!r} — "
                  f"using untrained smoke model so the API still starts.")
            self.fallback = True
            # Build an untrained tiny tokenizer + model so endpoints still respond.
            tmp_corpus = "/tmp/aria_fallback_corpus.txt"
            if not os.path.exists(tmp_corpus):
                with open(tmp_corpus, "w", encoding="utf-8") as f:
                    f.write("hello world. " * 200)
            vocab, merges = train_bpe([tmp_corpus], vocab_size=300, min_pair_freq=1, verbose=False)
            self.tokenizer = Tokenizer(vocab, merges)
            cfg = ModelConfig(**{**MODEL_PRESETS["smoke"].to_dict(),
                                 "vocab_size": self.tokenizer.vocab_size})
            self.model = Transformer(cfg)
        self.generator = Generator(self.model, self.tokenizer, device=self.device)

    def info(self) -> dict:
        cfg = self.model.config
        return {
            "checkpoint": self.checkpoint_path,
            "tokenizer": self.tokenizer_path,
            "device": str(self.device),
            "fallback_untrained": self.fallback,
            "vocab_size": self.tokenizer.vocab_size,
            "n_params": self.model.num_parameters(),
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "max_seq_len": cfg.max_seq_len,
        }

    def default_gen_cfg(self) -> GenerationConfig:
        s = get_settings()
        return GenerationConfig(
            max_new_tokens=s.aria_max_new_tokens,
            temperature=s.aria_default_temperature,
            top_p=s.aria_default_top_p,
            top_k=s.aria_default_top_k,
            repetition_penalty=1.1,
        )
