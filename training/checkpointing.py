"""Save / load training checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from model.config import ModelConfig
from model.transformer import Transformer


def save_checkpoint(
    path: str | Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[Transformer, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt
