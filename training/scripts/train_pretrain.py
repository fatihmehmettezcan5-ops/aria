"""Pre-training entry point. Usage:

    python -m training.scripts.train_pretrain --config configs/train_smoke.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from data.dataset import PackedTokenDataset
from data.tokenization.pack import pack_corpus
from model.config import MODEL_PRESETS, ModelConfig
from model.transformer import Transformer
from tokenizer.tokenizer import Tokenizer
from training.trainer import TrainConfig, Trainer


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = _load(args.config)

    tokenizer_path = cfg["tokenizer"]
    tok = Tokenizer.from_file(tokenizer_path)

    # 1) Pack the corpus into a .bin if missing
    bin_path = Path(cfg["data"]["bin"])
    meta_path = bin_path.with_suffix(".meta.json")
    if not bin_path.exists() or not meta_path.exists():
        pack_corpus(cfg["data"]["corpus"], tokenizer_path, str(bin_path))
    train_ds = PackedTokenDataset(bin_path, seq_len=int(cfg["train"]["seq_len"]))
    val_ds = train_ds if cfg.get("eval_on_train", True) else None

    # 2) Build model from preset, override vocab to match tokenizer
    preset = MODEL_PRESETS[cfg["model"]["preset"]]
    overrides = cfg["model"].get("override", {})
    base = {**preset.to_dict(), **overrides, "vocab_size": tok.vocab_size}
    mcfg = ModelConfig.from_dict(base)
    model = Transformer(mcfg)
    print(f"[train] model params: {model.num_parameters():,}")

    # 3) Train
    tcfg = TrainConfig(
        out_dir=cfg["train"]["out_dir"],
        total_steps=int(cfg["train"]["total_steps"]),
        batch_size=int(cfg["train"]["batch_size"]),
        grad_accum=int(cfg["train"].get("grad_accum", 1)),
        seq_len=int(cfg["train"]["seq_len"]),
        lr=float(cfg["train"]["lr"]),
        min_lr=cfg["train"].get("min_lr"),
        warmup_steps=int(cfg["train"].get("warmup_steps", 100)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.1)),
        grad_clip=float(cfg["train"].get("grad_clip", 1.0)),
        log_every=int(cfg["train"].get("log_every", 10)),
        eval_every=int(cfg["train"].get("eval_every", 500)),
        save_every=int(cfg["train"].get("save_every", 1000)),
        amp_dtype=cfg["train"].get("amp_dtype", "bf16"),
        grad_checkpointing=bool(cfg["train"].get("grad_checkpointing", False)),
        resume=cfg["train"].get("resume"),
        seed=int(cfg["train"].get("seed", 42)),
        device=cfg["train"].get("device", "auto"),
        extra={"tokenizer": tokenizer_path},
    )
    Trainer(model, train_ds, val_ds, tcfg).fit()


if __name__ == "__main__":
    main()
