"""Supervised fine-tuning entry point. Loads a pretrain checkpoint, then SFTs
on a JSONL of {input_ids, labels} examples (created by data.scripts.prepare_finetune).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from data.dataset import JsonlSFTDataset
from data.scripts.prepare_finetune import main as build_sft_jsonl  # noqa: F401 (re-export)
from training.checkpointing import load_checkpoint
from training.trainer import TrainConfig, Trainer
from tokenizer.tokenizer import Tokenizer


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="override checkpoint path")
    args = ap.parse_args()
    cfg = _load(args.config)

    ckpt_path = args.ckpt or cfg["pretrain_ckpt"]
    tokenizer_path = cfg["tokenizer"]
    tok = Tokenizer.from_file(tokenizer_path)

    # Auto-build SFT JSONL if missing
    sft_jsonl = Path(cfg["data"]["jsonl"])
    if not sft_jsonl.exists():
        sft_jsonl.parent.mkdir(parents=True, exist_ok=True)
        # Programmatic call equivalent to `python -m data.scripts.prepare_finetune ...`
        import sys
        sys.argv = [
            "prepare_finetune",
            "--tokenizer", tokenizer_path,
            "--out", str(sft_jsonl),
            "--seq-len", str(cfg["train"]["seq_len"]),
        ]
        from data.scripts import prepare_finetune
        prepare_finetune.main()

    train_ds = JsonlSFTDataset(sft_jsonl, seq_len=int(cfg["train"]["seq_len"]), pad_id=tok.pad_id)

    print(f"[finetune] loading pretrain ckpt: {ckpt_path}")
    model, _ = load_checkpoint(ckpt_path)
    print(f"[finetune] model params: {model.num_parameters():,}")

    tcfg = TrainConfig(
        out_dir=cfg["train"]["out_dir"],
        total_steps=int(cfg["train"]["total_steps"]),
        batch_size=int(cfg["train"]["batch_size"]),
        grad_accum=int(cfg["train"].get("grad_accum", 1)),
        seq_len=int(cfg["train"]["seq_len"]),
        lr=float(cfg["train"]["lr"]),
        min_lr=cfg["train"].get("min_lr"),
        warmup_steps=int(cfg["train"].get("warmup_steps", 50)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.1)),
        log_every=int(cfg["train"].get("log_every", 10)),
        eval_every=int(cfg["train"].get("eval_every", 200)),
        save_every=int(cfg["train"].get("save_every", 500)),
        amp_dtype=cfg["train"].get("amp_dtype", "bf16"),
        device=cfg["train"].get("device", "auto"),
        extra={"tokenizer": tokenizer_path, "from_pretrain": ckpt_path},
    )
    Trainer(model, train_ds, train_ds, tcfg).fit()


if __name__ == "__main__":
    main()
