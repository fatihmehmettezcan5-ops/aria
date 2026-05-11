"""The training loop: from-scratch, supports AMP, gradient accumulation,
gradient checkpointing, gradient clipping, TensorBoard, periodic eval."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from training.checkpointing import save_checkpoint
from training.optimizer import build_optimizer
from training.scheduler import WarmupCosineSchedule


@dataclass
class TrainConfig:
    out_dir: str
    total_steps: int
    batch_size: int = 32
    grad_accum: int = 1
    seq_len: int = 512
    lr: float = 3e-4
    min_lr: float | None = None
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    amp_dtype: str = "bf16"   # bf16 | fp16 | fp32
    grad_checkpointing: bool = False
    resume: str | None = None
    seed: int = 42
    device: str = "auto"
    extra: dict = field(default_factory=dict)


def _device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cuda" or (cfg.device == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _amp_dtype(cfg: TrainConfig, dev: torch.device):
    if dev.type != "cuda":
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.amp_dtype]


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with -100 ignored. Pre-shifted (labels already aligned)."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 50) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(ids)
            loss = causal_lm_loss(logits, labels)
            n = (labels != -100).sum().item()
            total_loss += loss.item() * n
            total_tokens += n
    model.train()
    if total_tokens == 0:
        return {"val_loss": float("nan"), "val_ppl": float("nan")}
    avg = total_loss / total_tokens
    return {"val_loss": avg, "val_ppl": math.exp(min(avg, 20.0))}


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset | None,
        cfg: TrainConfig,
    ) -> None:
        self.cfg = cfg
        self.device = _device(cfg)
        torch.manual_seed(cfg.seed)

        self.model = model.to(self.device)
        if cfg.grad_checkpointing and hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing(True)

        self.optimizer = build_optimizer(self.model, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = WarmupCosineSchedule(
            peak_lr=cfg.lr, warmup_steps=cfg.warmup_steps,
            total_steps=cfg.total_steps, min_lr=cfg.min_lr,
        )

        self.amp = _amp_dtype(cfg, self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp == torch.float16)

        # Use pin_memory only for CUDA + non-zero workers
        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=0, pin_memory=pin, drop_last=True,
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=0, pin_memory=pin)
            if val_dataset is not None else None
        )

        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(self.out_dir / "tb")

        self.step = 0
        self.best_val = float("inf")

        if cfg.resume and Path(cfg.resume).exists():
            self._resume(cfg.resume)

    # -------- helpers --------
    def _resume(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.step = int(ckpt.get("step", 0))
        print(f"[trainer] resumed from {path} at step {self.step}")

    def _train_step(self, batch_iter) -> tuple[float, int]:
        self.optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        token_total = 0
        for accum in range(self.cfg.grad_accum):
            try:
                batch = next(batch_iter)
            except StopIteration:
                return loss_total, token_total
            ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=self.amp, enabled=self.amp != torch.float32):
                logits = self.model(ids)
                loss = causal_lm_loss(logits, labels) / self.cfg.grad_accum
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_total += loss.item() * self.cfg.grad_accum
            token_total += (labels != -100).sum().item()

        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        lr = self.scheduler.apply(self.optimizer, self.step)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # log
        self.tb.add_scalar("train/loss", loss_total / max(1, self.cfg.grad_accum), self.step)
        self.tb.add_scalar("train/lr", lr, self.step)
        self.tb.add_scalar("train/grad_norm", float(gnorm), self.step)
        return loss_total / max(1, self.cfg.grad_accum), token_total

    def fit(self) -> None:
        self.model.train()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[trainer] starting on {self.device}, params={n_params:,}, "
              f"steps={self.cfg.total_steps}, batch={self.cfg.batch_size}x{self.cfg.grad_accum}")
        t0 = time.time()
        epoch_iter = iter(self.train_loader)
        loss_ema: float | None = None
        last_log = time.time()
        last_tok = 0
        toks = 0

        while self.step < self.cfg.total_steps:
            try:
                # peek by re-priming each step's accumulation inside _train_step
                pass
            except Exception:
                pass

            # Prepare an iterator that auto-restarts on epoch end.
            def _gen():
                nonlocal epoch_iter
                while True:
                    try:
                        yield next(epoch_iter)
                    except StopIteration:
                        epoch_iter = iter(self.train_loader)
                        yield next(epoch_iter)

            it = _gen()
            loss_val, tok = self._train_step(it)
            toks += tok
            loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val

            self.step += 1
            if self.step % self.cfg.log_every == 0:
                now = time.time()
                tps = (toks - last_tok) / max(1e-6, now - last_log)
                last_log, last_tok = now, toks
                print(
                    f"step {self.step:>7} | loss {loss_val:.4f} (ema {loss_ema:.4f}) "
                    f"| lr {self.scheduler.lr_at(self.step):.2e} | tok/s {tps:>8.0f}"
                )

            if self.val_loader is not None and self.step % self.cfg.eval_every == 0:
                metrics = evaluate(self.model, self.val_loader, self.device)
                print(f"  [eval] val_loss={metrics['val_loss']:.4f}  val_ppl={metrics['val_ppl']:.2f}")
                self.tb.add_scalar("val/loss", metrics["val_loss"], self.step)
                self.tb.add_scalar("val/ppl", metrics["val_ppl"], self.step)
                if metrics["val_loss"] < self.best_val:
                    self.best_val = metrics["val_loss"]
                    save_checkpoint(self.out_dir / "best.pt", self.model, self.optimizer, self.step,
                                    extra={"val_loss": metrics["val_loss"], **self.cfg.extra})

            if self.step % self.cfg.save_every == 0:
                save_checkpoint(self.out_dir / f"step_{self.step}.pt",
                                self.model, self.optimizer, self.step, extra=self.cfg.extra)

        # final
        save_checkpoint(self.out_dir / "final.pt", self.model, self.optimizer, self.step, extra=self.cfg.extra)
        (self.out_dir / "train_done.json").write_text(json.dumps({
            "step": self.step, "best_val": self.best_val, "elapsed": time.time() - t0
        }))
        print(f"[trainer] done in {time.time()-t0:.1f}s. final → {self.out_dir/'final.pt'}")
