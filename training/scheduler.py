"""Linear-warmup + cosine-decay learning-rate schedule (from scratch)."""
from __future__ import annotations

import math


class WarmupCosineSchedule:
    def __init__(
        self,
        peak_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float | None = None,
    ) -> None:
        self.peak_lr = peak_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr = min_lr if min_lr is not None else peak_lr * 0.1

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        if step >= self.total_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cos

    def apply(self, optimizer, step: int) -> float:
        lr = self.lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr
