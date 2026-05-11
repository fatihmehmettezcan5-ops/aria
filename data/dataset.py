"""Memory-mapped packed-token dataset for language-model pretraining.

Format:
  data/processed/<name>.bin      uint16 little-endian token IDs
  data/processed/<name>.meta.json metadata (vocab_size, n_tokens)

For fine-tuning we use a JSONL of {"input_ids": [...], "labels": [...]}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    """Random fixed-length windows from a single .bin file. Causal LM target = shifted input."""

    def __init__(self, bin_path: str | Path, seq_len: int) -> None:
        self.path = Path(bin_path)
        self.seq_len = seq_len
        meta = json.loads((self.path.with_suffix(".meta.json")).read_text())
        self.n_tokens = int(meta["n_tokens"])
        self.vocab_size = int(meta["vocab_size"])
        # Memory-map for low RAM use
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        # Number of non-overlapping windows of (seq_len + 1)
        self._n = max(0, (self.n_tokens - 1) // self.seq_len)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Random starting position covers more of the corpus than fixed strides.
        # We use idx to seed for reproducibility.
        rng = np.random.default_rng(idx)
        start = int(rng.integers(0, self.n_tokens - self.seq_len - 1))
        chunk = np.asarray(self.data[start : start + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return {"input_ids": x, "labels": y}


class JsonlSFTDataset(Dataset):
    """Supervised fine-tune dataset where examples already contain input_ids/labels."""

    def __init__(self, jsonl_path: str | Path, seq_len: int, pad_id: int) -> None:
        self.path = Path(jsonl_path)
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.examples: list[tuple[list[int], list[int]]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                ids = ex["input_ids"][: seq_len]
                labels = ex["labels"][: seq_len]
                self.examples.append((ids, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids, labels = self.examples[idx]
        L = self.seq_len
        # Right-pad
        pad_n = L - len(ids)
        ids = ids + [self.pad_id] * pad_n
        labels = labels + [-100] * pad_n
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
