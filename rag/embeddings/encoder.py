"""Hashing TF-IDF embedder — fully from scratch, deterministic, dependency-free.

We deliberately keep this as our default embedder so the system "just works"
without training a separate neural encoder. It produces fixed-dim L2-normalised
vectors suitable for cosine similarity in pgvector.

For higher quality, see `rag.embeddings.neural_encoder` which uses our own
Transformer's hidden state (mean-pooled). That requires a trained checkpoint.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class HashingTfidfEncoder:
    """Fixed-dim hashing TF-IDF.

    - Tokenise (word-level, unicode-aware).
    - Hash each token to one of `dim` buckets via SHA1.
    - TF: log(1 + count). IDF: maintained as an online estimate from observed corpus.
    - L2-normalise the resulting vector.

    Updating IDF online is optional; if you call `fit(corpus)` first, the IDF
    statistics will be computed from that corpus, otherwise a uniform IDF is used.
    """

    def __init__(self, dim: int = 768, ngram: tuple[int, int] = (1, 2)) -> None:
        self.dim = dim
        self.ngram = ngram
        self._df = np.ones(dim, dtype=np.float64)   # document frequency per bucket
        self._n_docs = 1
        self._lock = threading.Lock()

    # ---- tokenisation ----
    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t.lower() for t in _TOKEN_RE.findall(text)]

    def _ngram_tokens(self, text: str) -> list[str]:
        toks = self._tokens(text)
        out: list[str] = list(toks) if self.ngram[0] <= 1 else []
        for n in range(max(2, self.ngram[0]), self.ngram[1] + 1):
            for i in range(len(toks) - n + 1):
                out.append(" ".join(toks[i : i + n]))
        return out

    def _bucket(self, token: str) -> int:
        h = hashlib.sha1(token.encode("utf-8", "ignore")).digest()
        return int.from_bytes(h[:8], "little") % self.dim

    # ---- fit ----
    def fit(self, corpus: list[str]) -> None:
        df = np.zeros(self.dim, dtype=np.float64)
        for text in corpus:
            buckets = {self._bucket(t) for t in self._ngram_tokens(text)}
            for b in buckets:
                df[b] += 1
        with self._lock:
            self._df = df + 1.0
            self._n_docs = max(1, len(corpus)) + 1

    def update(self, corpus: list[str]) -> None:
        """Online IDF update — appendable (used as users upload new docs)."""
        with self._lock:
            for text in corpus:
                buckets = {self._bucket(t) for t in self._ngram_tokens(text)}
                for b in buckets:
                    self._df[b] += 1
            self._n_docs += len(corpus)

    # ---- encode ----
    def encode_one(self, text: str) -> np.ndarray:
        toks = self._ngram_tokens(text)
        if not toks:
            return np.zeros(self.dim, dtype=np.float32)
        tf = Counter()
        for t in toks:
            tf[self._bucket(t)] += 1
        vec = np.zeros(self.dim, dtype=np.float64)
        with self._lock:
            n_docs = self._n_docs
            df = self._df
        for b, c in tf.items():
            tf_w = math.log(1.0 + c)
            idf = math.log((1.0 + n_docs) / (1.0 + df[b])) + 1.0
            vec[b] = tf_w * idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode_one(t) for t in texts]

    # ---- persistence ----
    def state_dict(self) -> dict:
        return {"dim": self.dim, "ngram": list(self.ngram),
                "df": self._df.tolist(), "n_docs": self._n_docs}

    def load_state_dict(self, state: dict) -> None:
        self.dim = int(state["dim"])
        self.ngram = tuple(state["ngram"])
        self._df = np.asarray(state["df"], dtype=np.float64)
        self._n_docs = int(state["n_docs"])
