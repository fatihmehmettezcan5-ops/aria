"""BPE tokenizer (encode / decode) using merges learned by bpe_trainer."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import regex as re

from tokenizer.special_tokens import SpecialTokens
from tokenizer.utils import GPT2_SPLIT_PATTERN, bytes_to_unicode, unicode_to_bytes


class Tokenizer:
    def __init__(
        self,
        vocab: list[str],
        merges: list[tuple[str, str]],
        specials: list[str] | None = None,
    ) -> None:
        self.id_to_token: list[str] = list(vocab)
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.id_to_token)}
        self.bpe_ranks: dict[tuple[str, str], int] = {tuple(m): i for i, m in enumerate(merges)}
        self.specials: list[str] = specials if specials is not None else SpecialTokens.all()
        self.b2u = bytes_to_unicode()
        self.u2b = unicode_to_bytes()
        self._pat = re.compile(GPT2_SPLIT_PATTERN)
        # Regex that matches any special token (so we can split around them).
        if self.specials:
            self._special_pat = re.compile(
                "(" + "|".join(re.escape(s) for s in sorted(self.specials, key=len, reverse=True)) + ")"
            )
        else:
            self._special_pat = None

        # Convenience IDs (raise if missing)
        S = SpecialTokens
        self.pad_id = self.token_to_id.get(S.PAD, 0)
        self.unk_id = self.token_to_id.get(S.UNK, 1)
        self.bos_id = self.token_to_id.get(S.BOS, 2)
        self.eos_id = self.token_to_id.get(S.EOS, 3)
        self.end_id = self.token_to_id.get(S.END, self.eos_id)

    # ---------- persistence ----------
    @classmethod
    def from_file(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            vocab=data["vocab"],
            merges=[tuple(m) for m in data["merges"]],
            specials=data.get("specials"),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "version": 1,
                    "specials": self.specials,
                    "vocab": self.id_to_token,
                    "merges": [list(p) for p in self.bpe_ranks.keys()],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    # ---------- core BPE ----------
    @lru_cache(maxsize=4096)
    def _bpe(self, word: str) -> tuple[str, ...]:
        """Apply learned merges to a single pretoken (already byte→unicode mapped)."""
        if len(word) < 2:
            return (word,)
        symbols = list(word)
        while True:
            pairs = {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}
            best = None
            best_rank = None
            for p in pairs:
                r = self.bpe_ranks.get(p)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best = p
            if best is None:
                break
            a, b = best
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
            if len(symbols) == 1:
                break
        return tuple(symbols)

    # ---------- encode / decode ----------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        # Split around special tokens so they get their own single ID.
        parts: list[tuple[str, bool]]  # (segment, is_special)
        if self._special_pat is not None:
            parts = []
            for chunk in self._special_pat.split(text):
                if chunk == "":
                    continue
                parts.append((chunk, chunk in self.token_to_id and chunk in self.specials))
        else:
            parts = [(text, False)]

        for seg, is_special in parts:
            if is_special:
                ids.append(self.token_to_id[seg])
                continue
            for pretoken in self._pat.findall(seg):
                # bytes → printable unicode → bpe → ids
                mapped = "".join(self.b2u[b] for b in pretoken.encode("utf-8"))
                for sym in self._bpe(mapped):
                    tid = self.token_to_id.get(sym)
                    if tid is None:
                        # Fall back to byte-level decomposition; every byte char
                        # was added to the vocab during training.
                        for ch in sym:
                            ids.append(self.token_to_id.get(ch, self.unk_id))
                    else:
                        ids.append(tid)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        out_chars: list[str] = []
        special_set = set(self.specials)
        for i in ids:
            if i < 0 or i >= len(self.id_to_token):
                continue
            tok = self.id_to_token[i]
            if tok in special_set:
                if skip_special:
                    continue
                # Render special tokens literally so chat formats round-trip.
                out_chars.append(tok)
            else:
                out_chars.append(tok)
        text_unicode = "".join(out_chars)
        # Map unicode-of-bytes back to actual bytes; unmapped chars (i.e. literal
        # special-token text) are passed through as-is.
        byte_buf = bytearray()
        out: list[str] = []

        def flush_bytes():
            if byte_buf:
                out.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()

        # Split text_unicode into runs: special-token spans (kept literal) vs byte-mapped runs.
        # We re-use the special-token regex.
        if self._special_pat is not None and not skip_special:
            for chunk in self._special_pat.split(text_unicode):
                if chunk in special_set:
                    flush_bytes()
                    out.append(chunk)
                else:
                    for ch in chunk:
                        b = self.u2b.get(ch)
                        if b is None:
                            flush_bytes()
                            out.append(ch)
                        else:
                            byte_buf.append(b)
            flush_bytes()
        else:
            for ch in text_unicode:
                b = self.u2b.get(ch)
                if b is None:
                    flush_bytes()
                    out.append(ch)
                else:
                    byte_buf.append(b)
            flush_bytes()
        return "".join(out)

    # Convenience for chat formatting (used by training data + agent).
    def encode_chat(self, turns: list[tuple[str, str]], add_bos: bool = True) -> list[int]:
        """turns = [(role, content), ...] where role ∈ {system,user,assistant,tool}."""
        S = SpecialTokens
        role_map = {
            "system": S.SYSTEM,
            "user": S.USER,
            "assistant": S.ASSISTANT,
            "tool": S.TOOL_RESULT,
        }
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for role, content in turns:
            tag = role_map.get(role, S.USER)
            ids.append(self.token_to_id[tag])
            ids.extend(self.encode(content))
            ids.append(self.token_to_id[S.END])
        return ids
