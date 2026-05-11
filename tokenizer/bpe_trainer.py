"""Byte-Pair Encoding trainer (from scratch).

Algorithm:
  1. Pretokenise the corpus with a GPT-2-style regex.
  2. Map each pretoken's bytes to printable unicode chars (so the entire BPE
     domain is text — keeps things debuggable).
  3. Each pretoken becomes a tuple of single-char "symbols" with a frequency.
  4. Repeatedly:
       a) count all adjacent symbol-pairs (weighted by pretoken freq)
       b) merge the most-frequent pair into a new symbol everywhere
       c) record the merge rule
     until vocabulary target is reached.
  5. The final vocab is: special tokens + 256 byte-symbols + all merge results.

This is a straightforward O(n_merges * unique_pretokens) implementation,
not the highly optimised priority-queue version, but it's easy to read and
plenty fast for a smoke corpus or a few hundred MB of text.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import regex as re
import yaml
from tqdm import tqdm

from tokenizer.special_tokens import SpecialTokens
from tokenizer.utils import GPT2_SPLIT_PATTERN, bytes_to_unicode


def _iter_corpus(paths: list[str]):
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for f in sorted(path.rglob("*.txt")):
                yield f.read_text(encoding="utf-8", errors="replace")
        elif path.is_file():
            yield path.read_text(encoding="utf-8", errors="replace")


def _pretokenise(text: str, pattern) -> list[str]:
    return [m.group(0) for m in pattern.finditer(text)]


def _to_symbols(pretoken: str, b2u: dict[int, str]) -> tuple[str, ...]:
    return tuple(b2u[b] for b in pretoken.encode("utf-8"))


def train_bpe(
    corpus_paths: list[str],
    vocab_size: int,
    min_pair_freq: int = 2,
    verbose: bool = True,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Returns (vocab_tokens_in_id_order, merges)."""
    pat = re.compile(GPT2_SPLIT_PATTERN)
    b2u = bytes_to_unicode()

    # 1) Count pretokens
    if verbose:
        print("[bpe] pretokenising corpus...")
    pretoken_freqs: Counter[str] = Counter()
    for text in _iter_corpus(corpus_paths):
        for pt in _pretokenise(text, pat):
            pretoken_freqs[pt] += 1
    if verbose:
        print(f"[bpe] {len(pretoken_freqs):,} unique pretokens")

    # 2) Convert to symbol-tuples + freq
    word_freqs: dict[tuple[str, ...], int] = {}
    for pt, f in pretoken_freqs.items():
        word_freqs[_to_symbols(pt, b2u)] = f

    # 3) Initial vocab: specials + 256 byte symbols
    specials = SpecialTokens.all()
    base_alphabet = list(b2u.values())
    vocab: list[str] = list(specials) + base_alphabet
    seen: set[str] = set(vocab)

    target_merges = vocab_size - len(vocab)
    if target_merges <= 0:
        raise ValueError(
            f"vocab_size {vocab_size} too small (need > {len(vocab)} for specials+bytes)"
        )

    # 4) Iteratively merge best pair.
    # We maintain `pair_counts` (global weighted pair counts) and `pair_to_words`
    # (which words contain a given pair) so each merge updates only affected words.
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_to_words: dict[tuple[str, str], set[tuple[str, ...]]] = {}

    def add_word(word: tuple[str, ...], freq: int) -> None:
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pair_counts[p] += freq
            pair_to_words.setdefault(p, set()).add(word)

    def remove_word(word: tuple[str, ...], freq: int) -> None:
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pair_counts[p] -= freq
            if pair_counts[p] <= 0:
                pair_counts.pop(p, None)
                pair_to_words.pop(p, None)
            else:
                ws = pair_to_words.get(p)
                if ws is not None:
                    ws.discard(word)

    for w, f in word_freqs.items():
        add_word(w, f)

    merges: list[tuple[str, str]] = []
    pbar = tqdm(total=target_merges, disable=not verbose, desc="bpe merges")
    while len(merges) < target_merges and pair_counts:
        best_pair, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        if best_count < min_pair_freq:
            break
        a, b = best_pair
        new_sym = a + b
        if new_sym not in seen:
            vocab.append(new_sym)
            seen.add(new_sym)
        merges.append(best_pair)

        # Apply merge to all words containing this pair.
        affected = list(pair_to_words.get(best_pair, ()))
        for w in affected:
            f = word_freqs.pop(w, None)
            if f is None:
                continue
            # Build new word by merging adjacent (a,b) → ab
            new_w: list[str] = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    new_w.append(new_sym)
                    i += 2
                else:
                    new_w.append(w[i])
                    i += 1
            new_word = tuple(new_w)
            remove_word(w, f)
            word_freqs[new_word] = word_freqs.get(new_word, 0) + f
            add_word(new_word, f)
        pbar.update(1)
    pbar.close()

    return vocab, merges


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab, merges = train_bpe(
        corpus_paths=cfg["corpus"],
        vocab_size=int(cfg["vocab_size"]),
        min_pair_freq=int(cfg.get("min_pair_freq", 2)),
        verbose=True,
    )

    # Save in our portable JSON format.
    out_path = out_dir / cfg.get("filename", "tokenizer.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 1,
                "specials": SpecialTokens.all(),
                "vocab": vocab,  # list ordered by id
                "merges": [list(m) for m in merges],
            },
            f,
            ensure_ascii=False,
        )
    print(f"[bpe] wrote {out_path} (vocab={len(vocab)}, merges={len(merges)})")


if __name__ == "__main__":
    main()
