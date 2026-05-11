"""Tokenise a directory of .txt files and pack the IDs into a single .bin."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tokenizer.tokenizer import Tokenizer


def pack_corpus(
    corpus_dirs: list[str],
    tokenizer_path: str,
    out_path: str,
    add_eos_between_docs: bool = True,
) -> None:
    tok = Tokenizer.from_file(tokenizer_path)
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for d in corpus_dirs:
        p = Path(d)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.txt")))
        elif p.is_file():
            files.append(p)
    print(f"[pack] {len(files)} files")
    if tok.vocab_size > 65535:
        raise ValueError("vocab_size > 65535 needs uint32 storage")

    # Stream-write to disk to avoid OOM on big corpora.
    n_tokens = 0
    with open(out_path_p, "wb") as f:
        for fp in tqdm(files, desc="tokenising"):
            text = fp.read_text(encoding="utf-8", errors="replace")
            ids = tok.encode(text)
            if add_eos_between_docs:
                ids.append(tok.eos_id)
            arr = np.asarray(ids, dtype=np.uint16)
            arr.tofile(f)
            n_tokens += len(arr)

    meta = {"vocab_size": tok.vocab_size, "n_tokens": n_tokens}
    out_path_p.with_suffix(".meta.json").write_text(json.dumps(meta))
    print(f"[pack] wrote {out_path_p} — {n_tokens:,} tokens")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pack_corpus(args.corpus, args.tokenizer, args.out)


if __name__ == "__main__":
    main()
