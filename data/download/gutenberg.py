"""Download Project Gutenberg books (public-domain).

Gutenberg has rate limits; this script is intentionally polite. Pass `--ids`
to choose specific books, or `--top` to fetch the top-N popular books.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import urllib.request

from data.processing.cleaner import clean_text


# Curated default IDs covering English + a couple of multilingual works.
DEFAULT_IDS = [
    1342,    # Pride and Prejudice
    11,      # Alice's Adventures in Wonderland
    84,      # Frankenstein
    1661,    # Sherlock Holmes
    98,      # A Tale of Two Cities
    74,      # Tom Sawyer
    345,     # Dracula
    2701,    # Moby Dick
    1080,    # A Modest Proposal
    100,     # Shakespeare complete
    16328,   # Beowulf
    219,     # Heart of Darkness
    2554,    # Crime and Punishment (English)
    8800,    # Divine Comedy
    25344,   # Scarlet Letter
]

USER_AGENT = "AriaTrainer/1.0 (educational; +https://github.com/aria)"


def _strip_gutenberg_header(text: str) -> str:
    start = re.search(r"\*\*\*\s*START OF.+?\*\*\*", text, re.IGNORECASE)
    end = re.search(r"\*\*\*\s*END OF.+?\*\*\*", text, re.IGNORECASE)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start()]
    return text.strip()


def download_one(book_id: int, out_dir: Path) -> Path | None:
    out = out_dir / f"gutenberg_{book_id}.txt"
    if out.exists():
        return out
    urls = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="replace")
            text = clean_text(_strip_gutenberg_header(raw))
            if len(text) > 500:
                out.write_text(text, encoding="utf-8")
                return out
        except Exception as e:  # noqa: BLE001
            print(f"  ! {url}: {e}")
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/gutenberg")
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--sleep", type=float, default=1.5)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = args.ids or DEFAULT_IDS
    for bid in ids:
        print(f"[gutenberg] {bid}")
        p = download_one(bid, out_dir)
        if p:
            print(f"  -> {p} ({p.stat().st_size:,} bytes)")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
