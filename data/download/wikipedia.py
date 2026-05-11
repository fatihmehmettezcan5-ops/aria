"""Sample Wikipedia articles via the public REST 'extract' API.

This is intentionally small — designed to give us a few MB of clean
encyclopedic text without needing a 20GB dump. Pass `--n` for total
article count and `--lang` for `en` / `tr` etc.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from data.processing.cleaner import clean_text

USER_AGENT = "AriaTrainer/1.0 (educational; contact@example.com)"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_random_titles(lang: str, n: int) -> list[str]:
    titles: list[str] = []
    while len(titles) < n:
        url = (
            f"https://{lang}.wikipedia.org/w/api.php?action=query&list=random"
            f"&rnnamespace=0&rnlimit=20&format=json"
        )
        data = _get(url)
        for item in data.get("query", {}).get("random", []):
            titles.append(item["title"])
            if len(titles) >= n:
                break
        time.sleep(0.5)
    return titles


def fetch_extract(lang: str, title: str) -> str | None:
    qs = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "redirects": 1,
        }
    )
    data = _get(f"https://{lang}.wikipedia.org/w/api.php?{qs}")
    pages = data.get("query", {}).get("pages", {})
    for p in pages.values():
        ex = p.get("extract")
        if ex:
            return clean_text(ex)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/wikipedia")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    out_dir = Path(args.out) / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[wiki] requesting {args.n} random titles in {args.lang}…")
    titles = fetch_random_titles(args.lang, args.n)
    written = 0
    for t in titles:
        safe = "".join(c if c.isalnum() else "_" for c in t)[:80] or "untitled"
        out = out_dir / f"{safe}.txt"
        if out.exists():
            continue
        try:
            text = fetch_extract(args.lang, t)
            if text and len(text) > 500:
                out.write_text(text, encoding="utf-8")
                written += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ! {t}: {e}")
        time.sleep(0.3)
    print(f"[wiki] wrote {written} files to {out_dir}")


if __name__ == "__main__":
    main()
