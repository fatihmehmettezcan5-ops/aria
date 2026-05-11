"""Generate a small synthetic+stylised corpus for the smoke training run.

We don't require an internet download. Instead we synthesise a varied,
multi-domain corpus on the fly: simple instructions, narrative snippets,
arithmetic problems, basic Python code, and a touch of Turkish so we can
verify TR roundtrips. This is enough to exercise the full pipeline end-to-end
and produce a tiny model that doesn't output gibberish.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

EN_FACTS = [
    "The sun is a star at the centre of our solar system.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "Python is a high-level interpreted programming language.",
    "An octopus has three hearts and blue blood.",
    "Mount Everest is the highest mountain above sea level.",
    "The Pacific is the largest ocean on Earth.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "A leap year occurs every four years to keep the calendar aligned.",
    "Honey never spoils when stored properly in a sealed jar.",
    "The Great Wall of China stretches for thousands of kilometres.",
]

TR_FACTS = [
    "Ankara, Türkiye'nin başkentidir.",
    "Su deniz seviyesinde yüz derecede kaynar.",
    "Güneş, Güneş Sistemi'nin merkezindeki bir yıldızdır.",
    "İstanbul iki kıtada yer alan bir şehirdir.",
    "Python yüksek seviyeli bir programlama dilidir.",
    "Yapay zekâ, makinelerin öğrenmesine ve karar vermesine olanak tanır.",
    "Dünya kendi ekseni etrafında yaklaşık yirmi dört saatte döner.",
    "Bal, doğru saklandığında bozulmayan bir besindir.",
]

INSTRUCTIONS = [
    ("Add two numbers", "def add(a, b):\n    return a + b\n"),
    ("Reverse a string", "def reverse(s):\n    return s[::-1]\n"),
    ("Find the maximum of a list", "def maximum(xs):\n    return max(xs)\n"),
    ("Square a number", "def square(x):\n    return x * x\n"),
    ("Greet a person", "def greet(name):\n    return f'Hello, {name}!'\n"),
    ("Check if a number is even", "def is_even(n):\n    return n % 2 == 0\n"),
]

CONVERSATIONS = [
    [
        ("user", "What is the capital of France?"),
        ("assistant", "The capital of France is Paris."),
    ],
    [
        ("user", "How many continents are there?"),
        ("assistant", "There are seven continents on Earth."),
    ],
    [
        ("user", "Write a haiku about the sea."),
        ("assistant", "Endless blue horizon\nWaves whisper to the white sand\nGulls cry overhead"),
    ],
    [
        ("user", "Türkiye'nin başkenti neresidir?"),
        ("assistant", "Türkiye'nin başkenti Ankara'dır."),
    ],
    [
        ("user", "Add 5 and 7."),
        ("assistant", "5 plus 7 equals 12."),
    ],
    [
        ("user", "What does this Python function do?\ndef add(a, b):\n    return a + b"),
        ("assistant", "It returns the sum of two numbers a and b."),
    ],
]


def synthesise(out_path: Path, n_lines: int = 6000, seed: int = 7) -> None:
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []

    for _ in range(n_lines):
        kind = rng.choices(["fact", "tr", "math", "code", "story"], weights=[3, 1, 2, 2, 2])[0]
        if kind == "fact":
            parts.append(rng.choice(EN_FACTS))
        elif kind == "tr":
            parts.append(rng.choice(TR_FACTS))
        elif kind == "math":
            a, b = rng.randint(1, 50), rng.randint(1, 50)
            op = rng.choice(["+", "-", "*"])
            r = eval(f"{a}{op}{b}")  # noqa: S307 — safe constants
            parts.append(f"Question: What is {a} {op} {b}?\nAnswer: {a} {op} {b} = {r}.")
        elif kind == "code":
            desc, fn = rng.choice(INSTRUCTIONS)
            parts.append(f"Task: {desc}.\n{fn}")
        else:
            parts.append(
                "Once upon a time in a small village, "
                + rng.choice(["a baker", "a poet", "a sailor", "a child"])
                + " " + rng.choice(["dreamed of distant lands.", "wrote about the stars.",
                                    "sang to the river.", "learned to read the wind."])
            )

    # Plus a synthetic chat-format block to seed conversational structure.
    chat_text = []
    for _ in range(800):
        conv = rng.choice(CONVERSATIONS)
        for role, content in conv:
            chat_text.append(f"<{role.upper()}>\n{content}\n</{role.upper()}>")
        chat_text.append("")
    parts.append("\n".join(chat_text))

    out_path.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"[smoke] wrote {out_path} ({out_path.stat().st_size:,} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/smoke/corpus.txt")
    ap.add_argument("--lines", type=int, default=6000)
    args = ap.parse_args()
    synthesise(Path(args.out), n_lines=args.lines)


if __name__ == "__main__":
    main()
