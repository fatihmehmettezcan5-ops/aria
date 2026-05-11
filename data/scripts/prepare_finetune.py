"""Build a JSONL of supervised fine-tuning examples (input_ids/labels).

Examples are formatted with our chat tokens:

   <BOS><SYSTEM>...<END><USER>...<END><ASSISTANT>...<END>

For tool examples we include synthetic <TOOL_CALL>/<TOOL_RESULT> spans so the
model learns to emit them. Loss is masked to only the assistant output (and
tool-call spans) — i.e. we don't train on the user/system prompt tokens.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tokenizer.special_tokens import SpecialTokens
from tokenizer.tokenizer import Tokenizer


SYSTEM_DEFAULT_EN = (
    "You are Aria, a small but helpful AI assistant. "
    "Answer concisely. When you need to compute math, search the web, fetch "
    "a page or query the knowledge base, emit a tool call wrapped in "
    "<TOOL_CALL>...</TOOL_CALL> with valid JSON, then wait for "
    "<TOOL_RESULT>...</TOOL_RESULT> before answering."
)

SYSTEM_DEFAULT_TR = (
    "Sen Aria adlı küçük ama yardımcı bir yapay zekâ asistanısın. "
    "Kısa ve net yanıt ver. Matematik, web araması, sayfa getirme veya bilgi "
    "tabanı sorgusu gerektiğinde <TOOL_CALL>...</TOOL_CALL> içinde geçerli "
    "JSON ile araç çağrısı yap; <TOOL_RESULT>...</TOOL_RESULT> dönmeden cevap verme."
)

# A small but varied seed of conversational examples, including tool use.
EXAMPLES = [
    # Plain QA
    {"sys": "en", "u": "Hello!", "a": "Hi! How can I help you today?"},
    {"sys": "en", "u": "What is the capital of France?", "a": "The capital of France is Paris."},
    {"sys": "en", "u": "Who wrote Hamlet?", "a": "William Shakespeare wrote Hamlet."},
    {"sys": "en", "u": "Name three primary colours.", "a": "Red, blue, and yellow."},
    {"sys": "tr", "u": "Merhaba!", "a": "Merhaba! Sana nasıl yardımcı olabilirim?"},
    {"sys": "tr", "u": "Türkiye'nin başkenti neresidir?", "a": "Türkiye'nin başkenti Ankara'dır."},
    {"sys": "tr", "u": "Suyun kaynama noktası kaç derecedir?", "a": "Su deniz seviyesinde 100°C'de kaynar."},

    # Calculator tool use
    {"sys": "en", "u": "What is 23 times 47?",
     "tool_call": {"name": "calculator", "arguments": {"expression": "23 * 47"}},
     "tool_result": {"result": 1081},
     "a": "23 × 47 = 1081."},
    {"sys": "en", "u": "Compute (15+9)*3.",
     "tool_call": {"name": "calculator", "arguments": {"expression": "(15+9)*3"}},
     "tool_result": {"result": 72},
     "a": "(15 + 9) × 3 = 72."},
    {"sys": "tr", "u": "12 ile 8'in çarpımı kaçtır?",
     "tool_call": {"name": "calculator", "arguments": {"expression": "12 * 8"}},
     "tool_result": {"result": 96},
     "a": "12 × 8 = 96."},

    # Time tool
    {"sys": "en", "u": "What is today's date?",
     "tool_call": {"name": "current_time", "arguments": {}},
     "tool_result": {"datetime": "2024-06-15T10:23:00Z"},
     "a": "Today's date is 2024-06-15 (UTC)."},

    # KB tool
    {"sys": "en", "u": "What does my uploaded document say about budget?",
     "tool_call": {"name": "knowledge_base", "arguments": {"query": "budget"}},
     "tool_result": {"hits": [{"snippet": "The Q3 budget is $1.2M.", "filename": "report.pdf"}]},
     "a": "According to your document `report.pdf`, the Q3 budget is $1.2M."},

    # Web search tool
    {"sys": "en", "u": "Search the web for 'OpenAI'.",
     "tool_call": {"name": "web_search", "arguments": {"query": "OpenAI"}},
     "tool_result": {"results": [{"title": "OpenAI", "url": "https://openai.com",
                                  "snippet": "AI research and deployment company"}]},
     "a": "I found a result: **OpenAI** — an AI research and deployment company (https://openai.com)."},
]


def build_example(tok: Tokenizer, ex: dict, seq_len: int) -> dict | None:
    S = SpecialTokens
    sys_text = SYSTEM_DEFAULT_TR if ex.get("sys") == "tr" else SYSTEM_DEFAULT_EN

    ids: list[int] = [tok.bos_id]
    labels: list[int] = [-100]

    def add(role_tag: str, text: str, train: bool) -> None:
        ids.append(tok.token_to_id[role_tag]); labels.append(-100)
        text_ids = tok.encode(text)
        ids.extend(text_ids)
        labels.extend(text_ids if train else [-100] * len(text_ids))
        ids.append(tok.token_to_id[S.END])
        labels.append(tok.token_to_id[S.END] if train else -100)

    # System (no training)
    add(S.SYSTEM, sys_text, train=False)
    # User (no training)
    add(S.USER, ex["u"], train=False)

    # Assistant (train) — possibly with tool call and post-tool answer.
    ids.append(tok.token_to_id[S.ASSISTANT]); labels.append(-100)

    if "tool_call" in ex:
        # Tool call span — TRAIN (model must learn to emit it).
        for sp in (S.TOOL_CALL,):
            ids.append(tok.token_to_id[sp]); labels.append(tok.token_to_id[sp])
        call_json = json.dumps(ex["tool_call"], ensure_ascii=False)
        call_ids = tok.encode(call_json)
        ids.extend(call_ids); labels.extend(call_ids)
        ids.append(tok.token_to_id[S.TOOL_CALL_END]); labels.append(tok.token_to_id[S.TOOL_CALL_END])

        # Tool result span — DO NOT TRAIN (model receives this from the system).
        for sp in (S.TOOL_RESULT,):
            ids.append(tok.token_to_id[sp]); labels.append(-100)
        res_json = json.dumps(ex["tool_result"], ensure_ascii=False)
        res_ids = tok.encode(res_json)
        ids.extend(res_ids); labels.extend([-100] * len(res_ids))
        ids.append(tok.token_to_id[S.TOOL_RESULT_END]); labels.append(-100)

    # Final natural-language answer — TRAIN
    ans_ids = tok.encode(ex["a"])
    ids.extend(ans_ids); labels.extend(ans_ids)
    ids.append(tok.token_to_id[S.END]); labels.append(tok.token_to_id[S.END])

    if len(ids) > seq_len:
        return None
    return {"input_ids": ids, "labels": labels}


def synthesise_math_examples(n: int, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for _ in range(n):
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        op = rng.choice(["+", "-", "*"])
        expr = f"{a} {op} {b}"
        result = eval(expr.replace(" ", ""))  # noqa: S307
        lang = rng.choice(["en", "tr"])
        if lang == "en":
            u = f"What is {expr}?"
            a_text = f"{expr} = {result}."
        else:
            u = f"{expr} kaçtır?"
            a_text = f"{expr} = {result}."
        out.append({
            "sys": lang, "u": u,
            "tool_call": {"name": "calculator", "arguments": {"expression": expr.replace(" ", "")}},
            "tool_result": {"result": result},
            "a": a_text,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--synth-math", type=int, default=200,
                    help="how many synthetic math+tool examples to add")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    rng = random.Random(0)
    examples = list(EXAMPLES) + synthesise_math_examples(args.synth_math, rng)
    rng.shuffle(examples)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_p.open("w", encoding="utf-8") as f:
        for ex in examples:
            built = build_example(tok, ex, args.seq_len)
            if built is None:
                continue
            f.write(json.dumps(built, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"[finetune] wrote {n_written} examples to {out_p}")


if __name__ == "__main__":
    main()
