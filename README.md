---
title: Aria
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: A from-scratch AI assistant — own model, tokenizer, RAG, tools.
---

# Aria — A From-Scratch AI Assistant

Aria is a complete, self-contained AI assistant where **everything is built
from scratch**: the BPE tokenizer, the decoder-only Transformer (RoPE,
RMSNorm, SwiGLU, GQA, KV-cache), the data pipeline, the training loop, the
inference engine, the tool-calling system, the RAG stack, the FastAPI
backend, the Next.js UI, and the Docker deployment.

> **No pretrained weights. No external LLM APIs. No `from_pretrained()`.
> No LangChain, LlamaIndex, transformers/AutoModel, sentence-transformers.**
> Allowed primitives only: PyTorch / NumPy / standard Python.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                              Browser                                 │
│             Next.js 14 App Router · Tailwind · TS                    │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ /api/proxy → cookies, SSE
┌────────────────────────▼─────────────────────────────────────────────┐
│                       FastAPI backend                                │
│  routers: health · chat · documents · tools · model                  │
│  services: ChatService · DocumentService · ModelService              │
│  middleware: api-key auth · rate-limit · structured logging          │
└──┬──────────────┬─────────────────────┬─────────────────┬────────────┘
   │              │                     │                 │
┌──▼──┐    ┌──────▼──────┐      ┌───────▼────────┐  ┌─────▼─────┐
│ DB  │    │ Agent loop  │      │ Tool registry  │  │ RAG       │
│ PG+ │    │ orchestr.   │      │ calc · search  │  │ embedder  │
│vec  │    │ tool-loop   │      │ fetch · code   │  │ chunker   │
└─────┘    │ prompt bldr │      │ kb · time · file│  │ retriever │
           └──────┬──────┘      └────────────────┘  └───────────┘
                  │
           ┌──────▼──────┐
           │  Inference  │  Generator · Sampler · KV-cache · Streaming
           │   engine    │
           └──────┬──────┘
                  │
           ┌──────▼──────┐
           │  Our model  │  Decoder-only Transformer (PyTorch)
           │  + tokenizer│  RoPE · RMSNorm · SwiGLU · GQA
           └─────────────┘
```

## Repo layout

| Path           | What lives here |
|----------------|-----------------|
| `tokenizer/`   | BPE trainer + encoder/decoder + special tokens (+ tests) |
| `model/`       | Embeddings, RoPE, attention (MHA/GQA), SwiGLU MLP, RMSNorm, KV-cache, full Transformer (+ tests) |
| `data/`        | Download / clean / dedupe / pack / tokenize pipelines |
| `training/`    | Loop, scheduler, optimizer, checkpoints, AMP, TensorBoard |
| `inference/`   | Generator: greedy / temperature / top-k / top-p / repetition penalty + streaming (+ tests) |
| `tools/`       | Tool schema, registry, parser, executor + 7 built-in tools (+ tests) |
| `memory/`      | Conversation history, sessions, summarisation, long-term memory |
| `rag/`         | TF-IDF embedder (+ neural alternative), parsers, chunker, pgvector store, retriever (+ tests) |
| `agent/`       | Prompt builder, response parser, agent orchestrator (+ tests) |
| `backend/`     | FastAPI app, routers, services, schemas, middleware, Alembic migrations |
| `frontend/`    | Next.js 14 chat UI |
| `infrastructure/` | Dockerfiles, compose, Nginx, deploy scripts |
| `configs/`     | YAML configs for tokenizer, model sizes, training, inference |
| `docs/`        | Architecture · model · training · deployment · API · extending · troubleshooting |
| `tests/`       | End-to-end integration tests |

## 60-second quick start (CPU smoke test)

```bash
# 1. Install deps
make setup

# 2. Train tiny tokenizer + tiny model on bundled mini-corpus (~3 min on CPU)
make smoke-train

# 3. Start the full stack (DB, backend, frontend)
make up
# → open http://localhost:3000
```

The smoke checkpoint that lands at `runs/smoke/{tokenizer.json,final.pt}`
is intentionally tiny (~426K params) — it exists so the pipeline runs
end-to-end. Real quality needs real training (see below).

## Real training (GPU)

```bash
# 1. Download some public-domain text
python -m data.download.gutenberg
python -m data.download.wikipedia --n 500
python -m data.download.wikipedia --lang tr --n 200

# 2. Train tokenizer
python -m tokenizer.bpe_trainer --config configs/tokenizer.yaml

# 3. Pretrain
python -m training.scripts.train_pretrain --config configs/train_small.yaml

# 4. Fine-tune (chat + tool use)
python -m training.scripts.train_finetune --config configs/finetune_small.yaml \
       --ckpt runs/pretrain/best.pt

# 5. Serve
ARIA_CHECKPOINT=runs/finetune/best.pt \
ARIA_TOKENIZER=runs/tokenizer/tokenizer.json \
docker compose -f infrastructure/docker-compose.yml up
```

See [`docs/`](docs/) for full guides.

## What works (verified)

- ✅ **Tokenizer**: BPE from scratch, 6 round-trip tests pass (ASCII + Türkçe + emoji + special tokens + save/load).
- ✅ **Model**: Forward shape, save/load, parameter count, **KV-cache equivalence to full forward** — 8 tests pass.
- ✅ **Inference**: Greedy + sampler (temp/top-k/top-p/rep-penalty), streaming generator — 3 tests pass.
- ✅ **Tools**: Registry dispatch, parser, calculator AST safety, 6 tests pass.
- ✅ **RAG**: Sentence-aware chunker, hashing TF-IDF encoder, similarity ranking — 3 tests pass.
- ✅ **Agent**: Prompt builder, system prompt EN/TR, end-to-end loop emits tool/done events — 2 tests pass.
- ✅ **End-to-end**: Train tokenizer → train model → generate → run agent loop in one test.
- ✅ **Smoke training pipeline runs in 60s on CPU**, loss drops 6.7 → 0.45 in pretrain, then SFT loss drops 10 → 1.7.

```
$ pytest -q
29 passed in ~5s
```

## Honest expectations

This is an educational, end-to-end build. The smoke-trained model is tiny
and trained on a tiny corpus — it produces real text but not coherent
answers. The architecture, training code, agent loop, RAG and deployment
are all production-shaped: scale them by giving more compute + data, or
by jumping to the `medium`/`base` presets.

## License

MIT. See [`LICENSE`](LICENSE).
