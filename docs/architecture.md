# Architecture

Aria is split into small, replaceable layers. Every layer has a clean
boundary and tests; nothing reaches into another layer's internals.

```
                       ┌────────────────────────────────┐
                       │         Browser (UI)           │
                       │   Next.js · React · Tailwind   │
                       └──────────────┬─────────────────┘
                          /api/proxy/*│  (cookies + SSE)
                                      ▼
   ┌──────────────────────────── FastAPI ─────────────────────────────┐
   │  routers (chat / documents / tools / model / health)             │
   │  middleware (auth · rate-limit · structured logs)                │
   │  services  (chat_service · document_service · model_service)     │
   └──────────────┬───────────────────┬───────────────────┬───────────┘
                  │                   │                   │
        ┌─────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
        │  Agent loop      │ │  Tool registry  │ │  Document RAG   │
        │  prompt_builder  │ │  calculator     │ │  parser         │
        │  orchestrator    │ │  web_search     │ │  chunker        │
        │  tool parser     │ │  url_fetch      │ │  TF-IDF encoder │
        └─────────┬────────┘ │  file_read      │ │  pgvector       │
                  │          │  code_exec      │ │  retriever      │
                  ▼          │  knowledge_base │ └────────┬────────┘
        ┌──────────────────┐ │  current_time   │          │
        │  Inference engine│ └─────────────────┘          │
        │  Generator       │                              │
        │  Sampler · KV    │                              ▼
        │  cache · Stops   │                     ┌────────────────┐
        └─────────┬────────┘                     │ PostgreSQL +   │
                  │                              │ pgvector       │
                  ▼                              └────────────────┘
        ┌──────────────────┐
        │  Our Transformer │  decoder-only · RoPE · RMSNorm · SwiGLU · GQA
        │  + BPE tokenizer │
        └──────────────────┘
```

## Data flow — one chat turn

1. Browser POSTs `{content}` to `/api/proxy/chat/sessions/{id}/messages`.
2. Next.js proxies the request as-is to FastAPI; cookies + SSE stream are passed through.
3. `chat_service.stream_message`:
   - persists the user message,
   - builds a `ToolContext` (with retriever + document service),
   - constructs an `AgentOrchestrator` and yields its events as SSE.
4. The orchestrator builds the prompt (system + tool docs + history + user),
   then calls `Generator.stream(...)` and yields tokens.
5. While yielding, it watches the decoded buffer for a complete
   `<TOOL_CALL>{...}</TOOL_CALL>` span; if one appears it:
   - parses it,
   - executes the tool,
   - re-builds the prompt with the tool result span appended,
   - resumes generation.
6. When the model finishes (EOS / `<END>` / max tokens), the orchestrator
   emits a `done` event with the final text and the full tool-call trace.
7. `chat_service` persists the assistant message including the tool trace.

## Why these choices

| Choice | Reason |
|--|--|
| Decoder-only Transformer | Standard, well-understood, the right shape for chat. |
| RoPE | Cheap relative positions, plays well with KV-cache. |
| RMSNorm + SwiGLU | Pre-norm + gated MLP — strong at small parameter counts. |
| GQA | KV memory savings for inference without big quality loss. |
| Byte-level BPE | Deterministic round-trip for any unicode (TR/EN/code/emoji). |
| Hashing TF-IDF embedder | Works *immediately* with no training. Swap to neural via `rag/embeddings/model.py` later. |
| In-band tool calls (`<TOOL_CALL>` spans) | Trains via vanilla LM loss; no special head needed. |
| pgvector | Single store for chunks + embeddings; familiar Postgres ops. |
| FastAPI streaming | SSE works through Nginx with `proxy_buffering off`. |
| Next.js standalone | Tiny prod image, file-based routing, SSR-friendly proxy route. |
