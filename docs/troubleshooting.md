# Troubleshooting

## "Model is untrained" warning in the UI

You haven't run training yet, or the env vars don't point at a checkpoint.
Either run `make smoke-train` or set:

```
ARIA_CHECKPOINT=/absolute/path/to/final.pt
ARIA_TOKENIZER=/absolute/path/to/tokenizer.json
```

## Tokenizer training is very slow

The default trainer is O(merges × unique pretokens). For >1GB corpora,
either:
- raise `min_pair_freq` in the YAML (skips rare merges),
- or pre-shard your corpus and train on a subset.

## Generation is gibberish

Smoke models are tiny — they *will* be incoherent. For real quality:
- train at least the `small` preset on hundreds of millions of tokens,
- then SFT on diverse instruction data,
- and sample with `temperature 0.5–0.7`, `top_p 0.9`, `repetition_penalty 1.1`.

## SSE is buffered (UI shows whole answer at once)

Nginx buffers by default. The supplied config sets `proxy_buffering off`
on `/api/`, but if you're behind another proxy (Cloudflare etc.) you
may need to disable buffering there too, or send `Cache-Control: no-cache,
no-transform`.

## `pgvector` extension missing

The `pgvector/pgvector:pg16` image already ships the extension; the first
migration runs `CREATE EXTENSION IF NOT EXISTS vector;`. If you use a
managed Postgres, ensure pgvector is enabled (RDS, Supabase, Neon all
support it).

## Out-of-memory during training

- Reduce `seq_len` and/or `batch_size` in the train YAML.
- Increase `grad_accum` to keep the effective batch.
- Set `grad_checkpointing: true`.
- Use `amp_dtype: bf16` (Ampere+) or `fp16` (older).

## Tool call is emitted but never closes

The model needs to learn `<TOOL_CALL>` AND `</TOOL_CALL>`. Make sure
your SFT examples include both, and that `prepare_finetune.py` is the one
generating your JSONL. The orchestrator stops on `</TOOL_CALL>` as one of
the stop strings, so an unclosed call will exhaust `max_new_tokens`.

## "ModuleNotFoundError: backend"

Run from the project root. Inside the container, `WORKDIR=/app` is the
project root and `backend.main:app` resolves correctly.

## Frontend can't reach backend

Inside docker, the frontend should hit `BACKEND_INTERNAL_URL=http://backend:8000`
(set in compose). Outside docker dev, set `NEXT_PUBLIC_API_BASE` to the
backend URL the *browser* should use.

## Permission denied on `runs/` after `chmod`

The dev compose mounts your project root into the container as `app`
user. If the model files are owned by root from a prior `docker compose
run` invocation, do `sudo chown -R $USER runs/`.
