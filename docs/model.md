# Model

Aria uses a from-scratch decoder-only Transformer. Every component is
implemented in `model/` using only PyTorch primitives (no `transformers`,
no `from_pretrained`, no flash-attention helpers).

## Components

| File | What it does |
|--|--|
| `model/embeddings.py` | Token embedding wrapper (`nn.Embedding`). |
| `model/rope.py` | RoPE: `build_rope_cache`, `apply_rope`, KV-cache offset support. |
| `model/rmsnorm.py` | Pre-norm RMSNorm in fp32 for stability. |
| `model/attention.py` | MHA / GQA with explicit scaled dot-product + causal mask + KV cache. |
| `model/feedforward.py` | SwiGLU MLP (`gate * up → down`). |
| `model/transformer_block.py` | Pre-norm block: x + Attn(norm(x)); x + FFN(norm(x)). |
| `model/transformer.py` | Stack + final norm + LM head, with weight tying. Save/load. |
| `model/kv_cache.py` | `LayerKVCache` / `KVCache` for incremental decode. |
| `model/config.py` | Dataclass + named presets: tiny / small / medium / base / smoke. |

## Sizes

| Preset | Layers | d_model | Heads | KV heads | d_ff | Ctx | Approx. params |
|--|--|--|--|--|--|--|--|
| smoke  | 2  | 128  | 4  | 2  | 256  | 256  | ~0.5M (vocab dependent) |
| tiny   | 6  | 256  | 4  | 4  | 688  | 512  | ~10M |
| small  | 12 | 512  | 8  | 4  | 1376 | 1024 | ~50M |
| medium | 16 | 768  | 12 | 4  | 2048 | 2048 | ~150M |
| base   | 24 | 1024 | 16 | 4  | 2730 | 4096 | ~350M |

(All include the embedding/output matrix; with weight tying we count it once.)

## Forward / inference equivalence

`model/tests/test_transformer.py::test_cached_decode_matches_full` proves
that step-by-step generation with the KV cache produces *exactly* the same
logits as a single-shot full-context forward pass. This is the most important
correctness test for the inference path.

## Training a real (not smoke) model

See `docs/training.md` for the full procedure. Hyperparameter starting points:

| Preset | Tokens | Batch (seq*grad_accum) | LR (peak) | Warmup | GPU |
|--|--|--|--|--|--|
| small  | ~5B    | 64 × 1024              | 3e-4      | 1k    | 1 × 12GB |
| medium | ~30B   | 256 × 2048             | 2.5e-4    | 2k    | 4 × 24GB |
| base   | ~100B+ | 1M tokens / step       | 2e-4      | 4k    | multi-GPU |

These are *starting points*; tune by your data + budget.
