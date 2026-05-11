# Training Guide

Aria has three training stages:

1. **Tokenizer training** (BPE on a text corpus).
2. **Pretraining** (next-token prediction on packed token streams).
3. **Supervised fine-tuning** (chat + tool-use format with masked loss).

## 0. Smoke test (CPU, ~3 min)

```bash
make smoke-train
```

This runs the entire pipeline end-to-end on a synthetic mini-corpus.
Produces `runs/smoke/{tokenizer.json,final.pt}` so the API can boot
immediately. Loss should drop from ~7 to <1 in ~400 steps.

## 1. Real corpus

Download some public-domain text:

```bash
python -m data.download.gutenberg              # ~15 books, ~30MB
python -m data.download.wikipedia --n 500      # 500 random EN articles
python -m data.download.wikipedia --lang tr --n 200  # 200 TR articles
```

Files land in `data/raw/{gutenberg,wikipedia/{en,tr}}/*.txt`.

## 2. Train tokenizer

Edit `configs/tokenizer.yaml` to point at your corpus dirs and pick a vocab
size (32k is a good default).

```bash
make tokenizer
# → runs/tokenizer/tokenizer.json
```

## 3. Pretrain

```bash
make pretrain
# uses configs/train_small.yaml
```

This packs the corpus into `data/processed/pretrain_small.bin` (uint16,
memory-mapped), then trains. Watch with TensorBoard:

```bash
tensorboard --logdir runs/pretrain/tb
```

You should see `train/loss` decrease, `train/lr` follow the warmup-cosine
shape, and `val/ppl` follow `train/loss`. Throughput depends on your GPU:
expect 5k–30k tokens/sec on a single consumer GPU at `small` size with bf16.

### Resuming

Set `train.resume: runs/pretrain/step_5000.pt` in the YAML, or pass
`--resume` if you wrap the script.

### Multi-GPU

The trainer is single-process by default; for multi-GPU, wrap the script with
`torchrun`:

```bash
torchrun --standalone --nproc_per_node=4 -m training.scripts.train_pretrain \
         --config configs/train_medium.yaml
```

The model is plain `nn.Module`; wrap with `DDP` inside `Trainer.__init__`
when `dist.is_initialized()`.

## 4. Fine-tune (chat + tool calling)

```bash
make finetune
# uses configs/finetune_small.yaml + runs/pretrain/best.pt
```

The fine-tune dataset is generated from `data/scripts/prepare_finetune.py` —
add your own examples to the `EXAMPLES` list there. The format is:

```
<BOS><SYSTEM>...<END>
<USER>...<END>
<ASSISTANT><TOOL_CALL>{...}</TOOL_CALL><TOOL_RESULT>{...}</TOOL_RESULT>final answer<END>
```

Loss is masked to **only the assistant's emitted tokens** (including the
tool-call span — the model must learn to emit it — but excluding the
tool-result span, which arrives from the system).

## 5. Serve

Point the API at your fine-tuned checkpoint:

```bash
ARIA_CHECKPOINT=runs/finetune/best.pt \
ARIA_TOKENIZER=runs/tokenizer/tokenizer.json \
docker compose -f infrastructure/docker-compose.yml up
```

## Tips

- Start with `tiny` to sanity-check loss curves before spending GPU hours.
- **Always** validate KV-cache equivalence after editing attention:
  `pytest model/tests/test_transformer.py -k cached_decode`.
- Use `bf16` if your GPU supports it (Ampere+). Fall back to `fp32` on CPU.
- Gradient accumulation is your friend on small GPUs.
- Repetition penalty (1.05–1.15) at inference helps small models a lot.
