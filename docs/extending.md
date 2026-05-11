# Extending Aria

## Adding a new tool

1. Create `tools/builtin/my_tool.py`:

```python
from tools.schema import ToolContext, ToolSpec

async def _run(args, ctx: ToolContext):
    return {"echo": args.get("message", "")}

TOOL = ToolSpec(
    name="echo",
    description="Echo back a message — useful for testing.",
    parameters={
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
        "additionalProperties": False,
    },
    handler=_run,
)
```

2. Register it in `tools/registry.py::get_default_registry`:

```python
from tools.builtin.my_tool import TOOL as my_tool
reg.register(my_tool)
```

3. Add a few SFT examples in `data/scripts/prepare_finetune.py` so the
   model learns to call your tool, then re-run finetune.

That's it — the agent loop and frontend pick it up automatically.

## Switching the embedder (RAG)

Default: `HashingTfidfEncoder` (no training required, dim=768).

Use the neural encoder backed by your trained Transformer:

```python
# in backend/services/document_service.py
from rag.embeddings.model import NeuralEncoder
from backend.services.model_service import ModelService

ms = ModelService.get()
self.encoder = NeuralEncoder(ms.model, ms.tokenizer, device=str(ms.device))
```

Note: the pgvector column is `Vector(768)`. If your model's `d_model` ≠ 768,
either change `EMB_DIM` in `backend/database/models.py` (and create a new
migration that recreates the table) or project to 768 inside the encoder.

## Improving generation quality

- **Train more.** A 50M-param model on 5B tokens already starts to feel
  coherent. The smoke run is intentionally tiny.
- **Bigger preset.** `medium` / `base` need more compute but are dramatic
  improvements at chat tasks.
- **Better SFT data.** The default `EXAMPLES` list is a seed; replace it
  with your own task-specific examples (especially tool-use traces).
- **Sampling.** For small models, low-ish temperature (0.5–0.7) + top_p 0.9
  + repetition_penalty 1.1 is a good baseline.

## Adding auth (beyond API key)

Replace `backend/middleware/auth.py::require_api_key` with a real
auth dependency: OAuth2 password flow, GitHub OAuth, magic links etc.
The rest of the codebase doesn't assume a particular auth model — every
router endpoint takes a single auth dependency.

## Adding multi-user

1. Add a `users` table + `user_id` FK on `sessions`, `documents`, `memories`.
2. Replace the auth dep with one that returns the current user.
3. Filter all queries in `backend/services/*` by `user_id`.

The `ToolContext` already has a `user_id` slot — pass it from your auth dep.
