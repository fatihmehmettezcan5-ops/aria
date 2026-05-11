"""Search the user's uploaded documents (RAG)."""
from __future__ import annotations

from typing import Any

from tools.schema import ToolContext, ToolSpec


async def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    k = max(1, min(int(args.get("top_k") or 5), 12))
    if not query:
        return {"error": "query required"}
    retriever = (ctx.metadata or {}).get("retriever")
    if retriever is None:
        return {"error": "no_retriever"}
    try:
        hits = await retriever.retrieve(query, top_k=k, user_id=ctx.user_id)
    except Exception as e:  # noqa: BLE001
        return {"error": f"retrieve_failed: {e}"}
    return {"query": query, "hits": hits}


TOOL = ToolSpec(
    name="knowledge_base",
    description=(
        "Search the user's uploaded documents for passages relevant to a query. "
        "Each hit returns filename, chunk index, score and snippet — cite these as sources."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 12, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=_run,
)
