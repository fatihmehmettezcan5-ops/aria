"""Read an uploaded document by ID and return its parsed text content.

Resolves the document via DocumentService (passed in ToolContext.metadata).
"""
from __future__ import annotations

from typing import Any

from tools.schema import ToolContext, ToolSpec

MAX_RETURN_CHARS = 8_000


async def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    doc_id = (args.get("document_id") or "").strip()
    if not doc_id:
        return {"error": "document_id required"}
    svc = (ctx.metadata or {}).get("document_service")
    if svc is None:
        return {"error": "no_document_service"}
    try:
        doc = await svc.get_full_text(doc_id)
    except Exception as e:  # noqa: BLE001
        return {"error": f"read_failed: {e}"}
    if doc is None:
        return {"error": "document_not_found"}
    text = doc.get("content", "") or ""
    if len(text) > MAX_RETURN_CHARS:
        text = text[:MAX_RETURN_CHARS] + "\n...[truncated]"
    return {
        "document_id": doc_id,
        "filename": doc.get("filename"),
        "text": text,
    }


TOOL = ToolSpec(
    name="file_read",
    description="Read the full text content of a previously-uploaded document by its ID.",
    parameters={
        "type": "object",
        "properties": {"document_id": {"type": "string"}},
        "required": ["document_id"],
        "additionalProperties": False,
    },
    handler=_run,
)
