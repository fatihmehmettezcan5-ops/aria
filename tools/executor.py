"""Tool execution helper used by the agent loop."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tools.parser import ParsedToolCall
from tools.registry import ToolRegistry
from tools.schema import ToolContext


@dataclass
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    summary: str


def _summarise(name: str, args: dict, result: dict) -> str:
    if "error" in result:
        return f"{name}: error — {result['error']}"
    if name == "calculator":
        return f"calculator({args.get('expression')}) = {result.get('result')}"
    if name == "web_search":
        return f"web_search('{(args.get('query') or '')[:50]}') → {len(result.get('results', []))} results"
    if name == "url_fetch":
        return f"url_fetch → '{(result.get('title') or result.get('url', ''))[:50]}'"
    if name == "knowledge_base":
        return f"knowledge_base('{(args.get('query') or '')[:50]}') → {len(result.get('hits', []))} hits"
    if name == "code_exec":
        return f"code_exec → exit {result.get('exit_code')}"
    if name == "current_time":
        return f"current_time → {result.get('datetime')}"
    if name == "file_read":
        return f"file_read({args.get('document_id')}) → {len(result.get('text', ''))} chars"
    return f"{name} done"


async def execute_call(
    call: ParsedToolCall,
    registry: ToolRegistry,
    ctx: ToolContext,
) -> ToolExecution:
    result = await registry.call(call.name, call.arguments, ctx)
    summary = _summarise(call.name, call.arguments, result)
    return ToolExecution(
        name=call.name,
        arguments=call.arguments,
        result=result,
        summary=summary,
    )


def format_tool_result_for_model(execution: ToolExecution) -> str:
    """Render a tool result as a string the model can consume in-context."""
    payload = {"name": execution.name, "result": execution.result}
    return json.dumps(payload, ensure_ascii=False)
