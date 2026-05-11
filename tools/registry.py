"""Tool registry + dispatch."""
from __future__ import annotations

import inspect
from typing import Any

from tools.schema import ToolContext, ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_json_schema() for t in self._tools.values()]

    async def call(self, name: str, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown_tool: {name}"}
        try:
            res = tool.handler(args, ctx)
            if inspect.isawaitable(res):
                res = await res
            if not isinstance(res, dict):
                res = {"result": res}
            return res
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}


_default: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """Lazy-build a registry containing all built-in tools."""
    global _default
    if _default is not None:
        return _default
    reg = ToolRegistry()
    # Local import to avoid cycles
    from tools.builtin.calculator import TOOL as calc
    from tools.builtin.web_search import TOOL as websearch
    from tools.builtin.url_fetch import TOOL as urlfetch
    from tools.builtin.file_read import TOOL as fileread
    from tools.builtin.code_exec import TOOL as codeexec
    from tools.builtin.knowledge_base import TOOL as kb
    from tools.builtin.datetime_tool import TOOL as dt
    for t in (calc, websearch, urlfetch, fileread, codeexec, kb, dt):
        reg.register(t)
    _default = reg
    return reg
