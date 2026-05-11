"""Web search via DuckDuckGo HTML (no API key required)."""
from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from tools.builtin._safe_http import DEFAULT_TIMEOUT, USER_AGENT
from tools.schema import ToolContext, ToolSpec


async def _ddg(query: str, max_results: int) -> list[dict[str, Any]]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT,
                                 headers={"User-Agent": USER_AGENT},
                                 follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    out: list[dict[str, Any]] = []
    for res in soup.select(".result")[:max_results]:
        a = res.select_one("a.result__a")
        sn = res.select_one(".result__snippet")
        if not a:
            continue
        out.append({
            "title": a.get_text(strip=True),
            "url": a.get("href", ""),
            "snippet": sn.get_text(" ", strip=True) if sn else "",
        })
    return out


async def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    n = max(1, min(int(args.get("max_results") or 5), 10))
    if not query:
        return {"error": "query required"}
    try:
        results = await _ddg(query, n)
    except Exception as e:  # noqa: BLE001
        return {"error": f"search_failed: {e}"}
    return {"query": query, "results": results, "provider": "duckduckgo"}


TOOL = ToolSpec(
    name="web_search",
    description="Search the public web (DuckDuckGo). Returns title/url/snippet for each result.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=_run,
)
