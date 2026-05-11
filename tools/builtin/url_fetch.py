"""Fetch a single URL and return cleaned main-text content."""
from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup

from tools.builtin._safe_http import UnsafeURLError, safe_get
from tools.schema import ToolContext, ToolSpec

MAX_RETURN_CHARS = 12_000


def _extract(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        t.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return title, "\n".join(lines)


async def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    url = (args.get("url") or "").strip()
    if not url:
        return {"error": "url required"}
    try:
        status, headers, body = await safe_get(url)
    except UnsafeURLError as e:
        return {"error": f"unsafe_url: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"fetch_failed: {e}"}
    ctype = headers.get("content-type", "")
    if "text/html" not in ctype and "xhtml" not in ctype:
        snippet = body[:MAX_RETURN_CHARS].decode("utf-8", errors="replace")
        return {"url": url, "status": status, "content_type": ctype, "text": snippet}
    title, text = _extract(body.decode("utf-8", errors="replace"))
    if len(text) > MAX_RETURN_CHARS:
        text = text[:MAX_RETURN_CHARS] + "\n...[truncated]"
    return {"url": url, "status": status, "title": title, "text": text}


TOOL = ToolSpec(
    name="url_fetch",
    description=(
        "Fetch a single webpage by URL and return cleaned main-text content. "
        "Use after web_search to read a result, or when the user provides a URL."
    ),
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    },
    handler=_run,
)
