"""Return the current UTC datetime."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tools.schema import ToolContext, ToolSpec


async def _run(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "datetime": now.isoformat(),
        "date": now.date().isoformat(),
        "time": now.time().isoformat(timespec="seconds"),
        "timezone": "UTC",
    }


TOOL = ToolSpec(
    name="current_time",
    description="Get the current UTC date and time. Useful for time-aware answers.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
    handler=_run,
)
