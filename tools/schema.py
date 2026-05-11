"""Tool definition schema."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


# A tool handler may be sync or async; both are supported by the executor.
ToolHandler = Callable[[dict[str, Any], "ToolContext"], "Awaitable[dict[str, Any]] | dict[str, Any]"]


@dataclass
class ToolContext:
    """Runtime context passed to tool handlers."""
    db: Any | None = None
    user_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]   # JSON schema
    handler: ToolHandler

    def to_json_schema(self) -> dict[str, Any]:
        """Serialise the tool definition for the model / docs."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
