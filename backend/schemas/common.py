from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class OkResponse(BaseModel):
    ok: bool = True
    detail: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    detail: str
