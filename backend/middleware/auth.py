"""Optional API-key auth. If ARIA_API_KEY is empty, auth is bypassed."""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from backend.config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    s = get_settings()
    if not s.aria_api_key:
        return  # disabled
    if x_api_key != s.aria_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing api key")
