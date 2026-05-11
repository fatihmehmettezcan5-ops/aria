"""Singletons reused across routers."""
from __future__ import annotations

from functools import lru_cache

from backend.services.chat_service import ChatService


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService()
