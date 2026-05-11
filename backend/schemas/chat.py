from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    title: str | None = None
    language: Literal["en", "tr"] = "en"


class SessionUpdate(BaseModel):
    title: str | None = None
    language: Literal["en", "tr"] | None = None


class SessionOut(BaseModel):
    id: uuid.UUID
    title: str
    language: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    class Config:
        from_attributes = True


class SessionDetail(BaseModel):
    session: SessionOut
    messages: list[MessageOut]


class SendMessage(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50


class GenerateResponse(BaseModel):
    text: str
