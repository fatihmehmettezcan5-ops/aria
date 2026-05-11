from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    file_type: str
    file_size: int
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentChunkOut(BaseModel):
    id: uuid.UUID
    chunk_index: int
    content: str

    class Config:
        from_attributes = True


class DocumentDetail(BaseModel):
    document: DocumentOut
    chunks: list[DocumentChunkOut]


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryHit(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    snippet: str
    score: float | None
