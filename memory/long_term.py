"""Long-term memory store backed by the same RAG vector index.

We treat each "memory" as a chunk in pgvector with a `memory_type` tag.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from backend.database.models import Memory


def add_memory(db: Session, content: str, embedding: list[float],
               memory_type: str = "fact", metadata: dict[str, Any] | None = None) -> Memory:
    m = Memory(content=content, embedding=embedding, memory_type=memory_type,
               meta=metadata or {})
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def list_memories(db: Session, limit: int = 100) -> list[Memory]:
    from sqlalchemy import select
    return list(db.execute(select(Memory).order_by(Memory.created_at.desc()).limit(limit)).scalars())


def delete_memory(db: Session, mid: uuid.UUID) -> bool:
    m = db.get(Memory, mid)
    if not m:
        return False
    db.delete(m)
    db.commit()
    return True
