"""DB-backed session helpers (thin wrapper, the DB models live in backend/)."""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import Message, Session as ChatSession


def list_sessions(db: Session) -> list[ChatSession]:
    return list(db.execute(select(ChatSession).order_by(ChatSession.updated_at.desc())).scalars())


def get_session(db: Session, sid: uuid.UUID) -> ChatSession | None:
    return db.get(ChatSession, sid)


def create_session(db: Session, title: str = "New chat", metadata: dict[str, Any] | None = None) -> ChatSession:
    s = ChatSession(title=title, meta=metadata or {})
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def append_message(db: Session, sid: uuid.UUID, role: str, content: str,
                   metadata: dict[str, Any] | None = None) -> Message:
    m = Message(session_id=sid, role=role, content=content, meta=metadata or {})
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def delete_session(db: Session, sid: uuid.UUID) -> bool:
    s = db.get(ChatSession, sid)
    if not s:
        return False
    db.delete(s)
    db.commit()
    return True
