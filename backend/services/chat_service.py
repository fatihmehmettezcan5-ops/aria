"""Chat orchestration service: ties model, tools, RAG, and DB together."""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from sqlalchemy import select
from sqlalchemy.orm import Session

from agent.orchestrator import AgentEvent, AgentOrchestrator
from backend.database.connection import session_factory
from backend.database.models import Message, Session as ChatSession
from backend.services.document_service import DocumentService
from backend.services.model_service import ModelService
from inference.generator import GenerationConfig
from rag.retrieval.retriever import Retriever
from tools.registry import get_default_registry
from tools.schema import ToolContext


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


class ChatService:
    def __init__(self) -> None:
        self.model_svc = ModelService.get()
        self.doc_svc = DocumentService.get()
        self.registry = get_default_registry()
        self.retriever = Retriever(self.doc_svc.encoder, db_factory=session_factory)

    # ---- DB helpers ----
    def list_sessions(self, db: Session) -> list[ChatSession]:
        return list(db.execute(select(ChatSession).order_by(ChatSession.updated_at.desc())).scalars())

    def get_session(self, db: Session, sid: uuid.UUID) -> ChatSession | None:
        return db.get(ChatSession, str(sid))

    def create_session(self, db: Session, *, title: str | None, language: str) -> ChatSession:
        s = ChatSession(title=title or ("Yeni sohbet" if language == "tr" else "New chat"),
                        language=language)
        db.add(s)
        db.commit()
        db.refresh(s)
        return s

    def delete_session(self, db: Session, sid: uuid.UUID) -> bool:
        s = db.get(ChatSession, str(sid))
        if not s:
            return False
        db.delete(s)
        db.commit()
        return True

    def history_pairs(self, session: ChatSession) -> list[tuple[str, str]]:
        return [(m.role, m.content) for m in session.messages if m.role in ("user", "assistant")]

    # ---- streaming chat ----
    async def stream_message(
        self,
        db: Session,
        session: ChatSession,
        user_text: str,
        *,
        gen_overrides: dict | None = None,
    ) -> AsyncIterator[bytes]:
        # Persist user message
        umsg = Message(session_id=session.id, role="user", content=user_text, meta={})
        db.add(umsg)
        db.commit()

        history = self.history_pairs(session)
        # Last entry is the just-added user message — drop it (we pass user_message separately).
        history = [h for h in history if not (h[0] == "user" and h[1] == user_text)][:50]

        gen_cfg = self.model_svc.default_gen_cfg()
        if gen_overrides:
            for k, v in gen_overrides.items():
                if v is not None and hasattr(gen_cfg, k):
                    setattr(gen_cfg, k, v)

        agent = AgentOrchestrator(self.model_svc.generator, self.registry, gen_cfg=gen_cfg)
        ctx = ToolContext(
            db=db,
            session_id=str(session.id),
            metadata={"retriever": self.retriever, "document_service": self.doc_svc},
        )

        yield _sse("start", {"session_id": str(session.id)})

        final_text = ""
        tool_calls: list[dict] = []
        try:
            async for ev in agent.run(
                language=session.language,
                history=history,
                user_message=user_text,
                ctx=ctx,
            ):
                yield _sse(ev.type, ev.data)
                if ev.type == "done":
                    final_text = ev.data.get("final_text", "") or ""
                    tool_calls = ev.data.get("tool_calls", [])
        except Exception as e:  # noqa: BLE001
            yield _sse("error", {"message": str(e)})

        # Persist assistant message with tool trace
        amsg = Message(
            session_id=session.id, role="assistant",
            content=final_text or "",
            meta={"tool_calls": tool_calls},
        )
        db.add(amsg)
        # Auto-title from first user message if still default
        if session.title in ("New chat", "Yeni sohbet"):
            session.title = user_text.strip()[:60] or session.title
        db.commit()
