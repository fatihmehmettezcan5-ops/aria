from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.dependencies import get_chat_service
from backend.middleware.auth import require_api_key
from backend.middleware.rate_limit import check_rate_limit
from backend.schemas.chat import (
    MessageOut, SendMessage, SessionCreate, SessionDetail, SessionOut, SessionUpdate,
)
from backend.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


def _service() -> ChatService:
    return get_chat_service()


# ---- sessions ----

@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(db: Annotated[Session, Depends(get_db)], _=Depends(require_api_key)):
    return [SessionOut.model_validate(s) for s in _service().list_sessions(db)]


@router.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(body: SessionCreate, db: Annotated[Session, Depends(get_db)],
                   _=Depends(require_api_key)):
    s = _service().create_session(db, title=body.title, language=body.language)
    return SessionOut.model_validate(s)


@router.get("/sessions/{sid}", response_model=SessionDetail)
def get_session(sid: uuid.UUID, db: Annotated[Session, Depends(get_db)],
                _=Depends(require_api_key)):
    s = _service().get_session(db, sid)
    if not s:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    return SessionDetail(
        session=SessionOut.model_validate(s),
        messages=[MessageOut.model_validate(m) for m in s.messages],
    )


@router.patch("/sessions/{sid}", response_model=SessionOut)
def update_session(sid: uuid.UUID, body: SessionUpdate,
                   db: Annotated[Session, Depends(get_db)], _=Depends(require_api_key)):
    s = _service().get_session(db, sid)
    if not s:
        raise HTTPException(404, "not found")
    if body.title is not None:
        s.title = body.title.strip()[:200] or s.title
    if body.language is not None:
        s.language = body.language
    db.commit(); db.refresh(s)
    return SessionOut.model_validate(s)


@router.delete("/sessions/{sid}", status_code=204)
def delete_session(sid: uuid.UUID, db: Annotated[Session, Depends(get_db)],
                   _=Depends(require_api_key)):
    if not _service().delete_session(db, sid):
        raise HTTPException(404, "not found")


# ---- messaging ----

@router.post("/sessions/{sid}/messages")
async def stream_message(sid: uuid.UUID, body: SendMessage, request: Request,
                         db: Annotated[Session, Depends(get_db)],
                         _=Depends(require_api_key)):
    check_rate_limit(request)
    s = _service().get_session(db, sid)
    if not s:
        raise HTTPException(404, "not found")
    overrides = {"temperature": body.temperature, "top_p": body.top_p, "top_k": body.top_k}
    gen = _service().stream_message(db, s, body.content, gen_overrides=overrides)
    return StreamingResponse(
        gen, media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
