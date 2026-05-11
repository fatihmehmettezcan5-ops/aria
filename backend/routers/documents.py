from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.config import get_settings
from backend.middleware.auth import require_api_key
from backend.schemas.documents import (
    DocumentChunkOut, DocumentDetail, DocumentOut, QueryHit, QueryRequest,
)
from backend.services.document_service import DocumentService
from rag.documents.parser import ALLOWED_EXTS

router = APIRouter(prefix="/documents", tags=["documents"])


def _svc() -> DocumentService:
    return DocumentService.get()


@router.get("", response_model=list[DocumentOut])
async def list_documents(_=Depends(require_api_key)):
    docs = await _svc().list_documents()
    return [DocumentOut.model_validate(d) for d in docs]


@router.post("/upload", response_model=DocumentOut, status_code=201)
async def upload(file: UploadFile = File(...), _=Depends(require_api_key)):
    settings = get_settings()
    name = (file.filename or "file").lower()
    if not any(name.endswith(ext) for ext in ALLOWED_EXTS):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "unsupported file type")
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file too large")
    doc = await _svc().ingest_upload(
        filename=file.filename or "file",
        mime=file.content_type or "application/octet-stream",
        data=data,
    )
    return DocumentOut.model_validate(doc)


@router.get("/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: uuid.UUID, _=Depends(require_api_key)):
    res = await _svc().get_document(str(doc_id))
    if not res:
        raise HTTPException(404, "not found")
    doc, chunks = res
    return DocumentDetail(
        document=DocumentOut.model_validate(doc),
        chunks=[DocumentChunkOut.model_validate(c) for c in chunks],
    )


@router.delete("/{doc_id}", status_code=204)
async def delete_document(doc_id: uuid.UUID, _=Depends(require_api_key)):
    ok = await _svc().delete_document(str(doc_id))
    if not ok:
        raise HTTPException(404, "not found")


@router.post("/query", response_model=list[QueryHit])
async def query(req: QueryRequest, _=Depends(require_api_key)):
    from backend.dependencies import get_chat_service
    cs = get_chat_service()
    hits = await cs.retriever.retrieve(req.query, top_k=req.top_k)
    return [QueryHit(**h) for h in hits]
