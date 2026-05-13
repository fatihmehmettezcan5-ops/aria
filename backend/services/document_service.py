"""Document ingestion + read service for tools and the API."""
from __future__ import annotations

import threading
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.connection import session_factory
from backend.database.models import Document, DocumentChunk
from rag.embeddings.encoder import HashingTfidfEncoder
from rag.pipeline import ingest


class DocumentService:
    _instance: "DocumentService | None" = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "DocumentService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = DocumentService()
            return cls._instance

    def __init__(self) -> None:
        # 768 dims to match the pgvector column.
        self.encoder = HashingTfidfEncoder(dim=768, ngram=(1, 2))

    # --- ingest / list / read ---
    async def ingest_upload(self, *, filename: str, mime: str, data: bytes) -> Document:
        db = session_factory()
        try:
            return ingest(db, encoder=self.encoder, filename=filename, mime=mime, data=data)
        finally:
            db.close()

    async def list_documents(self) -> list[Document]:
        db = session_factory()
        try:
            return list(db.execute(
                select(Document).order_by(Document.created_at.desc())
            ).scalars())
        finally:
            db.close()

    async def get_document(self, doc_id: str) -> tuple[Document, list[DocumentChunk]] | None:
        db = session_factory()
        try:
            d = db.get(Document, str(doc_id))
            if not d:
                return None
            chunks = list(d.chunks)
            return d, chunks
        finally:
            db.close()

    async def get_full_text(self, doc_id: str) -> dict | None:
        db = session_factory()
        try:
            d = db.get(Document, str(doc_id))
            if not d:
                return None
            return {"filename": d.filename, "content": d.content}
        finally:
            db.close()

    async def delete_document(self, doc_id: str) -> bool:
        db = session_factory()
        try:
            d = db.get(Document, str(doc_id))
            if not d:
                return False
            db.delete(d)
            db.commit()
            return True
        finally:
            db.close()
