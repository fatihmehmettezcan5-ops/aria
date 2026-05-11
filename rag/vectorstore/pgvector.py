"""pgvector-backed vector store for document chunks."""
from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import Document, DocumentChunk


def upsert_chunks(
    db: Session,
    *,
    document_id: uuid.UUID,
    chunks: list[str],
    embeddings: list[np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> int:
    rows = [
        DocumentChunk(
            document_id=document_id,
            chunk_index=i,
            content=chunk,
            embedding=emb.tolist(),
            meta=(metadata or {}),
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True))
    ]
    db.add_all(rows)
    db.commit()
    return len(rows)


def similarity_search(
    db: Session,
    *,
    query_vec: np.ndarray,
    top_k: int = 5,
    document_ids: list[uuid.UUID] | None = None,
) -> list[dict[str, Any]]:
    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            Document.filename,
            DocumentChunk.embedding.cosine_distance(query_vec.tolist()).label("distance"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by("distance")
        .limit(top_k)
    )
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
    rows = db.execute(stmt).all()
    return [
        {
            "chunk_id": str(r.id),
            "document_id": str(r.document_id),
            "filename": r.filename,
            "chunk_index": r.chunk_index,
            "snippet": r.content,
            "score": float(1.0 - r.distance) if r.distance is not None else None,
        }
        for r in rows
    ]
