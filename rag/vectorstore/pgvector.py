"""Vector store. Uses pgvector when available, falls back to in-Python cosine."""
from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import Document, DocumentChunk, USE_PGVECTOR


def upsert_chunks(
    db: Session,
    *,
    document_id: str,
    chunks: list[str],
    embeddings: list[np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> int:
    rows = [
        DocumentChunk(
            document_id=str(document_id),
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
    document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if USE_PGVECTOR:
        return _pgvector_search(db, query_vec, top_k, document_ids)
    return _python_cosine_search(db, query_vec, top_k, document_ids)


def _pgvector_search(db, query_vec, top_k, document_ids):
    stmt = (
        select(
            DocumentChunk.id, DocumentChunk.document_id, DocumentChunk.chunk_index,
            DocumentChunk.content, Document.filename,
            DocumentChunk.embedding.cosine_distance(query_vec.tolist()).label("distance"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by("distance").limit(top_k)
    )
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
    rows = db.execute(stmt).all()
    return [
        {
            "chunk_id": str(r.id), "document_id": str(r.document_id),
            "filename": r.filename, "chunk_index": r.chunk_index,
            "snippet": r.content,
            "score": float(1.0 - r.distance) if r.distance is not None else None,
        }
        for r in rows
    ]


def _python_cosine_search(db, query_vec, top_k, document_ids):
    """Fallback for SQLite: pull all chunks (or filtered), cosine in numpy."""
    stmt = select(
        DocumentChunk.id, DocumentChunk.document_id, DocumentChunk.chunk_index,
        DocumentChunk.content, DocumentChunk.embedding, Document.filename,
    ).join(Document, Document.id == DocumentChunk.document_id)
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
    rows = db.execute(stmt).all()
    if not rows:
        return []
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    scored = []
    for r in rows:
        emb = r.embedding
        if emb is None:
            continue
        v = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(v) + 1e-9
        score = float(np.dot(q, v) / n)
        scored.append((score, r))
    scored.sort(key=lambda t: -t[0])
    out = []
    for score, r in scored[:top_k]:
        out.append({
            "chunk_id": str(r.id), "document_id": str(r.document_id),
            "filename": r.filename, "chunk_index": r.chunk_index,
            "snippet": r.content, "score": score,
        })
    return out
