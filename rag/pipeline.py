"""End-to-end document ingestion: parse → chunk → embed → persist."""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from backend.database.models import Document
from rag.documents.chunker import chunk_text
from rag.documents.parser import parse_bytes
from rag.embeddings.encoder import HashingTfidfEncoder
from rag.vectorstore.pgvector import upsert_chunks


def ingest(
    db: Session,
    *,
    encoder: HashingTfidfEncoder,
    filename: str,
    mime: str,
    data: bytes,
) -> Document:
    text = parse_bytes(filename, mime, data)
    doc = Document(
        filename=filename,
        file_type=mime,
        file_size=len(data),
        content=text,
        meta={},
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    chunks = chunk_text(text)
    if not chunks:
        return doc
    # Update IDF with this new corpus, then encode (uses fresh stats).
    encoder.update(chunks)
    embeddings = encoder.encode(chunks)
    upsert_chunks(db, document_id=doc.id, chunks=chunks, embeddings=embeddings,
                  metadata={"filename": filename})
    return doc
