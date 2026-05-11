"""Retrieval glue: encode query → search pgvector → return hits."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from rag.embeddings.encoder import HashingTfidfEncoder
from rag.vectorstore.pgvector import similarity_search


class Retriever:
    def __init__(self, encoder: HashingTfidfEncoder, db_factory) -> None:
        self.encoder = encoder
        # db_factory is a callable returning a fresh Session (avoids cross-thread issues).
        self.db_factory: Any = db_factory

    async def retrieve(self, query: str, *, top_k: int = 5,
                       user_id: str | None = None) -> list[dict[str, Any]]:
        vec = self.encoder.encode_one(query)
        db: Session = self.db_factory()
        try:
            return similarity_search(db, query_vec=vec, top_k=top_k)
        finally:
            db.close()
