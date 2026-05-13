"""SQLAlchemy engine + session helpers. Supports Postgres+pgvector and SQLite."""
from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import get_settings

_settings = get_settings()
_url = _settings.database_url

# SQLite needs special connect_args
connect_args = {"check_same_thread": False} if _url.startswith("sqlite") else {}

engine = create_engine(_url, pool_pre_ping=True, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def session_factory() -> Session:
    return SessionLocal()


def init_db_if_needed() -> None:
    """Auto-create tables on SQLite where Alembic isn't available."""
    if _url.startswith("sqlite"):
        # Import models so they register on Base.metadata
        from backend.database import models  # noqa: F401
        Base.metadata.create_all(bind=engine)
