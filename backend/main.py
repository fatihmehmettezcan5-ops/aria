"""FastAPI application entrypoint.

Serves both the API (under /api) and a built-in HTML chat UI (at /).
The HTML UI is enough for Hugging Face Spaces deployment without a
separate frontend service.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.config import get_settings
from backend.database.connection import init_db_if_needed
from backend.middleware.logging import configure_logging, get_logger
from backend.routers import chat as chat_router
from backend.routers import documents as documents_router
from backend.routers import health as health_router
from backend.routers import model as model_router
from backend.routers import tools as tools_router

configure_logging()
log = get_logger("aria.api")
_settings = get_settings()
_static_dir = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-create tables for SQLite (HF Spaces). For Postgres, use Alembic.
    try:
        init_db_if_needed()
    except Exception as e:  # noqa: BLE001
        log.warning("init_db_failed", error=str(e))
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Aria API",
        version="0.1.0",
        description="From-scratch AI assistant: chat, RAG, tool calling.",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        t0 = time.perf_counter()
        try:
            resp = await call_next(request)
        except Exception:
            log.exception("unhandled_error", path=request.url.path, method=request.method)
            return JSONResponse({"detail": "internal server error"}, status_code=500)
        log.info("request",
                 method=request.method, path=request.url.path,
                 status=resp.status_code,
                 duration_ms=round((time.perf_counter() - t0) * 1000, 2))
        return resp

    app.include_router(health_router.router)
    app.include_router(chat_router.router, prefix="/api")
    app.include_router(documents_router.router, prefix="/api")
    app.include_router(tools_router.router, prefix="/api")
    app.include_router(model_router.router, prefix="/api")

    # Built-in HTML UI at root
    @app.get("/", include_in_schema=False)
    def root():
        index = _static_dir / "index.html"
        if index.exists():
            return FileResponse(index)
        return JSONResponse({"detail": "UI not bundled; API only."}, status_code=200)

    return app


app = create_app()
