"""FastAPI application entrypoint."""
from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.middleware.logging import configure_logging, get_logger
from backend.routers import chat as chat_router
from backend.routers import documents as documents_router
from backend.routers import health as health_router
from backend.routers import model as model_router
from backend.routers import tools as tools_router

configure_logging()
log = get_logger("aria.api")
_settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Aria API",
        version="0.1.0",
        description="From-scratch AI assistant: chat, RAG, tool calling.",
        docs_url="/docs",
        redoc_url=None,
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
    return app


app = create_app()
