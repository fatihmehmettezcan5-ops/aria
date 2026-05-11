from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.middleware.auth import require_api_key
from tools.registry import get_default_registry
from tools.schema import ToolContext

router = APIRouter(prefix="/tools", tags=["tools"])


class ExecRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = {}


@router.get("")
def list_tools(_=Depends(require_api_key)) -> list[dict]:
    return get_default_registry().schemas()


@router.post("/execute")
async def execute(req: ExecRequest, _=Depends(require_api_key)) -> dict:
    reg = get_default_registry()
    if not reg.get(req.name):
        raise HTTPException(404, f"unknown tool: {req.name}")
    return await reg.call(req.name, req.arguments, ToolContext())
