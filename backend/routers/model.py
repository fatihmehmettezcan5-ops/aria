from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.middleware.auth import require_api_key
from backend.schemas.chat import GenerateRequest, GenerateResponse
from backend.services.model_service import ModelService
from inference.generator import GenerationConfig

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info")
def info(_=Depends(require_api_key)) -> dict:
    return ModelService.get().info()


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, _=Depends(require_api_key)) -> GenerateResponse:
    svc = ModelService.get()
    cfg = GenerationConfig(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=1.1,
    )
    text = svc.generator.generate_text(req.prompt, cfg)
    return GenerateResponse(text=text)
