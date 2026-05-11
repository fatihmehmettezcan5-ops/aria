"""Backend settings (env-driven)."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["development", "production", "test"] = "development"
    app_secret: str = "change-me"
    public_base_url: str = "http://localhost:3000"
    allowed_origins: str = "http://localhost:3000"

    aria_checkpoint: str = "runs/smoke/final.pt"
    aria_tokenizer: str = "runs/smoke/tokenizer.json"
    aria_device: Literal["auto", "cpu", "cuda"] = "auto"
    aria_max_new_tokens: int = 512
    aria_default_temperature: float = 0.8
    aria_default_top_p: float = 0.95
    aria_default_top_k: int = 50

    database_url: str = "postgresql+psycopg://aria:aria@db:5432/aria"
    redis_url: str = "redis://redis:6379/0"

    max_upload_mb: int = 20
    max_fetch_bytes: int = 2_000_000
    rate_limit_per_minute: int = 60

    aria_api_key: str = ""

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
