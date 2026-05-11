"""Health endpoint smoke test (in-process, no DB or model needed)."""
import importlib

from fastapi.testclient import TestClient


def test_health_endpoint(monkeypatch):
    # Use sqlite to avoid needing Postgres for this test.
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("ARIA_API_KEY", "")
    # Import after env set
    import backend.config as cfg
    cfg.get_settings.cache_clear()
    main = importlib.reload(importlib.import_module("backend.main"))
    client = TestClient(main.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
