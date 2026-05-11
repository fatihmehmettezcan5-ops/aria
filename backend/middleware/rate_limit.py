"""Per-IP rate limiter. Uses Redis if reachable, else in-memory fallback."""
from __future__ import annotations

import time
from collections import defaultdict, deque

import redis
from fastapi import HTTPException, Request, status

from backend.config import get_settings

_settings = get_settings()
_buckets: dict[str, deque] = defaultdict(deque)
_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis | None:
    global _redis
    if _redis is not None:
        return _redis
    try:
        c = redis.Redis.from_url(_settings.redis_url, socket_timeout=0.5)
        c.ping()
        _redis = c
        return _redis
    except Exception:  # noqa: BLE001
        return None


def check_rate_limit(request: Request) -> None:
    limit = _settings.rate_limit_per_minute
    if limit <= 0:
        return
    ip = request.client.host if request.client else "anon"
    key = f"rl:{ip}"
    now = time.time()
    window = 60.0
    r = _get_redis()
    if r is not None:
        try:
            pipe = r.pipeline()
            pipe.zremrangebyscore(key, 0, now - window)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, int(window) + 5)
            _, count, _, _ = pipe.execute()
            if count >= limit:
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
            return
        except redis.RedisError:
            pass
    bucket = _buckets[key]
    while bucket and bucket[0] < now - window:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
    bucket.append(now)
