"""SSRF-safe HTTP helpers shared across web-touching tools."""
from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

import httpx

ALLOWED_SCHEMES = {"http", "https"}
USER_AGENT = "AriaBot/1.0 (+https://github.com/aria)"
DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
MAX_FETCH_BYTES = int(os.getenv("MAX_FETCH_BYTES", "2000000"))


class UnsafeURLError(ValueError):
    pass


def _is_private(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return (
        a.is_private or a.is_loopback or a.is_link_local
        or a.is_multicast or a.is_reserved or a.is_unspecified
    )


def assert_safe_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"scheme not allowed: {p.scheme}")
    host = p.hostname
    if not host:
        raise UnsafeURLError("missing host")
    if host.lower() in {"localhost", "metadata.google.internal"}:
        raise UnsafeURLError(f"blocked host: {host}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise UnsafeURLError(f"dns failure: {e}") from e
    for info in infos:
        if _is_private(info[4][0]):
            raise UnsafeURLError(f"private/reserved IP: {info[4][0]}")


async def safe_get(url: str, *, headers: dict[str, str] | None = None,
                   max_bytes: int | None = None) -> tuple[int, dict[str, str], bytes]:
    assert_safe_url(url)
    cap = max_bytes or MAX_FETCH_BYTES
    h = {"User-Agent": USER_AGENT, **(headers or {})}
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True,
                                 headers=h, max_redirects=5) as client:
        async with client.stream("GET", url) as resp:
            chunks: list[bytes] = []
            total = 0
            async for c in resp.aiter_bytes():
                total += len(c)
                if total > cap:
                    raise UnsafeURLError(f"response > {cap} bytes")
                chunks.append(c)
            return resp.status_code, dict(resp.headers), b"".join(chunks)
