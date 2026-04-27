"""Tests for the in-memory IP rate limiter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from cozyvoice.middleware.rate_limit import RateLimiter


def _make_request(ip: str = "127.0.0.1") -> MagicMock:
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = ip
    req.headers = {}
    return req


async def test_under_limit_passes():
    limiter = RateLimiter(requests_per_minute=5)
    req = _make_request("10.0.0.1")
    # 5 requests should all succeed
    for _ in range(5):
        await limiter.check(req)  # no exception


async def test_over_limit_raises_429():
    limiter = RateLimiter(requests_per_minute=3)
    req = _make_request("10.0.0.2")
    for _ in range(3):
        await limiter.check(req)
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check(req)
    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["error"]["code"] == "RATE_LIMITED"


async def test_different_ips_are_independent():
    limiter = RateLimiter(requests_per_minute=2)
    req_a = _make_request("192.168.1.1")
    req_b = _make_request("192.168.1.2")

    await limiter.check(req_a)
    await limiter.check(req_a)
    # req_a is at limit; req_b should still be fine
    await limiter.check(req_b)
    await limiter.check(req_b)

    with pytest.raises(HTTPException):
        await limiter.check(req_a)
    with pytest.raises(HTTPException):
        await limiter.check(req_b)


async def test_no_client_uses_unknown_key():
    limiter = RateLimiter(requests_per_minute=2)
    req = MagicMock()
    req.client = None
    # Should not raise for the first two calls
    await limiter.check(req)
    await limiter.check(req)
    with pytest.raises(HTTPException):
        await limiter.check(req)


async def test_old_entries_are_cleaned():
    """Entries older than 60 seconds are pruned, freeing the window."""
    import time

    limiter = RateLimiter(requests_per_minute=2)
    ip = "10.1.1.1"
    # Manually inject stale timestamps (older than 60s)
    limiter._requests[ip] = [time.monotonic() - 120, time.monotonic() - 90]

    req = _make_request(ip)
    # Both old entries should be pruned; first check succeeds
    await limiter.check(req)
    assert len(limiter._requests[ip]) == 1  # only the new timestamp


async def test_rate_limiter_integration_with_transcribe_endpoint():
    """Verify 429 is returned via the transcribe endpoint when rate limit exceeded."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from cozyvoice.api.rest import router
    from cozyvoice.providers.stt.mock import MockSTT

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.stt = MockSTT(canned_text="hello")
    app.state.tts = None
    app.state.tts_config = {}
    app.state.brain_client = None
    app.state.rate_limiter = RateLimiter(requests_per_minute=2)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("t.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
        r1 = await c.post("/v1/voice/transcribe", files=files)
        r2 = await c.post("/v1/voice/transcribe", files=files)
        r3 = await c.post("/v1/voice/transcribe", files=files)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429


def _make_request_with_headers(ip: str = "127.0.0.1", xff: str | None = None, xri: str | None = None) -> MagicMock:
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = ip
    headers = {}
    if xff is not None:
        headers["x-forwarded-for"] = xff
    if xri is not None:
        headers["x-real-ip"] = xri
    req.headers = headers
    return req


async def test_rate_limit_uses_x_forwarded_for():
    limiter = RateLimiter(requests_per_minute=2)
    req_a = _make_request_with_headers(ip="172.17.0.1", xff="10.0.0.1, 172.17.0.1")
    req_b = _make_request_with_headers(ip="172.17.0.1", xff="10.0.0.2, 172.17.0.1")
    await limiter.check(req_a)
    await limiter.check(req_a)
    await limiter.check(req_b)
    await limiter.check(req_b)
    with pytest.raises(HTTPException):
        await limiter.check(req_a)
    with pytest.raises(HTTPException):
        await limiter.check(req_b)


async def test_rate_limit_uses_x_real_ip_fallback():
    limiter = RateLimiter(requests_per_minute=2)
    req = _make_request_with_headers(ip="172.17.0.1", xri="192.168.1.100")
    await limiter.check(req)
    await limiter.check(req)
    with pytest.raises(HTTPException):
        await limiter.check(req)


async def test_rate_limit_falls_back_to_client_host():
    limiter = RateLimiter(requests_per_minute=2)
    req = _make_request_with_headers(ip="10.10.10.10")
    await limiter.check(req)
    await limiter.check(req)
    with pytest.raises(HTTPException):
        await limiter.check(req)
