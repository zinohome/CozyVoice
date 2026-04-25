"""Layer 1: /v1/voice/transcribe endpoint tests."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from cozyvoice.providers.base import STTResult
from cozyvoice.providers.stt.mock import MockSTT


def _make_app():
    from cozyvoice.api.rest import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.stt = MockSTT(canned_text="北京天气怎么样")
    app.state.tts = None
    app.state.tts_config = {}
    app.state.brain_client = None
    return app


async def test_transcribe_returns_text() -> None:
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
        r = await c.post("/v1/voice/transcribe", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "北京天气怎么样"
    assert "language" in body


async def test_transcribe_stt_failure_returns_502() -> None:
    from unittest.mock import AsyncMock
    app = _make_app()
    app.state.stt = AsyncMock()
    app.state.stt.transcribe = AsyncMock(side_effect=ConnectionError("stt down"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"fake-audio", "audio/wav")}
        r = await c.post("/v1/voice/transcribe", files=files)
    assert r.status_code == 502


async def test_transcribe_no_auth_required() -> None:
    """Layer 1 transcribe does NOT require JWT (unlike /voice/chat)."""
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
        r = await c.post("/v1/voice/transcribe", files=files)
    assert r.status_code == 200
