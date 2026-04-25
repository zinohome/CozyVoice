"""Layer 2/3: /v1/voice/chat SSE streaming tests."""

from __future__ import annotations

import json
from base64 import b64decode
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from cozyvoice.providers.base import STTResult
from cozyvoice.providers.stt.mock import MockSTT
from cozyvoice.providers.tts.mock import MockTTS


def _make_app(*, tts_enabled: bool = False):
    from cozyvoice.api.rest import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.stt = MockSTT(canned_text="北京天气")
    app.state.tts = MockTTS() if tts_enabled else None
    app.state.tts_config = {"default_voice_id": "mock", "default_format": "wav"}

    brain = AsyncMock()

    async def fake_stream(**kwargs):
        for chunk in ["北京明天", "晴，22度"]:
            yield chunk

    brain.chat_stream = fake_stream
    app.state.brain_client = brain
    return app


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into list of {event, data} dicts."""
    events = []
    current_event = None
    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            data_str = line[6:].strip()
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = data_str
            events.append({"event": current_event, "data": data})
            current_event = None
    return events


async def test_voice_chat_sse_without_tts() -> None:
    app = _make_app(tts_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
        data = {"session_id": "s1", "personality_id": "p1"}
        r = await c.post(
            "/v1/voice/chat?tts=false",
            files=files, data=data,
            headers={"Authorization": "Bearer fake-jwt"},
        )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    events = _parse_sse(r.text)
    event_types = [e["event"] for e in events]
    assert "stt" in event_types
    assert "reply_done" in event_types
    assert "tts_audio" not in event_types

    stt_event = next(e for e in events if e["event"] == "stt")
    assert stt_event["data"]["text"] == "北京天气"

    done_event = next(e for e in events if e["event"] == "reply_done")
    assert "北京明天" in done_event["data"]["text"]


async def test_voice_chat_sse_with_tts() -> None:
    app = _make_app(tts_enabled=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
        data = {"session_id": "s1", "personality_id": "p1"}
        r = await c.post(
            "/v1/voice/chat?tts=true",
            files=files, data=data,
            headers={"Authorization": "Bearer fake-jwt"},
        )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    event_types = [e["event"] for e in events]
    assert "tts_audio" in event_types
    tts_event = next(e for e in events if e["event"] == "tts_audio")
    assert tts_event["data"]["format"] in ("wav", "mp3")
    audio_bytes = b64decode(tts_event["data"]["base64"])
    assert len(audio_bytes) > 0


async def test_voice_chat_sse_reply_chunks_streamed() -> None:
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
        data = {"session_id": "s1", "personality_id": "p1"}
        r = await c.post(
            "/v1/voice/chat",
            files=files, data=data,
            headers={"Authorization": "Bearer fake-jwt"},
        )
    events = _parse_sse(r.text)
    chunk_events = [e for e in events if e["event"] == "reply_chunk"]
    assert len(chunk_events) >= 2
    assert chunk_events[0]["data"]["delta"] == "北京明天"


async def test_voice_chat_missing_auth_returns_401() -> None:
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"fake", "audio/wav")}
        data = {"session_id": "s1", "personality_id": "p1"}
        r = await c.post("/v1/voice/chat", files=files, data=data)
    assert r.status_code == 401


async def test_voice_chat_stt_failure_returns_error_event() -> None:
    app = _make_app()
    app.state.stt = AsyncMock()
    app.state.stt.transcribe = AsyncMock(side_effect=ConnectionError("stt down"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"fake", "audio/wav")}
        data = {"session_id": "s1", "personality_id": "p1"}
        r = await c.post(
            "/v1/voice/chat",
            files=files, data=data,
            headers={"Authorization": "Bearer fake-jwt"},
        )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) >= 1
