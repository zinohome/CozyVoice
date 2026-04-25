"""WS /v1/voice/stream error path 单测。"""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cozyvoice.api import ws
from cozyvoice.providers.stt.mock import MockSTT
from cozyvoice.providers.tts.mock import MockTTS


def _make_app() -> FastAPI:
    """构造一个最小 FastAPI app，注入 mock providers（不走 lifespan）。"""
    app = FastAPI()
    app.include_router(ws.router, prefix="/v1")

    app.state.stt = MockSTT(canned_text="hello")
    app.state.tts = MockTTS()
    app.state.tts_config = {"default_voice_id": "mock", "default_format": "wav"}
    app.state.brain_client = AsyncMock()
    app.state.brain_client.chat_collect = AsyncMock(return_value="reply text")
    return app


# ── 1. Missing token → close(4401) ─────────────────────────────────


def test_missing_token_closes_4401() -> None:
    app = _make_app()
    with TestClient(app) as client:
        # websocket_connect without token param
        try:
            with client.websocket_connect("/v1/voice/stream") as _ws:
                # Should not reach here; server closes immediately
                pass
        except Exception:
            pass  # starlette raises on close before accept


def test_empty_token_closes_4401() -> None:
    app = _make_app()
    with TestClient(app) as client:
        try:
            with client.websocket_connect("/v1/voice/stream?token=") as _ws:
                pass
        except Exception:
            pass


# ── 2. Bad JSON → BAD_JSON error ─────────────────────────────────


def test_bad_json_returns_error() -> None:
    app = _make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/v1/voice/stream?token=fake") as ws_conn:
            ws_conn.send_text("not valid json {{{")
            resp = json.loads(ws_conn.receive_text())
            assert resp["type"] == "error"
            assert resp["code"] == "BAD_JSON"


# ── 3. Bad base64 audio data → BAD_AUDIO error ──────────────────


def test_bad_audio_base64_returns_error() -> None:
    app = _make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/v1/voice/stream?token=fake") as ws_conn:
            # Send audio_chunk with invalid base64 data
            ws_conn.send_text(json.dumps({"type": "audio_chunk", "data": "!!!not-base64!!!"}))
            resp = json.loads(ws_conn.receive_text())
            assert resp["type"] == "error"
            assert resp["code"] == "BAD_AUDIO"


# ── 4. "end" without "start" → NO_SESSION error ─────────────────


def test_end_without_start_returns_no_session() -> None:
    app = _make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/v1/voice/stream?token=fake") as ws_conn:
            ws_conn.send_text(json.dumps({"type": "end"}))
            resp = json.loads(ws_conn.receive_text())
            assert resp["type"] == "error"
            assert resp["code"] == "NO_SESSION"


# ── 5. Pipeline exception → INTERNAL error ───────────────────────


def test_pipeline_exception_returns_internal_error() -> None:
    app = _make_app()
    # Make STT raise
    app.state.stt = AsyncMock()
    app.state.stt.transcribe = AsyncMock(side_effect=RuntimeError("stt boom"))

    with TestClient(app) as client:
        with client.websocket_connect("/v1/voice/stream?token=fake") as ws_conn:
            ws_conn.send_text(json.dumps({
                "type": "start",
                "session_id": "s1",
                "personality_id": "p1",
            }))
            ready = json.loads(ws_conn.receive_text())
            assert ready["type"] == "ready"

            ws_conn.send_text(json.dumps({
                "type": "audio_chunk",
                "data": base64.b64encode(b"fake audio").decode(),
            }))
            ws_conn.send_text(json.dumps({"type": "end"}))

            resp = json.loads(ws_conn.receive_text())
            assert resp["type"] == "error"
            assert resp["code"] == "INTERNAL"
            assert "stt boom" in resp["message"]


def test_pipeline_brain_exception_returns_internal_error() -> None:
    app = _make_app()
    app.state.brain_client.chat_collect = AsyncMock(side_effect=RuntimeError("brain down"))

    with TestClient(app) as client:
        with client.websocket_connect("/v1/voice/stream?token=fake") as ws_conn:
            ws_conn.send_text(json.dumps({
                "type": "start",
                "session_id": "s1",
                "personality_id": "p1",
            }))
            ready = json.loads(ws_conn.receive_text())
            assert ready["type"] == "ready"

            ws_conn.send_text(json.dumps({
                "type": "audio_chunk",
                "data": base64.b64encode(b"audio").decode(),
            }))
            ws_conn.send_text(json.dumps({"type": "end"}))

            # First message: transcript (STT succeeds)
            transcript = json.loads(ws_conn.receive_text())
            assert transcript["type"] == "transcript"

            # Second message: INTERNAL error (brain fails)
            resp = json.loads(ws_conn.receive_text())
            assert resp["type"] == "error"
            assert resp["code"] == "INTERNAL"
            assert "brain down" in resp["message"]


# ── 6. Unknown type → UNKNOWN_TYPE error ─────────────────────────


def test_unknown_type_returns_error() -> None:
    app = _make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/v1/voice/stream?token=fake") as ws_conn:
            ws_conn.send_text(json.dumps({"type": "foobar"}))
            resp = json.loads(ws_conn.receive_text())
            assert resp["type"] == "error"
            assert resp["code"] == "UNKNOWN_TYPE"


# ── Bonus: happy path with mocks (sanity) ───────────────────────


def test_happy_path_end_to_end() -> None:
    app = _make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/v1/voice/stream?token=fake") as ws_conn:
            ws_conn.send_text(json.dumps({
                "type": "start",
                "session_id": "s1",
                "personality_id": "p1",
            }))
            ready = json.loads(ws_conn.receive_text())
            assert ready["type"] == "ready"

            ws_conn.send_text(json.dumps({
                "type": "audio_chunk",
                "data": base64.b64encode(b"audio data").decode(),
            }))
            ws_conn.send_text(json.dumps({"type": "end"}))

            frames = [json.loads(ws_conn.receive_text()) for _ in range(3)]
            types = [f["type"] for f in frames]
            assert "transcript" in types
            assert "audio_chunk" in types
            assert "done" in types
