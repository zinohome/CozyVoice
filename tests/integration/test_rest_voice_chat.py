"""REST /v1/voice/chat SSE integration test (mock Brain + Mock STT/TTS)."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.integration


async def test_voice_chat_sse_roundtrip(monkeypatch) -> None:
    from cozyvoice.providers.stt.mock import MockSTT
    from cozyvoice.providers.tts.mock import MockTTS
    import cozyvoice.main as main_mod

    monkeypatch.setattr(main_mod, "_build_stt", lambda cfg: MockSTT(canned_text="查上海天气"))
    monkeypatch.setattr(main_mod, "_build_tts", lambda cfg: MockTTS())

    from cozyvoice.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        app.state.stt = MockSTT(canned_text="查上海天气")
        app.state.tts = MockTTS()
        app.state.tts_config = {"default_voice_id": "mock", "default_format": "wav"}

        async def fake_stream(**kwargs):
            assert kwargs["message"] == "查上海天气"
            for chunk in ["上海今天", "多云 22°C"]:
                yield chunk

        app.state.brain_client.chat_stream = fake_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            files = {"audio": ("hi.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
            data = {
                "session_id": "00000000-0000-0000-0000-000000000000",
                "personality_id": "00000000-0000-0000-0000-000000000001",
            }
            r = await c.post(
                "/v1/voice/chat?tts=true", files=files, data=data,
                headers={"Authorization": "Bearer fake-jwt"},
            )
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]

            events = []
            for line in r.text.split("\n"):
                if line.startswith("event: "):
                    events.append(line[7:].strip())
            assert "stt" in events
            assert "reply_done" in events
            assert "tts_audio" in events
