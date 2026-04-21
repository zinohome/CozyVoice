"""REST /v1/voice/chat 集成测试（mock Brain + Mock STT/TTS）。"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.integration


async def test_voice_chat_roundtrip(monkeypatch) -> None:
    from cozyvoice.main import create_app
    from cozyvoice.providers.stt.mock import MockSTT
    from cozyvoice.providers.tts.mock import MockTTS

    app = create_app()
    async with app.router.lifespan_context(app):
        app.state.stt = MockSTT(canned_text="查上海天气")
        app.state.tts = MockTTS()
        app.state.tts_config = {"default_voice_id": "mock", "default_format": "wav"}

        async def fake_collect(**kwargs):
            assert kwargs["message"] == "查上海天气"
            return "上海今天多云 22°C"
        app.state.brain_client.chat_collect = fake_collect

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            files = {"audio": ("hi.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
            data = {
                "session_id": "00000000-0000-0000-0000-000000000000",
                "personality_id": "00000000-0000-0000-0000-000000000001",
            }
            r = await c.post(
                "/v1/voice/chat", files=files, data=data,
                headers={"Authorization": "Bearer fake-jwt"},
            )
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("audio/")
            assert r.content.startswith(b"RIFF")
            assert "X-Brain-Transcript-In" in r.headers
