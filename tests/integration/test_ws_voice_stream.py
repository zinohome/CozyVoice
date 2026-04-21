"""WS /v1/voice/stream 集成测试。"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_ws_roundtrip() -> None:
    from cozyvoice.main import create_app
    from cozyvoice.providers.stt.mock import MockSTT
    from cozyvoice.providers.tts.mock import MockTTS

    app = create_app()

    # TestClient 会触发 lifespan；startup 完成后我们替换掉 state 里的 providers
    with TestClient(app) as client:
        app.state.stt = MockSTT(canned_text="算一下")
        app.state.tts = MockTTS()
        app.state.tts_config = {"default_voice_id": "mock", "default_format": "wav"}

        async def fake_collect(**kwargs):
            return "好的"

        app.state.brain_client.chat_collect = fake_collect

        with client.websocket_connect("/v1/voice/stream?token=fake") as ws:
            ws.send_text(json.dumps({
                "type": "start",
                "session_id": "00000000-0000-0000-0000-000000000000",
                "personality_id": "00000000-0000-0000-0000-000000000001",
            }))
            ready = json.loads(ws.receive_text())
            assert ready["type"] == "ready"

            ws.send_text(json.dumps({"type": "audio_chunk", "data": base64.b64encode(b"xxx").decode()}))
            ws.send_text(json.dumps({"type": "end"}))

            frames = [json.loads(ws.receive_text()) for _ in range(3)]
            types = [f["type"] for f in frames]
            assert "transcript" in types
            assert "audio_chunk" in types
            assert "done" in types
