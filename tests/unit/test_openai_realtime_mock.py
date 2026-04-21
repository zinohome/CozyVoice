"""OpenAIRealtimeProvider 单测（mock websockets）。"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cozyvoice.providers.realtime.openai_realtime import OpenAIRealtimeProvider


async def test_open_session_sends_session_update() -> None:
    fake_ws = MagicMock()
    fake_ws.send = AsyncMock()
    fake_ws.close = AsyncMock()

    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)) as mock_connect:
        provider = OpenAIRealtimeProvider(api_key="sk-test")
        session = await provider.open_session(
            instructions="你是助手",
            voice="alloy",
            tools=[{"type": "function", "name": "weather"}],
        )
        assert session is not None

    mock_connect.assert_awaited_once()
    sent = json.loads(fake_ws.send.await_args_list[0].args[0])
    assert sent["type"] == "session.update"
    assert sent["session"]["instructions"] == "你是助手"
    assert sent["session"]["voice"] == "alloy"
    assert len(sent["session"]["tools"]) == 1


async def test_send_audio_base64() -> None:
    fake_ws = MagicMock()
    fake_ws.send = AsyncMock()
    fake_ws.close = AsyncMock()

    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)):
        provider = OpenAIRealtimeProvider(api_key="sk-test")
        session = await provider.open_session(instructions="x")
        await session.send_audio(b"\x00\x01\x02\x03")

    payload = json.loads(fake_ws.send.await_args_list[1].args[0])
    assert payload["type"] == "input_audio_buffer.append"
    assert payload["audio"] == "AAECAw=="


async def test_submit_tool_result_triggers_response_create() -> None:
    fake_ws = MagicMock()
    fake_ws.send = AsyncMock()
    fake_ws.close = AsyncMock()

    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)):
        provider = OpenAIRealtimeProvider(api_key="sk-test")
        session = await provider.open_session(instructions="x")
        await session.submit_tool_result("call_abc", '{"temperature": 22}')

    assert fake_ws.send.await_count == 3
    item_event = json.loads(fake_ws.send.await_args_list[1].args[0])
    assert item_event["item"]["call_id"] == "call_abc"
    assert item_event["item"]["output"] == '{"temperature": 22}'
    final = json.loads(fake_ws.send.await_args_list[2].args[0])
    assert final["type"] == "response.create"
