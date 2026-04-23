"""OpenAIRealtimeProvider 单测（mock websockets）。"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cozyvoice.providers.realtime.openai_realtime import OpenAIRealtimeProvider


def _make_fake_ws():
    fake_ws = MagicMock()
    fake_ws.send = AsyncMock()
    fake_ws.close = AsyncMock()
    # open_session 会 await 2 次 recv（welcome + session.updated ack）
    fake_ws.recv = AsyncMock(side_effect=[
        json.dumps({"type": "session.created", "session": {"id": "sess_fake"}}),
        json.dumps({"type": "session.updated", "session": {}}),
    ])
    return fake_ws


async def test_open_session_sends_session_update_ga_format() -> None:
    """GA 格式：session.type='realtime'，audio.input/output 嵌套。"""
    fake_ws = _make_fake_ws()
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
    # subprotocols 应只含 'realtime'（去 beta 标记）
    kwargs = mock_connect.await_args.kwargs
    assert kwargs.get("subprotocols") == ["realtime"]
    # session.update 走 GA 嵌套结构
    sent = json.loads(fake_ws.send.await_args_list[0].args[0])
    assert sent["type"] == "session.update"
    sess = sent["session"]
    assert sess["type"] == "realtime"
    assert sess["instructions"] == "你是助手"
    assert sess["audio"]["output"]["voice"] == "alloy"
    assert sess["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert len(sess["tools"]) == 1


async def test_send_audio_base64() -> None:
    fake_ws = _make_fake_ws()
    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)):
        provider = OpenAIRealtimeProvider(api_key="sk-test")
        session = await provider.open_session(instructions="x")
        await session.send_audio(b"\x00\x01\x02\x03")

    payload = json.loads(fake_ws.send.await_args_list[1].args[0])
    assert payload["type"] == "input_audio_buffer.append"
    assert payload["audio"] == "AAECAw=="


async def test_submit_tool_result_triggers_response_create() -> None:
    fake_ws = _make_fake_ws()
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
