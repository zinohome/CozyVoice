"""OpenAIRealtimeProvider 单测（mock websockets）。"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets

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


# ---------------------------------------------------------------------------
# 4) send_audio when _closed=True → no-op
# ---------------------------------------------------------------------------

async def test_send_audio_when_closed_is_noop() -> None:
    fake_ws = _make_fake_ws()
    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)):
        provider = OpenAIRealtimeProvider(api_key="sk-test")
        session = await provider.open_session(instructions="x")
        # Record how many sends happened during open_session (1 for session.update)
        sends_before = fake_ws.send.await_count
        session._closed = True
        await session.send_audio(b"\xff\xff")
        # No additional send should have happened
        assert fake_ws.send.await_count == sends_before


# ---------------------------------------------------------------------------
# 5) receive_events — normal yield + ConnectionClosed handling
# ---------------------------------------------------------------------------

async def test_receive_events_yields_parsed_json() -> None:
    """receive_events should yield parsed dicts from raw JSON ws frames."""
    events_raw = [
        json.dumps({"type": "response.audio.delta", "delta": "abc"}),
        json.dumps({"type": "response.done"}),
    ]

    class FakeIterableWS:
        send = AsyncMock()
        close = AsyncMock()
        recv = AsyncMock(side_effect=[
            json.dumps({"type": "session.created", "session": {"id": "s"}}),
            json.dumps({"type": "session.updated", "session": {}}),
        ])

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not events_raw:
                raise StopAsyncIteration
            return events_raw.pop(0)

    fake_ws = FakeIterableWS()
    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)):
        provider = OpenAIRealtimeProvider(api_key="sk-test")
        session = await provider.open_session(instructions="x")

    collected = []
    async for event in session.receive_events():
        collected.append(event)

    assert len(collected) == 2
    assert collected[0]["type"] == "response.audio.delta"
    assert collected[1]["type"] == "response.done"


async def test_receive_events_connection_closed_graceful() -> None:
    """When ws raises ConnectionClosed, receive_events should exit gracefully."""

    class ClosedWS:
        send = AsyncMock()
        close = AsyncMock()
        recv = AsyncMock(side_effect=[
            json.dumps({"type": "session.created", "session": {"id": "s"}}),
            json.dumps({"type": "session.updated", "session": {}}),
        ])

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise websockets.ConnectionClosed(None, None)

    fake_ws = ClosedWS()
    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)):
        provider = OpenAIRealtimeProvider(api_key="sk-test")
        session = await provider.open_session(instructions="x")

    collected = []
    async for event in session.receive_events():
        collected.append(event)

    assert collected == []


# ---------------------------------------------------------------------------
# 6) submit_tool_result when _closed=True → no-op
# ---------------------------------------------------------------------------

async def test_submit_tool_result_when_closed_is_noop() -> None:
    fake_ws = _make_fake_ws()
    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)):
        provider = OpenAIRealtimeProvider(api_key="sk-test")
        session = await provider.open_session(instructions="x")
        sends_before = fake_ws.send.await_count
        session._closed = True
        await session.submit_tool_result("call_99", '{"result": 1}')
        assert fake_ws.send.await_count == sends_before


# ---------------------------------------------------------------------------
# 7) close() sets _closed=True and calls ws.close()
# ---------------------------------------------------------------------------

async def test_close_sets_flag_and_closes_ws() -> None:
    fake_ws = _make_fake_ws()
    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)):
        provider = OpenAIRealtimeProvider(api_key="sk-test")
        session = await provider.open_session(instructions="x")
        assert not session._closed
        await session.close()
        assert session._closed
        fake_ws.close.assert_awaited_once()


async def test_close_exception_suppressed() -> None:
    fake_ws = _make_fake_ws()
    fake_ws.close = AsyncMock(side_effect=OSError("socket gone"))
    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)):
        provider = OpenAIRealtimeProvider(api_key="sk-test")
        session = await provider.open_session(instructions="x")
        # Should not raise
        await session.close()
        assert session._closed


# ---------------------------------------------------------------------------
# 8) base_url: http→ws, https→wss, auto-append /realtime
# ---------------------------------------------------------------------------

async def test_base_url_http_to_ws_conversion() -> None:
    fake_ws = _make_fake_ws()
    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)) as mock_connect:
        provider = OpenAIRealtimeProvider(api_key="sk-test", base_url="http://proxy.local/v1")
        await provider.open_session(instructions="x")

    url_called = mock_connect.await_args.args[0]
    assert url_called.startswith("ws://proxy.local/v1/realtime?model=")


async def test_base_url_https_to_wss_conversion() -> None:
    fake_ws = _make_fake_ws()
    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)) as mock_connect:
        provider = OpenAIRealtimeProvider(api_key="sk-test", base_url="https://proxy.local/v1")
        await provider.open_session(instructions="x")

    url_called = mock_connect.await_args.args[0]
    assert url_called.startswith("wss://proxy.local/v1/realtime?model=")


async def test_base_url_already_has_realtime_suffix() -> None:
    fake_ws = _make_fake_ws()
    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)) as mock_connect:
        provider = OpenAIRealtimeProvider(api_key="sk-test", base_url="https://proxy.local/v1/realtime")
        await provider.open_session(instructions="x")

    url_called = mock_connect.await_args.args[0]
    # Should NOT double-append /realtime
    assert "/realtime/realtime" not in url_called
    assert url_called.startswith("wss://proxy.local/v1/realtime?model=")


# ---------------------------------------------------------------------------
# 9) open_session welcome frame timeout → proceeds to session.update
# ---------------------------------------------------------------------------

async def test_open_session_welcome_timeout_proceeds() -> None:
    """If the welcome frame times out, open_session should still proceed."""
    fake_ws = MagicMock()
    fake_ws.send = AsyncMock()
    fake_ws.close = AsyncMock()
    fake_ws.recv = AsyncMock(side_effect=[
        asyncio.TimeoutError(),  # welcome timeout
        json.dumps({"type": "session.updated", "session": {}}),  # ack
    ])

    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)):
        provider = OpenAIRealtimeProvider(api_key="sk-test")
        session = await provider.open_session(instructions="x")
        assert session is not None

    sent = json.loads(fake_ws.send.await_args_list[0].args[0])
    assert sent["type"] == "session.update"


# ---------------------------------------------------------------------------
# 10) session.update ack timeout → still returns session
# ---------------------------------------------------------------------------

async def test_open_session_ack_timeout_proceeds() -> None:
    """If session.update ack times out, open_session should still return."""
    fake_ws = MagicMock()
    fake_ws.send = AsyncMock()
    fake_ws.close = AsyncMock()
    fake_ws.recv = AsyncMock(side_effect=[
        json.dumps({"type": "session.created", "session": {"id": "s1"}}),  # welcome ok
        asyncio.TimeoutError(),  # ack timeout
    ])

    with patch("cozyvoice.providers.realtime.openai_realtime.websockets.connect",
               new=AsyncMock(return_value=fake_ws)):
        provider = OpenAIRealtimeProvider(api_key="sk-test")
        session = await provider.open_session(instructions="hello")
        assert session is not None
