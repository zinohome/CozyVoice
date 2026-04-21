"""Voice Agent 单测（mock Realtime + mock Brain，不连 LiveKit）。"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cozyvoice.realtime_agent import RealtimeCallState, handle_realtime_call


class FakeRTSession:
    """模拟 Realtime 事件流。"""

    def __init__(self, events: list[dict]) -> None:
        self._events = events
        self.sent_audio: list[bytes] = []
        self.tool_results: list[tuple[str, str]] = []
        self.closed = False

    async def send_audio(self, pcm: bytes) -> None:
        self.sent_audio.append(pcm)

    async def receive_events(self):
        for e in self._events:
            yield e

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        self.tool_results.append((call_id, output))

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_rt_events():
    return [
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "查上海天气"},
        {
            "type": "response.function_call_arguments.done",
            "call_id": "call_1",
            "name": "weather",
            "arguments": json.dumps({"city": "上海"}),
        },
        {"type": "response.audio_transcript.delta", "delta": "上海"},
        {"type": "response.audio_transcript.delta", "delta": "天气 22 度"},
        {"type": "response.audio_transcript.done"},
        {"type": "response.done"},
    ]


async def test_handle_realtime_call_tool_proxy_and_summary(fake_rt_events) -> None:
    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await audio_in.put(None)  # sentinel 让 pump_in 立刻结束

    state = RealtimeCallState(jwt="t", session_id="s", personality_id="p")

    fake_session = FakeRTSession(fake_rt_events)

    mock_provider = MagicMock()
    mock_provider.open_session = AsyncMock(return_value=fake_session)

    mock_brain = MagicMock()
    mock_brain.startup = AsyncMock()
    mock_brain.shutdown = AsyncMock()
    mock_brain.fetch_voice_context = AsyncMock(return_value={
        "system_prompt": "s",
        "voice_id": "alloy",
        "allowed_tools": [{"type": "function", "name": "weather"}],
    })
    mock_brain.tool_proxy = AsyncMock(return_value={
        "status": "success",
        "result": {"temperature": 22},
    })
    mock_brain.voice_summary = AsyncMock(return_value={"messages_saved": 2, "tool_calls_saved": 1})

    with patch("cozyvoice.realtime_agent.BrainClient", return_value=mock_brain), \
         patch("cozyvoice.realtime_agent.OpenAIRealtimeProvider", return_value=mock_provider):
        await handle_realtime_call(
            audio_in=audio_in,
            audio_out=audio_out,
            state=state,
            brain_url="http://brain",
            openai_api_key="sk-test",
        )

    mock_brain.tool_proxy.assert_awaited_once()
    args = mock_brain.tool_proxy.await_args.kwargs
    assert args["tool_name"] == "weather"
    assert args["arguments"] == {"city": "上海"}

    assert len(fake_session.tool_results) == 1
    assert fake_session.tool_results[0][0] == "call_1"

    mock_brain.voice_summary.assert_awaited_once()
    summ = mock_brain.voice_summary.await_args.kwargs
    assert len(summ["turns"]) >= 1
    assert len(summ["tool_calls"]) == 1
