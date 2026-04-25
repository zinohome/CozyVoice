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


# ---------------------------------------------------------------------------
# Helper: run handle_realtime_call with given events / brain overrides
# ---------------------------------------------------------------------------

async def _run_call(events, brain_overrides=None):
    """Utility to run handle_realtime_call with FakeRTSession and mocked Brain."""
    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await audio_in.put(None)  # sentinel → pump_in exits immediately

    state = RealtimeCallState(jwt="tok", session_id="sid", personality_id="pid")
    fake_session = FakeRTSession(events)

    mock_provider = MagicMock()
    mock_provider.open_session = AsyncMock(return_value=fake_session)

    mock_brain = MagicMock()
    mock_brain.startup = AsyncMock()
    mock_brain.shutdown = AsyncMock()
    mock_brain.fetch_voice_context = AsyncMock(return_value={
        "system_prompt": "你是助手",
        "voice_id": "alloy",
        "allowed_tools": [],
    })
    mock_brain.tool_proxy = AsyncMock(return_value={"status": "success", "result": {"ok": True}})
    mock_brain.voice_summary = AsyncMock(return_value={"messages_saved": 0, "tool_calls_saved": 0})

    if brain_overrides:
        for k, v in brain_overrides.items():
            setattr(mock_brain, k, v)

    with patch("cozyvoice.realtime_agent.BrainClient", return_value=mock_brain), \
         patch("cozyvoice.realtime_agent.OpenAIRealtimeProvider", return_value=mock_provider):
        await handle_realtime_call(
            audio_in=audio_in,
            audio_out=audio_out,
            state=state,
            brain_url="http://brain",
            openai_api_key="sk-test",
        )

    return state, fake_session, mock_brain, audio_out


# ---------------------------------------------------------------------------
# 1) fetch_voice_context 失败 → fallback 默认值
# ---------------------------------------------------------------------------

async def test_fetch_voice_context_failure_fallback() -> None:
    """When fetch_voice_context raises, should use fallback defaults and still open session."""
    events = [{"type": "response.done"}]

    mock_brain_overrides = {
        "fetch_voice_context": AsyncMock(side_effect=RuntimeError("network error")),
    }
    state, fake_session, mock_brain, _ = await _run_call(events, mock_brain_overrides)

    # Brain still called voice_summary (call didn't crash)
    mock_brain.voice_summary.assert_awaited_once()
    # Session was opened and closed normally
    assert fake_session.closed


# ---------------------------------------------------------------------------
# 2) profile_context + memory_summary appended to instructions
# ---------------------------------------------------------------------------

async def test_profile_and_memory_appended_to_instructions() -> None:
    """profile_context and memory_summary from voice context should be included in instructions."""
    events = [{"type": "response.done"}]

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await audio_in.put(None)

    state = RealtimeCallState(jwt="t", session_id="s", personality_id="p")
    fake_session = FakeRTSession(events)

    mock_provider = MagicMock()
    mock_provider.open_session = AsyncMock(return_value=fake_session)

    mock_brain = MagicMock()
    mock_brain.startup = AsyncMock()
    mock_brain.shutdown = AsyncMock()
    mock_brain.fetch_voice_context = AsyncMock(return_value={
        "system_prompt": "Base prompt.",
        "voice_id": "shimmer",
        "allowed_tools": [],
        "profile_context": "User likes cats.",
        "memory_summary": "Previously talked about weather.",
    })
    mock_brain.voice_summary = AsyncMock(return_value={})

    with patch("cozyvoice.realtime_agent.BrainClient", return_value=mock_brain), \
         patch("cozyvoice.realtime_agent.OpenAIRealtimeProvider", return_value=mock_provider):
        await handle_realtime_call(
            audio_in=audio_in, audio_out=audio_out, state=state,
            brain_url="http://brain", openai_api_key="sk-test",
        )

    # open_session should have been called with instructions containing profile + memory
    call_kwargs = mock_provider.open_session.await_args.kwargs
    instr = call_kwargs["instructions"]
    assert "User likes cats." in instr
    assert "Previously talked about weather." in instr
    assert call_kwargs["voice"] == "shimmer"


# ---------------------------------------------------------------------------
# 3) voice_summary failure → only logs, no crash
# ---------------------------------------------------------------------------

async def test_voice_summary_failure_no_crash() -> None:
    events = [{"type": "response.done"}]
    mock_brain_overrides = {
        "voice_summary": AsyncMock(side_effect=RuntimeError("brain down")),
    }
    # Should not raise
    state, fake_session, mock_brain, _ = await _run_call(events, mock_brain_overrides)
    assert fake_session.closed


# ---------------------------------------------------------------------------
# 4) response.audio.delta → base64 decoded into audio_out
# ---------------------------------------------------------------------------

async def test_audio_delta_decoded_to_audio_out() -> None:
    import base64
    raw_audio = b"\x00\x01\x02\x03\xff"
    b64 = base64.b64encode(raw_audio).decode()
    events = [
        {"type": "response.audio.delta", "delta": b64},
        {"type": "response.output_audio.delta", "delta": b64},  # GA alias
        {"type": "response.done"},
    ]
    state, _, _, audio_out = await _run_call(events)

    chunks = []
    while not audio_out.empty():
        chunks.append(audio_out.get_nowait())
    assert len(chunks) == 2
    assert chunks[0] == raw_audio
    assert chunks[1] == raw_audio


# ---------------------------------------------------------------------------
# 5) function_call_arguments.delta → incremental args assembly
# ---------------------------------------------------------------------------

async def test_function_call_arguments_delta_incremental() -> None:
    """Delta events should accumulate args_buf, used when .done fires."""
    events = [
        {"type": "response.function_call_arguments.delta", "call_id": "c1", "name": "search", "delta": '{"q'},
        {"type": "response.function_call_arguments.delta", "call_id": "c1", "delta": '": "hello"}'},
        {
            "type": "response.function_call_arguments.done",
            "call_id": "c1",
            "name": "search",
            "arguments": None,  # force fallback to args_buf
        },
        {"type": "response.done"},
    ]
    state, fake_session, mock_brain, _ = await _run_call(events)

    mock_brain.tool_proxy.assert_awaited_once()
    call_kwargs = mock_brain.tool_proxy.await_args.kwargs
    assert call_kwargs["tool_name"] == "search"
    assert call_kwargs["arguments"] == {"q": "hello"}


# ---------------------------------------------------------------------------
# 6) json.loads failure on arguments → {"raw": full_args}
# ---------------------------------------------------------------------------

async def test_invalid_json_args_fallback_to_raw() -> None:
    events = [
        {
            "type": "response.function_call_arguments.done",
            "call_id": "c2",
            "name": "broken_tool",
            "arguments": "not-valid-json{{{",
        },
        {"type": "response.done"},
    ]
    state, _, mock_brain, _ = await _run_call(events)

    mock_brain.tool_proxy.assert_awaited_once()
    call_kwargs = mock_brain.tool_proxy.await_args.kwargs
    assert call_kwargs["arguments"] == {"raw": "not-valid-json{{{"}


# ---------------------------------------------------------------------------
# 7) tool_proxy failure → error recorded + submit_tool_result with error
# ---------------------------------------------------------------------------

async def test_tool_proxy_failure_records_error() -> None:
    events = [
        {
            "type": "response.function_call_arguments.done",
            "call_id": "c3",
            "name": "fail_tool",
            "arguments": "{}",
        },
        {"type": "response.done"},
    ]
    mock_brain_overrides = {
        "tool_proxy": AsyncMock(side_effect=RuntimeError("proxy timeout")),
    }
    state, fake_session, _, _ = await _run_call(events, mock_brain_overrides)

    # tool_calls should have an error entry
    assert len(state.tool_calls) == 1
    assert "error" in state.tool_calls[0]["result"]
    assert "proxy timeout" in state.tool_calls[0]["result"]["error"]

    # submit_tool_result should have been called with error payload
    assert len(fake_session.tool_results) == 1
    error_payload = json.loads(fake_session.tool_results[0][1])
    assert error_payload == {"error": "tool_proxy failed"}


# ---------------------------------------------------------------------------
# 8) error event type → no crash (just logged)
# ---------------------------------------------------------------------------

async def test_error_event_no_crash() -> None:
    events = [
        {"type": "error", "error": {"message": "server error", "code": 500}},
        {"type": "response.done"},
    ]
    state, fake_session, _, _ = await _run_call(events)
    assert fake_session.closed
