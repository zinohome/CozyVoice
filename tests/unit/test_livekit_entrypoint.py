"""LiveKit entrypoint unit tests (mock LiveKit primitives)."""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cozyvoice.livekit_entrypoint import (
    _forward_track_to_queue,
    _handle_participant,
    _main,
    _parse_participant_metadata,
    _pump_queue_to_source,
    entrypoint,
)


def test_parse_valid_metadata() -> None:
    raw = json.dumps({"brain_jwt": "tok", "session_id": "s1", "personality_id": "p1"})
    result = _parse_participant_metadata(raw)
    assert result["brain_jwt"] == "tok"
    assert result["session_id"] == "s1"


def test_parse_empty_metadata() -> None:
    assert _parse_participant_metadata(None) == {}
    assert _parse_participant_metadata("") == {}


def test_parse_invalid_json() -> None:
    result = _parse_participant_metadata("not-json{{{")
    assert result == {}


def test_parse_non_dict_json() -> None:
    result = _parse_participant_metadata('"just a string"')
    assert result == {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_participant(identity: str = "user1", metadata: str | None = None) -> MagicMock:
    """Create a mock RemoteParticipant."""
    p = MagicMock()
    p.identity = identity
    p.metadata = metadata
    p.track_publications = {}
    return p


def _make_ctx(remote_participants: dict | None = None) -> MagicMock:
    """Create a mock JobContext with room, local_participant, etc."""
    ctx = MagicMock()
    ctx.connect = AsyncMock()
    ctx.room = MagicMock()
    ctx.room.name = "test-room"
    ctx.room.local_participant = MagicMock()
    ctx.room.local_participant.publish_track = AsyncMock(
        return_value=MagicMock(sid="TR_AGENT")
    )
    ctx.room.remote_participants = remote_participants or {}
    # .on() stores handlers so tests can invoke them
    ctx.room._handlers: dict[str, list] = {}

    def _on(event: str, handler):
        ctx.room._handlers.setdefault(event, []).append(handler)

    ctx.room.on = _on
    ctx.add_shutdown_callback = MagicMock()
    return ctx


def _valid_metadata() -> str:
    return json.dumps({
        "brain_jwt": "tok",
        "session_id": "s1",
        "personality_id": "p1",
    })


# ---------------------------------------------------------------------------
# _forward_track_to_queue tests
# ---------------------------------------------------------------------------

class _FakeAudioStream:
    """Async-iterable that yields mock AudioFrameEvents, then stops."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = frames

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for data in self._frames:
            event = MagicMock()
            event.frame.data = data
            yield event

    async def aclose(self) -> None:
        pass


class _ErrorAudioStream(_FakeAudioStream):
    """Yields one frame then raises RuntimeError."""

    async def _iter(self):
        for data in self._frames:
            event = MagicMock()
            event.frame.data = data
            yield event
        raise RuntimeError("boom")


async def test_forward_track_normal_flow() -> None:
    """Normal iteration puts bytes then sentinel None."""
    q: asyncio.Queue = asyncio.Queue()
    track = MagicMock()
    frames = [b"\x01\x02", b"\x03\x04"]

    with patch("cozyvoice.livekit_entrypoint.rtc.AudioStream", return_value=_FakeAudioStream(frames)):
        await _forward_track_to_queue(track, q)

    items = []
    while not q.empty():
        items.append(q.get_nowait())
    # Should have the two frame bytes + None sentinel
    assert items == [b"\x01\x02", b"\x03\x04", None]


async def test_forward_track_exception_puts_none_and_closes() -> None:
    """On exception, put None sentinel and aclose the stream."""
    q: asyncio.Queue = asyncio.Queue()
    track = MagicMock()
    stream = _ErrorAudioStream([b"\xaa"])
    stream.aclose = AsyncMock()

    with patch("cozyvoice.livekit_entrypoint.rtc.AudioStream", return_value=stream):
        await _forward_track_to_queue(track, q)

    items = []
    while not q.empty():
        items.append(q.get_nowait())
    assert items == [b"\xaa", None]
    stream.aclose.assert_awaited_once()


async def test_forward_track_cancelled_propagates() -> None:
    """CancelledError propagates out (not swallowed)."""

    class _CancelStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise asyncio.CancelledError()

        async def aclose(self):
            pass

    q: asyncio.Queue = asyncio.Queue()
    stream = _CancelStream()
    stream.aclose = AsyncMock()

    with patch("cozyvoice.livekit_entrypoint.rtc.AudioStream", return_value=stream):
        with pytest.raises(asyncio.CancelledError):
            await _forward_track_to_queue(MagicMock(), q)

    # sentinel should still be placed in finally
    assert q.get_nowait() is None


# ---------------------------------------------------------------------------
# _pump_queue_to_source tests
# ---------------------------------------------------------------------------

async def test_pump_queue_none_sentinel_exits() -> None:
    """Receiving None sentinel exits the loop cleanly."""
    q: asyncio.Queue = asyncio.Queue()
    source = MagicMock()
    source.capture_frame = AsyncMock()

    await q.put(None)
    await _pump_queue_to_source(q, source)

    source.capture_frame.assert_not_awaited()


async def test_pump_queue_empty_bytes_skipped() -> None:
    """Empty bytes (b'') are skipped, then None exits."""
    q: asyncio.Queue = asyncio.Queue()
    source = MagicMock()
    source.capture_frame = AsyncMock()

    await q.put(b"")
    await q.put(None)

    with patch("cozyvoice.livekit_entrypoint.rtc.AudioFrame") as MockFrame:
        await _pump_queue_to_source(q, source)

    source.capture_frame.assert_not_awaited()


async def test_pump_queue_normal_pcm() -> None:
    """Normal PCM bytes get converted to AudioFrame and captured."""
    q: asyncio.Queue = asyncio.Queue()
    source = MagicMock()
    source.capture_frame = AsyncMock()

    # 4 bytes = 2 int16 samples
    pcm = b"\x01\x00\x02\x00"
    await q.put(pcm)
    await q.put(None)

    mock_frame = MagicMock()
    with patch("cozyvoice.livekit_entrypoint.rtc.AudioFrame", return_value=mock_frame) as MockFrame:
        await _pump_queue_to_source(q, source)

    MockFrame.assert_called_once()
    call_kwargs = MockFrame.call_args
    assert call_kwargs.kwargs["sample_rate"] == 24000
    assert call_kwargs.kwargs["num_channels"] == 1
    assert call_kwargs.kwargs["samples_per_channel"] == 2
    source.capture_frame.assert_awaited_once_with(mock_frame)


async def test_pump_queue_cancelled_propagates() -> None:
    """CancelledError propagates."""
    q: asyncio.Queue = asyncio.Queue()
    source = MagicMock()

    # Make queue.get raise CancelledError
    original_get = q.get
    q.get = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await _pump_queue_to_source(q, source)


async def test_pump_queue_multichannel() -> None:
    """When num_channels > 1, samples_per_channel is divided."""
    q: asyncio.Queue = asyncio.Queue()
    source = MagicMock()
    source.capture_frame = AsyncMock()

    # 8 bytes = 4 int16 samples; 2 channels → 2 samples_per_channel
    pcm = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    await q.put(pcm)
    await q.put(None)

    mock_frame = MagicMock()
    with patch("cozyvoice.livekit_entrypoint.rtc.AudioFrame", return_value=mock_frame) as MockFrame:
        await _pump_queue_to_source(q, source, num_channels=2)

    assert MockFrame.call_args.kwargs["samples_per_channel"] == 2
    assert MockFrame.call_args.kwargs["num_channels"] == 2


# ---------------------------------------------------------------------------
# _handle_participant tests
# ---------------------------------------------------------------------------

async def test_handle_participant_missing_metadata_skips() -> None:
    """If metadata lacks required fields, return immediately."""
    ctx = _make_ctx()
    participant = _make_participant(metadata=json.dumps({"brain_jwt": "tok"}))
    # missing session_id and personality_id

    await _handle_participant(ctx, participant)

    # Should not publish any track
    ctx.room.local_participant.publish_track.assert_not_awaited()


async def test_handle_participant_no_metadata_skips() -> None:
    """If metadata is None, return immediately."""
    ctx = _make_ctx()
    participant = _make_participant(metadata=None)

    await _handle_participant(ctx, participant)

    ctx.room.local_participant.publish_track.assert_not_awaited()


@patch.dict(os.environ, {"COZYVOICE_REALTIME_MODE": "cozy_pipeline"})
@patch("cozyvoice.livekit_entrypoint.run_cozy_pipeline", new_callable=AsyncMock)
async def test_handle_participant_cozy_pipeline_success(mock_pipeline: AsyncMock) -> None:
    """cozy_pipeline mode succeeds → return without entering openai path."""
    ctx = _make_ctx()
    participant = _make_participant(metadata=_valid_metadata())

    await _handle_participant(ctx, participant)

    mock_pipeline.assert_awaited_once()
    # Should NOT publish track (openai path)
    ctx.room.local_participant.publish_track.assert_not_awaited()


@patch.dict(os.environ, {"COZYVOICE_REALTIME_MODE": "cozy_pipeline"})
@patch("cozyvoice.livekit_entrypoint.handle_realtime_call", new_callable=AsyncMock)
@patch("cozyvoice.livekit_entrypoint.run_cozy_pipeline", new_callable=AsyncMock, side_effect=NotImplementedError("nope"))
async def test_handle_participant_cozy_pipeline_not_implemented_fallback(
    mock_pipeline: AsyncMock,
    mock_realtime: AsyncMock,
) -> None:
    """NotImplementedError from cozy_pipeline falls back to openai mode."""
    ctx = _make_ctx()
    participant = _make_participant(metadata=_valid_metadata())

    # handle_realtime_call completes immediately so the await-wait finishes
    mock_realtime.return_value = None

    with patch("cozyvoice.livekit_entrypoint.rtc.AudioSource"):
        with patch("cozyvoice.livekit_entrypoint.rtc.LocalAudioTrack") as MockTrack:
            MockTrack.create_audio_track.return_value = MagicMock()
            await _handle_participant(ctx, participant)

    mock_pipeline.assert_awaited_once()
    # Fell through to openai → published track
    ctx.room.local_participant.publish_track.assert_awaited_once()


@patch.dict(os.environ, {"COZYVOICE_REALTIME_MODE": "cozy_pipeline"})
@patch("cozyvoice.livekit_entrypoint.handle_realtime_call", new_callable=AsyncMock)
@patch("cozyvoice.livekit_entrypoint.run_cozy_pipeline", new_callable=AsyncMock, side_effect=RuntimeError("bad env"))
async def test_handle_participant_cozy_pipeline_runtime_error_fallback(
    mock_pipeline: AsyncMock,
    mock_realtime: AsyncMock,
) -> None:
    """RuntimeError from cozy_pipeline also falls back to openai."""
    ctx = _make_ctx()
    participant = _make_participant(metadata=_valid_metadata())
    mock_realtime.return_value = None

    with patch("cozyvoice.livekit_entrypoint.rtc.AudioSource"):
        with patch("cozyvoice.livekit_entrypoint.rtc.LocalAudioTrack") as MockTrack:
            MockTrack.create_audio_track.return_value = MagicMock()
            await _handle_participant(ctx, participant)

    mock_pipeline.assert_awaited_once()
    ctx.room.local_participant.publish_track.assert_awaited_once()


@patch.dict(os.environ, {"COZYVOICE_REALTIME_MODE": "openai", "OPENAI_API_KEY": "sk-test"})
@patch("cozyvoice.livekit_entrypoint.handle_realtime_call", new_callable=AsyncMock)
async def test_handle_participant_openai_mode_full_flow(mock_realtime: AsyncMock) -> None:
    """OpenAI mode: publish track, create call_task, cleanup on completion."""
    ctx = _make_ctx()
    participant = _make_participant(metadata=_valid_metadata())
    mock_realtime.return_value = None

    mock_source = MagicMock()
    mock_source.aclose = AsyncMock()

    with patch("cozyvoice.livekit_entrypoint.rtc.AudioSource", return_value=mock_source):
        with patch("cozyvoice.livekit_entrypoint.rtc.LocalAudioTrack") as MockTrack:
            MockTrack.create_audio_track.return_value = MagicMock()
            await _handle_participant(ctx, participant)

    ctx.room.local_participant.publish_track.assert_awaited_once()
    mock_realtime.assert_awaited_once()

    # Verify call kwargs
    call_kwargs = mock_realtime.call_args.kwargs
    assert call_kwargs["state"].jwt == "tok"
    assert call_kwargs["state"].session_id == "s1"
    assert call_kwargs["openai_api_key"] == "sk-test"


@patch.dict(os.environ, {"COZYVOICE_REALTIME_MODE": "openai", "OPENAI_API_KEY": "sk-test"})
@patch("cozyvoice.livekit_entrypoint.handle_realtime_call", new_callable=AsyncMock)
async def test_handle_participant_existing_published_tracks(mock_realtime: AsyncMock) -> None:
    """If participant already has published audio tracks, forwarder is created."""
    ctx = _make_ctx()
    participant = _make_participant(metadata=_valid_metadata())
    mock_realtime.return_value = None

    # Simulate already published audio track
    mock_pub = MagicMock()
    mock_pub.kind = MagicMock()  # Will be compared to rtc.TrackKind.KIND_AUDIO
    mock_pub.track = MagicMock()
    participant.track_publications = {"TR1": mock_pub}

    mock_source = MagicMock()
    mock_source.aclose = AsyncMock()

    with patch("cozyvoice.livekit_entrypoint.rtc.AudioSource", return_value=mock_source):
        with patch("cozyvoice.livekit_entrypoint.rtc.LocalAudioTrack") as MockTrack:
            MockTrack.create_audio_track.return_value = MagicMock()
            # Make pub.kind match TrackKind.KIND_AUDIO
            with patch("cozyvoice.livekit_entrypoint.rtc.TrackKind") as MockKind:
                mock_pub.kind = MockKind.KIND_AUDIO
                await _handle_participant(ctx, participant)

    mock_realtime.assert_awaited_once()


# ---------------------------------------------------------------------------
# entrypoint tests
# ---------------------------------------------------------------------------

async def test_entrypoint_connects_and_registers_handlers() -> None:
    """entrypoint connects to room and registers participant_connected handler."""
    ctx = _make_ctx()

    with patch("cozyvoice.livekit_entrypoint._handle_participant", new_callable=AsyncMock):
        await entrypoint(ctx)

    ctx.connect.assert_awaited_once()
    assert "participant_connected" in ctx.room._handlers
    ctx.add_shutdown_callback.assert_called_once()


async def test_entrypoint_handles_existing_participants() -> None:
    """If participants are already in room, tasks are created for them."""
    p1 = _make_participant(identity="alice", metadata=_valid_metadata())
    ctx = _make_ctx(remote_participants={"alice": p1})

    with patch(
        "cozyvoice.livekit_entrypoint._handle_participant",
        new_callable=AsyncMock,
    ) as mock_handle:
        await entrypoint(ctx)

        # Give the created task a chance to run
        await asyncio.sleep(0.05)

    # _handle_participant should have been called for alice
    mock_handle.assert_awaited_once_with(ctx, p1)


async def test_entrypoint_participant_connected_creates_task() -> None:
    """participant_connected event triggers _handle_participant."""
    ctx = _make_ctx()

    with patch(
        "cozyvoice.livekit_entrypoint._handle_participant",
        new_callable=AsyncMock,
    ) as mock_handle:
        await entrypoint(ctx)

        # Simulate participant connecting
        p = _make_participant(identity="bob", metadata=_valid_metadata())
        for handler in ctx.room._handlers.get("participant_connected", []):
            handler(p)

        await asyncio.sleep(0.05)

    mock_handle.assert_awaited_once_with(ctx, p)


async def test_entrypoint_duplicate_participant_ignored() -> None:
    """Same participant connecting twice does not create a second task."""
    ctx = _make_ctx()

    call_count = 0
    original_done = False

    async def _slow_handle(ctx, participant):
        nonlocal call_count, original_done
        call_count += 1
        await asyncio.sleep(0.2)
        original_done = True

    with patch(
        "cozyvoice.livekit_entrypoint._handle_participant",
        side_effect=_slow_handle,
    ):
        await entrypoint(ctx)

        p = _make_participant(identity="bob", metadata=_valid_metadata())
        for handler in ctx.room._handlers.get("participant_connected", []):
            handler(p)

        await asyncio.sleep(0.05)

        # Second connect with same identity while first is still running
        for handler in ctx.room._handlers.get("participant_connected", []):
            handler(p)

        await asyncio.sleep(0.05)

    # Should only have been called once
    assert call_count == 1


async def test_entrypoint_shutdown_callback_cancels_tasks() -> None:
    """The shutdown callback cancels running tasks."""
    ctx = _make_ctx()

    async def _blocking_handle(ctx, participant):
        await asyncio.sleep(100)

    with patch(
        "cozyvoice.livekit_entrypoint._handle_participant",
        side_effect=_blocking_handle,
    ):
        await entrypoint(ctx)

        p = _make_participant(identity="bob")
        for handler in ctx.room._handlers.get("participant_connected", []):
            handler(p)

        await asyncio.sleep(0.05)

    # Get the shutdown callback and call it
    cleanup = ctx.add_shutdown_callback.call_args[0][0]
    await cleanup()


# ---------------------------------------------------------------------------
# _main tests
# ---------------------------------------------------------------------------

@patch("cozyvoice.livekit_entrypoint.cli.run_app")
@patch("cozyvoice.livekit_entrypoint.logging.basicConfig")
def test_main_calls_cli_run_app(mock_logging: MagicMock, mock_run: MagicMock) -> None:
    """_main sets up logging and calls cli.run_app."""
    _main()
    mock_logging.assert_called_once()
    mock_run.assert_called_once()

    worker_opts = mock_run.call_args[0][0]
    assert worker_opts.entrypoint_fnc == entrypoint


@patch.dict(os.environ, {"LIVEKIT_AGENT_HTTP_PORT": "9999"})
@patch("cozyvoice.livekit_entrypoint.cli.run_app")
@patch("cozyvoice.livekit_entrypoint.logging.basicConfig")
def test_main_custom_port(mock_logging: MagicMock, mock_run: MagicMock) -> None:
    """_main respects LIVEKIT_AGENT_HTTP_PORT env var."""
    _main()
    worker_opts = mock_run.call_args[0][0]
    assert worker_opts.port == 9999
