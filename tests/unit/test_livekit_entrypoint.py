"""LiveKit entrypoint 单测（mock JobContext / Room / Participant）。

不连真 LiveKit / OpenAI —— 只验证：
  1) metadata 解析
  2) entrypoint() 注册了 participant_connected handler 并调用了 ctx.connect
  3) participant 缺 metadata 时被 skip（不会启动 _handle_participant 的 RealtimeCallState）
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cozyvoice import livekit_entrypoint as lke


def test_parse_participant_metadata_valid() -> None:
    raw = '{"brain_jwt": "t", "session_id": "s", "personality_id": "p"}'
    out = lke._parse_participant_metadata(raw)
    assert out == {"brain_jwt": "t", "session_id": "s", "personality_id": "p"}


def test_parse_participant_metadata_empty() -> None:
    assert lke._parse_participant_metadata(None) == {}
    assert lke._parse_participant_metadata("") == {}


def test_parse_participant_metadata_invalid_json() -> None:
    assert lke._parse_participant_metadata("not-json") == {}


class _FakeRoom:
    def __init__(self) -> None:
        self.name = "room-A"
        self._handlers: dict[str, list] = {}
        self.remote_participants: dict[str, object] = {}
        self.local_participant = MagicMock()

    def on(self, event: str, handler=None):
        if handler is None:
            def _wrap(fn):
                self._handlers.setdefault(event, []).append(fn)
                return fn
            return _wrap
        self._handlers.setdefault(event, []).append(handler)
        return handler

    def emit(self, event: str, *args, **kwargs) -> None:
        for h in self._handlers.get(event, []):
            h(*args, **kwargs)


class _FakeJobContext:
    def __init__(self) -> None:
        self.room = _FakeRoom()
        self.connect = AsyncMock()
        self._shutdown_cbs: list = []

    def add_shutdown_callback(self, cb) -> None:
        self._shutdown_cbs.append(cb)


@pytest.mark.asyncio
async def test_entrypoint_registers_participant_handler() -> None:
    ctx = _FakeJobContext()

    await lke.entrypoint(ctx)  # type: ignore[arg-type]

    ctx.connect.assert_awaited_once()
    # 至少注册了 participant_connected
    assert "participant_connected" in ctx.room._handlers
    assert len(ctx.room._handlers["participant_connected"]) >= 1
    # 有 shutdown 回调用于取消 participant 任务
    assert len(ctx._shutdown_cbs) >= 1


@pytest.mark.asyncio
async def test_handle_participant_skips_without_metadata(caplog) -> None:
    """participant 无 metadata → 直接 return，不起 handle_realtime_call。"""
    ctx = _FakeJobContext()
    participant = MagicMock()
    participant.identity = "no-meta"
    participant.metadata = None  # 缺失

    # 如果真的走到内部它会尝试 publish_track（在 fake room 上），可能出错。
    # 但没 metadata 应 early return。
    await lke._handle_participant(ctx, participant)  # type: ignore[arg-type]

    # local_participant.publish_track 不应被调
    ctx.room.local_participant.publish_track.assert_not_called()
