"""Edge TTS provider unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cozyvoice.providers.tts.edge_tts_provider import EdgeTTS


async def test_synthesize_returns_audio_bytes() -> None:
    tts = EdgeTTS()

    fake_communicate = AsyncMock()
    fake_communicate.stream = AsyncMock(return_value=_fake_stream([
        {"type": "audio", "data": b"chunk1"},
        {"type": "audio", "data": b"chunk2"},
        {"type": "WordBoundary", "data": None},
    ]))

    with patch("cozyvoice.providers.tts.edge_tts_provider.edge_tts.Communicate", return_value=fake_communicate):
        result = await tts.synthesize("你好", voice_id="zh-CN-XiaoxiaoNeural")
    assert result == b"chunk1chunk2"


async def test_synthesize_uses_correct_voice() -> None:
    tts = EdgeTTS(default_voice="zh-CN-YunxiNeural")

    fake_communicate = AsyncMock()
    fake_communicate.stream = AsyncMock(return_value=_fake_stream([
        {"type": "audio", "data": b"audio-data"},
    ]))

    with patch("cozyvoice.providers.tts.edge_tts_provider.edge_tts.Communicate") as mock_cls:
        mock_cls.return_value = fake_communicate
        await tts.synthesize("测试", voice_id="zh-CN-YunxiNeural")
    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args
    assert call_kwargs[1]["voice"] == "zh-CN-YunxiNeural" or call_kwargs[0][1] == "zh-CN-YunxiNeural"


async def test_list_voices_returns_chinese_voices() -> None:
    tts = EdgeTTS()
    voices = await tts.list_voices()
    assert len(voices) >= 2
    ids = {v.voice_id for v in voices}
    assert "zh-CN-XiaoxiaoNeural" in ids


async def test_name_attribute() -> None:
    tts = EdgeTTS()
    assert tts.name == "edge"


async def _fake_stream(items):
    for item in items:
        yield item
