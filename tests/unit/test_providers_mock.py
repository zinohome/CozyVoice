"""Mock providers 单测（验证 ABC 合约）。"""

from __future__ import annotations

from cozyvoice.providers.stt.mock import MockSTT
from cozyvoice.providers.tts.mock import MockTTS


async def test_mock_stt_returns_canned_text() -> None:
    stt = MockSTT(canned_text="hello")
    r = await stt.transcribe(b"ignored")
    assert r.text == "hello"


async def test_mock_tts_returns_wav_bytes() -> None:
    tts = MockTTS()
    r = await tts.synthesize("hi")
    assert r.startswith(b"RIFF")


async def test_mock_tts_list_voices() -> None:
    voices = await MockTTS().list_voices()
    assert len(voices) == 1
    assert voices[0].voice_id == "mock"
