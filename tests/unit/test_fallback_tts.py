"""FallbackTTS chain unit tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from cozyvoice.providers.base import TTSProvider, Voice
from cozyvoice.providers.tts.fallback import FallbackTTS, TTSAllProvidersFailedError


class StubTTS(TTSProvider):
    def __init__(self, name_: str, audio: bytes | None = None, error: Exception | None = None):
        self.name = name_
        self._audio = audio
        self._error = error
        self.default_voice = "v1"
        self.timeout_s = 5.0
        self.call_count = 0

    async def synthesize(self, text, voice_id="v1", format="mp3") -> bytes:
        self.call_count += 1
        if self._error:
            raise self._error
        return self._audio or b""

    async def list_voices(self) -> list[Voice]:
        return [Voice(voice_id="v1", name=self.name, language="zh")]


async def test_first_provider_succeeds() -> None:
    p1 = StubTTS("p1", audio=b"audio-from-p1")
    p2 = StubTTS("p2", audio=b"audio-from-p2")
    fallback = FallbackTTS(providers=[p1, p2])
    result = await fallback.synthesize("hello")
    assert result == b"audio-from-p1"
    assert p1.call_count == 1
    assert p2.call_count == 0


async def test_first_fails_second_succeeds() -> None:
    p1 = StubTTS("p1", error=ConnectionError("down"))
    p2 = StubTTS("p2", audio=b"audio-from-p2")
    fallback = FallbackTTS(providers=[p1, p2])
    result = await fallback.synthesize("hello")
    assert result == b"audio-from-p2"
    assert p1.call_count == 1
    assert p2.call_count == 1


async def test_all_fail_raises() -> None:
    p1 = StubTTS("p1", error=ConnectionError("p1 down"))
    p2 = StubTTS("p2", error=TimeoutError("p2 timeout"))
    fallback = FallbackTTS(providers=[p1, p2])
    with pytest.raises(TTSAllProvidersFailedError):
        await fallback.synthesize("hello")


async def test_timeout_triggers_fallback() -> None:
    async def slow_synth(text, voice_id="v1", format="mp3"):
        await asyncio.sleep(10)
        return b"too-slow"

    p1 = StubTTS("p1")
    p1.synthesize = slow_synth  # type: ignore[assignment]
    p1.timeout_s = 0.05
    p2 = StubTTS("p2", audio=b"fast-audio")
    fallback = FallbackTTS(providers=[p1, p2])
    result = await fallback.synthesize("hello")
    assert result == b"fast-audio"


async def test_list_voices_aggregates_all_providers() -> None:
    p1 = StubTTS("p1")
    p2 = StubTTS("p2")
    fallback = FallbackTTS(providers=[p1, p2])
    voices = await fallback.list_voices()
    assert len(voices) == 2


async def test_empty_providers_raises() -> None:
    fallback = FallbackTTS(providers=[])
    with pytest.raises(TTSAllProvidersFailedError):
        await fallback.synthesize("hello")
