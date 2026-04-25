"""TTS fallback chain integration test."""

from __future__ import annotations

import pytest

from cozyvoice.providers.base import TTSProvider, Voice
from cozyvoice.providers.tts.fallback import FallbackTTS, TTSAllProvidersFailedError
from cozyvoice.providers.tts.mock import MockTTS

pytestmark = pytest.mark.integration


class FailingTTS(TTSProvider):
    name = "failing"

    async def synthesize(self, text, voice_id="", format="mp3") -> bytes:
        raise ConnectionError("provider down")

    async def list_voices(self) -> list[Voice]:
        return []


async def test_fallback_chain_skips_failing_to_mock() -> None:
    failing = FailingTTS()
    failing.timeout_s = 1.0
    failing.default_voice = "v1"
    mock = MockTTS()
    mock.timeout_s = 5.0
    mock.default_voice = "mock"

    fallback = FallbackTTS(providers=[failing, mock])
    result = await fallback.synthesize("hello")
    assert result.startswith(b"RIFF")


async def test_all_failing_raises() -> None:
    f1 = FailingTTS()
    f1.timeout_s = 1.0
    f1.default_voice = "v1"
    f2 = FailingTTS()
    f2.timeout_s = 1.0
    f2.default_voice = "v2"

    fallback = FallbackTTS(providers=[f1, f2])
    with pytest.raises(TTSAllProvidersFailedError):
        await fallback.synthesize("hello")


async def test_fallback_list_voices_aggregates() -> None:
    failing = FailingTTS()
    mock = MockTTS()
    fallback = FallbackTTS(providers=[failing, mock])
    voices = await fallback.list_voices()
    assert len(voices) >= 1
