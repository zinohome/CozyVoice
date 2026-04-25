"""OpenAI TTS provider unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cozyvoice.providers.tts.openai_tts import OpenAITTS


@pytest.fixture
def mock_openai_client():
    client = AsyncMock()
    response = AsyncMock()
    response.content = b"fake-mp3-bytes"
    client.audio.speech.create = AsyncMock(return_value=response)
    return client


async def test_synthesize_returns_audio_bytes(mock_openai_client) -> None:
    tts = OpenAITTS(api_key="sk-test")
    tts._client = mock_openai_client
    result = await tts.synthesize("hello", voice_id="shimmer", format="mp3")
    assert result == b"fake-mp3-bytes"
    mock_openai_client.audio.speech.create.assert_awaited_once()


async def test_synthesize_passes_correct_params(mock_openai_client) -> None:
    tts = OpenAITTS(api_key="sk-test", model="tts-1-hd")
    tts._client = mock_openai_client
    await tts.synthesize("你好世界", voice_id="alloy", format="wav")
    call_kwargs = mock_openai_client.audio.speech.create.await_args.kwargs
    assert call_kwargs["input"] == "你好世界"
    assert call_kwargs["voice"] == "alloy"
    assert call_kwargs["model"] == "tts-1-hd"
    assert call_kwargs["response_format"] == "wav"


async def test_list_voices_returns_predefined() -> None:
    tts = OpenAITTS(api_key="sk-test")
    voices = await tts.list_voices()
    assert len(voices) >= 6
    ids = {v.voice_id for v in voices}
    assert "shimmer" in ids
    assert "alloy" in ids


async def test_name_attribute() -> None:
    tts = OpenAITTS(api_key="sk-test")
    assert tts.name == "openai"
