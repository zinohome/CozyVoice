"""OpenAIWhisperSTT unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cozyvoice.providers.stt.openai_whisper import OpenAIWhisperSTT
from cozyvoice.providers.base import STTResult


@pytest.fixture
def mock_openai_client():
    with patch("cozyvoice.providers.stt.openai_whisper.AsyncOpenAI") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.audio = MagicMock()
        mock_instance.audio.transcriptions = MagicMock()
        mock_instance.audio.transcriptions.create = AsyncMock()
        mock_cls.return_value = mock_instance
        yield mock_cls, mock_instance


def test_name_attribute():
    with patch("cozyvoice.providers.stt.openai_whisper.AsyncOpenAI"):
        stt = OpenAIWhisperSTT(api_key="test-key")
    assert stt.name == "openai_whisper"


@pytest.mark.asyncio
async def test_transcribe_returns_stt_result(mock_openai_client):
    _, mock_instance = mock_openai_client
    mock_resp = MagicMock()
    mock_resp.text = "Hello world"
    mock_instance.audio.transcriptions.create.return_value = mock_resp

    stt = OpenAIWhisperSTT(api_key="test-key")
    result = await stt.transcribe(b"audio-bytes")

    assert isinstance(result, STTResult)
    assert result.text == "Hello world"
    assert result.language is None


@pytest.mark.asyncio
async def test_transcribe_passes_model_param(mock_openai_client):
    _, mock_instance = mock_openai_client
    mock_resp = MagicMock()
    mock_resp.text = "Test"
    mock_instance.audio.transcriptions.create.return_value = mock_resp

    stt = OpenAIWhisperSTT(api_key="test-key", model="whisper-2")
    await stt.transcribe(b"audio")

    call_kwargs = mock_instance.audio.transcriptions.create.call_args[1]
    assert call_kwargs["model"] == "whisper-2"


@pytest.mark.asyncio
async def test_transcribe_with_language(mock_openai_client):
    _, mock_instance = mock_openai_client
    mock_resp = MagicMock()
    mock_resp.text = "Bonjour"
    mock_instance.audio.transcriptions.create.return_value = mock_resp

    stt = OpenAIWhisperSTT(api_key="test-key")
    result = await stt.transcribe(b"audio", language="fr")

    assert result.language == "fr"
    call_kwargs = mock_instance.audio.transcriptions.create.call_args[1]
    assert call_kwargs["language"] == "fr"


def test_transcribe_handles_base_url():
    with patch("cozyvoice.providers.stt.openai_whisper.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        OpenAIWhisperSTT(api_key="test-key", base_url="https://custom.api.com")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "https://custom.api.com"


def test_no_base_url_passes_none():
    with patch("cozyvoice.providers.stt.openai_whisper.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        OpenAIWhisperSTT(api_key="test-key")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] is None
