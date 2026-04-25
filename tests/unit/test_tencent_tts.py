"""TencentTTS unit tests."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cozyvoice.providers.base import Voice


@pytest.fixture
def mock_tencent_sdk():
    with patch("cozyvoice.providers.tts.tencent.credential") as mock_cred, \
         patch("cozyvoice.providers.tts.tencent.tts_client") as mock_tts_client, \
         patch("cozyvoice.providers.tts.tencent.ClientProfile"), \
         patch("cozyvoice.providers.tts.tencent.HttpProfile"):
        mock_cred.Credential.return_value = MagicMock()
        mock_client_instance = MagicMock()
        mock_tts_client.TtsClient.return_value = mock_client_instance
        yield mock_client_instance


def test_name_attribute(mock_tencent_sdk):
    from cozyvoice.providers.tts.tencent import TencentTTS
    tts = TencentTTS(secret_id="sid", secret_key="skey")
    assert tts.name == "tencent"


@pytest.mark.asyncio
async def test_list_voices_returns_predefined(mock_tencent_sdk):
    from cozyvoice.providers.tts.tencent import TencentTTS
    tts = TencentTTS(secret_id="sid", secret_key="skey")
    voices = await tts.list_voices()
    assert len(voices) == 5
    assert all(isinstance(v, Voice) for v in voices)
    voice_ids = [v.voice_id for v in voices]
    assert "101001" in voice_ids
    assert "101004" in voice_ids


@pytest.mark.asyncio
async def test_synthesize_with_custom_voice_id(mock_tencent_sdk):
    from cozyvoice.providers.tts.tencent import TencentTTS, models

    audio_bytes = b"fake-audio-data"
    encoded = base64.b64encode(audio_bytes).decode()

    mock_resp = MagicMock()
    mock_resp.Audio = encoded
    mock_tencent_sdk.TextToVoice.return_value = mock_resp

    with patch("cozyvoice.providers.tts.tencent.models") as mock_models:
        mock_req = MagicMock()
        mock_models.TextToVoiceRequest.return_value = mock_req

        tts = TencentTTS(secret_id="sid", secret_key="skey")
        result = await tts.synthesize("Hello", voice_id="101004")

    assert result == audio_bytes


@pytest.mark.asyncio
async def test_synthesize_decodes_base64_audio(mock_tencent_sdk):
    from cozyvoice.providers.tts.tencent import TencentTTS

    expected_audio = b"decoded-audio-content"
    encoded = base64.b64encode(expected_audio).decode()

    mock_resp = MagicMock()
    mock_resp.Audio = encoded
    mock_tencent_sdk.TextToVoice.return_value = mock_resp

    with patch("cozyvoice.providers.tts.tencent.models") as mock_models:
        mock_models.TextToVoiceRequest.return_value = MagicMock()
        tts = TencentTTS(secret_id="sid", secret_key="skey")
        result = await tts.synthesize("Test text")

    assert result == expected_audio
