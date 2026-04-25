"""OpenAI TTS provider (/v1/audio/speech)."""

from __future__ import annotations

from typing import Literal

from openai import AsyncOpenAI

from cozyvoice.providers.base import TTSProvider, Voice

_VOICES: list[Voice] = [
    Voice(voice_id="alloy", name="Alloy", language="multi", gender="neutral"),
    Voice(voice_id="echo", name="Echo", language="multi", gender="male"),
    Voice(voice_id="fable", name="Fable", language="multi", gender="neutral"),
    Voice(voice_id="onyx", name="Onyx", language="multi", gender="male"),
    Voice(voice_id="nova", name="Nova", language="multi", gender="female"),
    Voice(voice_id="shimmer", name="Shimmer", language="multi", gender="female"),
]


class OpenAITTS(TTSProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "tts-1",
        base_url: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def synthesize(
        self,
        text: str,
        voice_id: str = "shimmer",
        format: Literal["wav", "mp3", "pcm"] = "mp3",
    ) -> bytes:
        response = await self._client.audio.speech.create(
            model=self._model,
            voice=voice_id,
            input=text,
            response_format=format,
        )
        return response.content

    async def list_voices(self) -> list[Voice]:
        return list(_VOICES)
