"""OpenAI Whisper STT。"""

from __future__ import annotations

import io

from openai import AsyncOpenAI

from cozyvoice.providers.base import STTProvider, STTResult


class OpenAIWhisperSTT(STTProvider):
    name = "openai_whisper"

    def __init__(self, api_key: str, model: str = "whisper-1", base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def transcribe(
        self,
        audio: bytes,
        mime_type: str = "audio/wav",
        language: str | None = None,
    ) -> STTResult:
        ext = {"audio/wav": "wav", "audio/mpeg": "mp3", "audio/mp4": "m4a", "audio/ogg": "ogg"}.get(
            mime_type, "wav"
        )
        buf = io.BytesIO(audio)
        buf.name = f"audio.{ext}"
        kwargs: dict = {"file": buf, "model": self._model}
        if language:
            kwargs["language"] = language
        resp = await self._client.audio.transcriptions.create(**kwargs)
        return STTResult(text=resp.text, language=language)
