"""Mock STT，单测与本地演示用。"""

from __future__ import annotations

from cozyvoice.providers.base import STTProvider, STTResult


class MockSTT(STTProvider):
    name = "mock"

    def __init__(self, canned_text: str = "查上海天气") -> None:
        self._text = canned_text

    async def transcribe(
        self, audio: bytes, mime_type: str = "audio/wav", language: str | None = None
    ) -> STTResult:
        return STTResult(text=self._text, language=language or "zh")
