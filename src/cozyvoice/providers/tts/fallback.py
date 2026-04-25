"""FallbackTTS: ordered provider chain with per-provider timeout."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from cozyvoice.providers.base import TTSProvider, Voice

logger = logging.getLogger(__name__)


class TTSAllProvidersFailedError(Exception):
    pass


class FallbackTTS(TTSProvider):
    name = "fallback"

    def __init__(self, providers: list[TTSProvider]) -> None:
        self._providers = providers

    async def synthesize(
        self,
        text: str,
        voice_id: str = "",
        format: Literal["wav", "mp3", "pcm"] = "mp3",
    ) -> bytes:
        last_error: Exception | None = None
        for provider in self._providers:
            vid = voice_id or getattr(provider, "default_voice", "") or voice_id
            timeout = getattr(provider, "timeout_s", 10.0)
            try:
                return await asyncio.wait_for(
                    provider.synthesize(text, vid, format),
                    timeout=timeout,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "TTS provider %s failed (%s), trying next",
                    provider.name,
                    type(e).__name__,
                )
                continue
        raise TTSAllProvidersFailedError(
            f"all {len(self._providers)} TTS providers failed"
            + (f"; last error: {last_error}" if last_error else "")
        )

    async def list_voices(self) -> list[Voice]:
        result: list[Voice] = []
        for provider in self._providers:
            try:
                result.extend(await provider.list_voices())
            except Exception:
                pass
        return result
