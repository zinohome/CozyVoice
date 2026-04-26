"""FallbackTTS: ordered provider chain with per-provider timeout."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal

from cozyvoice.metrics import tts_duration_seconds, tts_fallback_total, tts_requests_total
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
        for idx, provider in enumerate(self._providers):
            vid = voice_id or getattr(provider, "default_voice", "") or voice_id
            timeout = getattr(provider, "timeout_s", 10.0)
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    provider.synthesize(text, vid, format),
                    timeout=timeout,
                )
                tts_duration_seconds.labels(provider=provider.name).observe(
                    time.perf_counter() - t0
                )
                tts_requests_total.labels(provider=provider.name, status="ok").inc()
                return result
            except Exception as e:
                tts_duration_seconds.labels(provider=provider.name).observe(
                    time.perf_counter() - t0
                )
                tts_requests_total.labels(provider=provider.name, status="error").inc()
                last_error = e
                logger.warning(
                    "TTS provider %s failed (%s), trying next",
                    provider.name,
                    type(e).__name__,
                )
                # Record fallback trigger if there is a next provider
                if idx + 1 < len(self._providers):
                    next_provider = self._providers[idx + 1]
                    tts_fallback_total.labels(
                        failed_provider=provider.name,
                        next_provider=next_provider.name,
                    ).inc()
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
