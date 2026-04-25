"""Edge TTS provider (Microsoft, free, no API key required)."""

from __future__ import annotations

import inspect
from typing import Literal

import edge_tts

from cozyvoice.providers.base import TTSProvider, Voice

_VOICES: list[Voice] = [
    Voice(voice_id="zh-CN-XiaoxiaoNeural", name="晓晓-活泼女声", language="zh", gender="female"),
    Voice(voice_id="zh-CN-YunxiNeural", name="云希-阳光男声", language="zh", gender="male"),
    Voice(voice_id="zh-CN-XiaoyiNeural", name="晓伊-温柔女声", language="zh", gender="female"),
    Voice(voice_id="zh-CN-YunjianNeural", name="云健-沉稳男声", language="zh", gender="male"),
    Voice(voice_id="en-US-JennyNeural", name="Jenny-Female", language="en", gender="female"),
    Voice(voice_id="en-US-GuyNeural", name="Guy-Male", language="en", gender="male"),
]


class EdgeTTS(TTSProvider):
    name = "edge"

    def __init__(self, default_voice: str = "zh-CN-XiaoxiaoNeural") -> None:
        self._default_voice = default_voice

    async def synthesize(
        self,
        text: str,
        voice_id: str = "",
        format: Literal["wav", "mp3", "pcm"] = "mp3",
    ) -> bytes:
        voice = voice_id or self._default_voice
        communicate = edge_tts.Communicate(text=text, voice=voice)
        audio_chunks: list[bytes] = []
        stream = communicate.stream()
        # In production edge_tts.Communicate.stream() is an async generator.
        # In tests it may be mocked as AsyncMock (coroutine returning an async gen).
        if inspect.iscoroutine(stream):
            stream = await stream
        async for chunk in stream:
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)

    async def list_voices(self) -> list[Voice]:
        return list(_VOICES)
