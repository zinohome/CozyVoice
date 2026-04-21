"""Mock TTS，返回固定 wav 字节（极小的伪 wav header + 静音）。"""

from __future__ import annotations

import struct
from typing import Literal

from cozyvoice.providers.base import TTSProvider, Voice


def _make_silent_wav(sample_rate: int = 16000, duration_ms: int = 100) -> bytes:
    num_samples = sample_rate * duration_ms // 1000
    data_size = num_samples * 2  # 16-bit mono
    header = b"RIFF"
    header += struct.pack("<I", 36 + data_size)
    header += b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    header += b"data"
    header += struct.pack("<I", data_size)
    return header + b"\x00" * data_size


class MockTTS(TTSProvider):
    name = "mock"

    async def synthesize(
        self, text: str, voice_id: str = "mock", format: Literal["wav", "mp3", "pcm"] = "wav"
    ) -> bytes:
        return _make_silent_wav()

    async def list_voices(self) -> list[Voice]:
        return [Voice(voice_id="mock", name="mock", language="zh")]
