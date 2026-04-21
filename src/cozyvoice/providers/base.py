"""Provider ABC：STT / TTS / Realtime 三套抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class STTResult:
    text: str
    language: str | None = None
    duration_ms: int = 0


@dataclass(frozen=True)
class Voice:
    voice_id: str
    name: str
    language: str
    gender: Literal["male", "female", "neutral", "unknown"] = "unknown"


class STTProvider(ABC):
    name: str = "abstract-stt"

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        mime_type: str = "audio/wav",
        language: str | None = None,
    ) -> STTResult:
        """音频 → 文本。"""


class TTSProvider(ABC):
    name: str = "abstract-tts"

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        format: Literal["wav", "mp3", "pcm"] = "wav",
    ) -> bytes:
        """文本 → 音频。"""

    @abstractmethod
    async def list_voices(self) -> list[Voice]: ...


class RealtimeProvider(ABC):
    """M1 仅 ABC 占位；M5 实装。"""

    name: str = "abstract-realtime"

    @abstractmethod
    async def start_session(self, **kwargs) -> object: ...
