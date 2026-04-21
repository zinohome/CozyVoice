"""RealtimeProvider + RealtimeSession ABC。M5 起实装。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class RealtimeSession(ABC):
    @abstractmethod
    async def send_audio(self, pcm_chunk: bytes) -> None: ...

    @abstractmethod
    async def receive_events(self) -> AsyncIterator[dict]:
        """yield 原始 provider 事件（function_call/audio/done/error 等）。"""

    @abstractmethod
    async def submit_tool_result(self, call_id: str, output: str) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class RealtimeProvider(ABC):
    name: str = "abstract-realtime"

    @abstractmethod
    async def open_session(
        self,
        instructions: str,
        voice: str = "alloy",
        tools: list[dict] | None = None,
    ) -> RealtimeSession: ...
