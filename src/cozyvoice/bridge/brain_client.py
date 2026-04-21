"""反向调 Brain /v1/chat/completions。透传 JWT + 注入 X-Source-Channel: voice。"""

from __future__ import annotations

import json
import uuid

import httpx


class BrainClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
        )

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat_collect(
        self,
        jwt: str,
        session_id: str | uuid.UUID,
        personality_id: str | uuid.UUID,
        message: str,
    ) -> str:
        """调 Brain SSE 端点，收齐所有 chunk 后拼出完整文本。"""
        if self._client is None:
            raise RuntimeError("BrainClient not started")

        headers = {
            "Authorization": f"Bearer {jwt}",
            "X-Source-Channel": "voice",
            "Content-Type": "application/json",
        }
        body = {
            "session_id": str(session_id),
            "personality_id": str(personality_id),
            "message": message,
        }

        collected = ""
        async with self._client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (data.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content") or ""
                collected += piece
        return collected
