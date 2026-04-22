"""OpenAI Realtime API（WebSocket 直调）实装。

协议参考：https://platform.openai.com/docs/guides/realtime
事件示例：
  session.update / session.created / input_audio_buffer.append
  response.create / response.audio.delta / response.function_call_arguments.*
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import AsyncIterator

import websockets

from cozyvoice.providers.realtime.base import RealtimeProvider, RealtimeSession

logger = logging.getLogger(__name__)

_OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"


class OpenAIRealtimeSession(RealtimeSession):
    def __init__(self, ws) -> None:
        self._ws = ws
        self._closed = False

    async def send_audio(self, pcm_chunk: bytes) -> None:
        if self._closed:
            return
        b64 = base64.b64encode(pcm_chunk).decode()
        await self._ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": b64,
        }))

    async def receive_events(self) -> AsyncIterator[dict]:
        try:
            async for raw in self._ws:
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("bad JSON from Realtime: %s", raw[:200])
        except websockets.ConnectionClosed:
            return

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        if self._closed:
            return
        await self._ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            },
        }))
        await self._ws.send(json.dumps({"type": "response.create"}))

    async def close(self) -> None:
        self._closed = True
        try:
            await self._ws.close()
        except Exception:
            pass


class OpenAIRealtimeProvider(RealtimeProvider):
    name = "openai_realtime"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-realtime-preview",
        base_url: str | None = None,
    ) -> None:
        """base_url：可选，OpenAI-compatible 代理的 Realtime 入口。

        默认 wss://api.openai.com/v1/realtime。若传入 http(s)://... 会自动替换 scheme 为 ws(s)://。
        支持 Realtime 的代理（如 New-API）可与 Chat/STT 共用同一个 base_url。
        """
        self._api_key = api_key
        self._model = model
        if base_url:
            # http:// → ws://, https:// → wss://
            ws_base = base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1).rstrip("/")
            # 若 base 未带 /realtime 路径后缀则自动拼
            if not ws_base.endswith("/realtime"):
                ws_base = f"{ws_base}/realtime"
            self._ws_base = ws_base
        else:
            self._ws_base = _OPENAI_REALTIME_URL

    async def open_session(
        self,
        instructions: str,
        voice: str = "alloy",
        tools: list[dict] | None = None,
    ) -> RealtimeSession:
        url = f"{self._ws_base}?model={self._model}"
        # OpenAI Realtime 要求 **只用 header 或 subprotocol 二选一**，
        # 同时发两者 → 4000 "You must only send one beta header or protocol entry."
        # 用 subprotocol-only（代理/浏览器通用兼容），不再发 OpenAI-Beta header。
        headers = {"Authorization": f"Bearer {self._api_key}"}
        ws = await websockets.connect(
            url,
            additional_headers=headers,
            subprotocols=["openai-beta.realtime-v1"],
        )

        # 读一条 server 的欢迎帧（session.created）确认协议正常，顺便捕获服务端 hello 消息
        try:
            first = await asyncio.wait_for(ws.recv(), timeout=5.0)
            logger.info("realtime welcome: %s", first[:500] if isinstance(first, str) else first[:500])
        except asyncio.TimeoutError:
            logger.warning("no welcome frame in 5s, proceeding to session.update")

        session_update = {
            "type": "session.update",
            "session": {
                "instructions": instructions,
                "voice": voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad"},
            },
        }
        if tools:
            session_update["session"]["tools"] = tools
        await ws.send(json.dumps(session_update))
        # 读 session.updated（或 error）；若是 error，日志里会带详细原因
        try:
            ack = await asyncio.wait_for(ws.recv(), timeout=5.0)
            logger.info("session.update ack: %s", ack[:500] if isinstance(ack, str) else ack[:500])
        except asyncio.TimeoutError:
            logger.warning("no ack for session.update in 5s")

        return OpenAIRealtimeSession(ws)
