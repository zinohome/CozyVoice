"""Brain LLM Adapter for livekit-agents.

把 Brain 的 `POST /v1/chat/voice`（SSE 流）包装成 livekit-agents `llm.LLM`，
使之能嵌入 `AgentSession(llm=...)` 跑自建 Realtime Pipeline（VAD → STT → LLM → TTS）。

设计要点
========

* **不传历史**：Brain 自己管会话上下文；Adapter 只从 chat_ctx 取最后一条 user message。
* **SSE 格式**：`data: {"choices":[{"delta":{"content":"..."}}]}` / `data: [DONE]`（与
  `BrainClient` 相同契约）。
* **错误处理**：非 2xx → `APIStatusError`；网络错 → `APIConnectionError`。

注意
----

Brain voice endpoint 现为 `POST /v1/chat/voice`（由另一 agent 并行开发）；若后续落地到
别的路径或改成 `/v1/chat/completions` 的 voice 变体，改 `_ENDPOINT` 常量即可。
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx
from livekit.agents import APIConnectionError, APIStatusError
from livekit.agents.llm import (
    LLM,
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    LLMStream,
)
from livekit.agents.llm.tool_context import Tool
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, NOT_GIVEN, NotGivenOr

logger = logging.getLogger("cozyvoice.brain_llm")

_ENDPOINT = "/v1/chat/voice"


def _extract_last_user_message(chat_ctx: ChatContext) -> str:
    """取 chat_ctx 里最后一条 role='user' 的 message 的文本内容。

    若 chat_ctx 为空或无 user message，返回空字符串（上层决定是否跳过请求）。
    """
    for msg in reversed(chat_ctx.items):
        # ChatMessage 才有 role / text_content；跳过 FunctionCall 等
        role = getattr(msg, "role", None)
        if role != "user":
            continue
        text = None
        if hasattr(msg, "text_content"):
            try:
                text = msg.text_content  # property
            except Exception:  # pragma: no cover - defensive
                text = None
        if text:
            return text
        # fallback: content 可能是 str
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for c in content:
                if isinstance(c, str):
                    return c
        return ""
    return ""


class BrainLLMAdapter(LLM):
    """把 Brain 的 voice SSE 端点包装成 livekit-agents LLM。"""

    def __init__(
        self,
        *,
        brain_url: str,
        brain_jwt: str,
        session_id: str,
        personality_id: str,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__()
        self._brain_url = brain_url.rstrip("/")
        self._brain_jwt = brain_jwt
        self._session_id = session_id
        self._personality_id = personality_id
        self._timeout = timeout
        # 允许测试注入 AsyncClient（MockTransport）
        self._http_client_override = http_client

    @property
    def model(self) -> str:
        return "cozy-brain-voice"

    @property
    def provider(self) -> str:
        return "cozy-brain"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        return _BrainLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=list(tools or []),
            conn_options=conn_options,
        )


class _BrainLLMStream(LLMStream):
    _llm: BrainLLMAdapter

    async def _run(self) -> None:
        assert isinstance(self._llm, BrainLLMAdapter)
        user_msg = _extract_last_user_message(self._chat_ctx)
        if not user_msg:
            logger.warning("BrainLLMAdapter: no user message in chat_ctx; emitting no chunks")
            return

        headers = {
            "Authorization": f"Bearer {self._llm._brain_jwt}",
            "X-Source-Channel": "voice-pipeline",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        body = {
            "session_id": self._llm._session_id,
            "personality_id": self._llm._personality_id,
            "message": user_msg,
        }

        chunk_id = uuid.uuid4().hex

        owns_client = self._llm._http_client_override is None
        client = self._llm._http_client_override or httpx.AsyncClient(
            base_url=self._llm._brain_url,
            timeout=httpx.Timeout(self._llm._timeout),
        )

        try:
            try:
                async with client.stream(
                    "POST", _ENDPOINT, json=body, headers=headers
                ) as resp:
                    if resp.status_code >= 400:
                        try:
                            err_body = await resp.aread()
                        except Exception:  # pragma: no cover
                            err_body = b""
                        raise APIStatusError(
                            f"Brain voice endpoint returned {resp.status_code}",
                            status_code=resp.status_code,
                            body=err_body.decode("utf-8", errors="replace")[:512] if err_body else None,
                        )

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            break
                        if not payload:
                            continue
                        try:
                            data = json.loads(payload)
                        except json.JSONDecodeError:
                            logger.debug("BrainLLMAdapter: skip non-JSON SSE line: %r", payload[:120])
                            continue

                        content = _extract_delta_content(data)
                        if not content:
                            continue

                        self._event_ch.send_nowait(
                            ChatChunk(
                                id=chunk_id,
                                delta=ChoiceDelta(role="assistant", content=content),
                            )
                        )
            except APIStatusError:
                raise
            except httpx.HTTPError as e:
                raise APIConnectionError(f"Brain voice endpoint connection error: {e}") from e
        finally:
            if owns_client:
                try:
                    await client.aclose()
                except Exception:  # pragma: no cover
                    pass


def _extract_delta_content(data: dict[str, Any]) -> str | None:
    """从 SSE payload 取 `choices[0].delta.content`。"""
    try:
        choices = data.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            return content
    except (AttributeError, IndexError, TypeError):
        return None
    return None


__all__ = ["BrainLLMAdapter"]
