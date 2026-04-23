"""BrainLLMAdapter 单测（mock httpx）。"""

from __future__ import annotations

import httpx
import pytest
from livekit.agents import APIConnectionError, APIStatusError
from livekit.agents.llm import ChatContext
from livekit.agents.types import APIConnectOptions

_NO_RETRY = APIConnectOptions(max_retry=0, timeout=10.0, retry_interval=0.01)

from cozyvoice.providers.brain_llm import BrainLLMAdapter


def _sse_bytes(chunks: list[str]) -> bytes:
    out = []
    for c in chunks:
        out.append(f"data: {c}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode()


def _make_adapter(handler) -> BrainLLMAdapter:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://brain")
    return BrainLLMAdapter(
        brain_url="http://brain",
        brain_jwt="test-jwt",
        session_id="sess-1",
        personality_id="pers-1",
        http_client=client,
    )


async def _collect(stream) -> tuple[list[str], list]:
    chunks: list = []
    contents: list[str] = []
    async for ev in stream:
        chunks.append(ev)
        if ev.delta and ev.delta.content:
            contents.append(ev.delta.content)
    return contents, chunks


async def test_streams_assistant_chunks() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = request.read().decode()
        captured["url"] = str(request.url)
        body = _sse_bytes([
            '{"choices":[{"delta":{"content":"你好"}}]}',
            '{"choices":[{"delta":{"content":"，世界"}}]}',
            '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
        ])
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    adapter = _make_adapter(handler)
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="你好吗")

    stream = adapter.chat(chat_ctx=chat_ctx)
    contents, chunks = await _collect(stream)

    assert contents == ["你好", "，世界"]
    # 每个 chunk 都应当带 role=assistant
    assert all(c.delta.role == "assistant" for c in chunks if c.delta)
    # URL 命中 /v1/chat/voice
    assert "/v1/chat/voice" in captured["url"]
    # 鉴权头
    assert captured["headers"]["authorization"] == "Bearer test-jwt"
    # 仅传最后一条 user message
    assert '"message":"你好吗"' in captured["body"]


async def test_only_last_user_message_is_sent() -> None:
    """chat_ctx 含 user / assistant / user 三条时，应只发最后那条 user。"""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(200, content=_sse_bytes([]), headers={"content-type": "text/event-stream"})

    adapter = _make_adapter(handler)
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="first question")
    chat_ctx.add_message(role="assistant", content="first answer")
    chat_ctx.add_message(role="user", content="second question")

    stream = adapter.chat(chat_ctx=chat_ctx)
    await _collect(stream)

    assert '"message":"second question"' in captured["body"]
    assert "first question" not in captured["body"]
    assert "first answer" not in captured["body"]


async def test_done_terminates_stream() -> None:
    """[DONE] 后流立刻关闭，不再消费后续内容。"""

    def handler(request: httpx.Request) -> httpx.Response:
        # 故意在 [DONE] 之后再放一条 —— 应该被忽略
        body = (
            b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n'
            b"data: [DONE]\n\n"
            b'data: {"choices":[{"delta":{"content":"should-not-appear"}}]}\n\n'
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    adapter = _make_adapter(handler)
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")

    stream = adapter.chat(chat_ctx=chat_ctx)
    contents, _ = await _collect(stream)
    assert contents == ["A"]


async def test_http_401_raises_api_status_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid token"})

    adapter = _make_adapter(handler)
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")

    stream = adapter.chat(chat_ctx=chat_ctx, conn_options=_NO_RETRY)
    with pytest.raises(APIStatusError) as exc_info:
        await _collect(stream)
    assert exc_info.value.status_code == 401


async def test_http_500_raises_api_status_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    adapter = _make_adapter(handler)
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")

    stream = adapter.chat(chat_ctx=chat_ctx, conn_options=_NO_RETRY)
    with pytest.raises(APIStatusError) as exc_info:
        await _collect(stream)
    assert exc_info.value.status_code == 500


async def test_network_error_raises_api_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    adapter = _make_adapter(handler)
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")

    stream = adapter.chat(chat_ctx=chat_ctx, conn_options=_NO_RETRY)
    with pytest.raises((APIConnectionError, Exception)) as exc_info:
        await _collect(stream)
    # 确认是 APIConnectionError 或其父类 APIError（被 LLMStream 包装重试）
    from livekit.agents import APIConnectionError as _ACE
    from livekit.agents._exceptions import APIError as _APIErr
    assert isinstance(exc_info.value, (_ACE, _APIErr))


async def test_empty_chat_ctx_produces_no_chunks() -> None:
    """chat_ctx 空 / 无 user message → adapter 不发请求、流立刻关闭。"""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, content=_sse_bytes([]), headers={"content-type": "text/event-stream"})

    adapter = _make_adapter(handler)
    chat_ctx = ChatContext()  # 空

    stream = adapter.chat(chat_ctx=chat_ctx)
    contents, chunks = await _collect(stream)
    assert contents == []
    assert chunks == []
    assert call_count["n"] == 0, "expected no HTTP request when chat_ctx has no user message"
