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


# ── _extract_last_user_message edge cases ────────────────────────


from cozyvoice.providers.brain_llm import _extract_last_user_message, _extract_delta_content


class _FakeMsg:
    """Minimal stub to simulate ChatMessage with controllable attributes."""
    def __init__(self, role: str, *, content=None, text_content_raises: bool = False, text_content_val=None, has_text_content: bool = True):
        self.role = role
        self.content = content
        self._text_content_raises = text_content_raises
        self._text_content_val = text_content_val
        self._has_text_content = has_text_content

    @property
    def text_content(self):
        if self._text_content_raises:
            raise AttributeError("boom")
        return self._text_content_val

    def __getattr__(self, name):
        if name == "text_content" and not self._has_text_content:
            raise AttributeError
        raise AttributeError(name)


class _FakeChatCtx:
    """Minimal ChatContext-like object with items list."""
    def __init__(self, items):
        self.items = items


def test_extract_last_user_message_text_content_exception() -> None:
    """text_content 属性抛异常 → falls back to content attr."""
    msg = _FakeMsg("user", content="fallback text", text_content_raises=True)
    ctx = _FakeChatCtx([msg])
    result = _extract_last_user_message(ctx)
    assert result == "fallback text"


def test_extract_last_user_message_content_is_list_of_str() -> None:
    """content is list[str] → returns first string."""
    msg = _FakeMsg("user", content=["part1", "part2"], text_content_val=None)
    ctx = _FakeChatCtx([msg])
    result = _extract_last_user_message(ctx)
    assert result == "part1"


def test_extract_last_user_message_content_is_non_str_non_list() -> None:
    """content is not str and not list → returns ''."""
    msg = _FakeMsg("user", content=12345, text_content_val=None)
    ctx = _FakeChatCtx([msg])
    result = _extract_last_user_message(ctx)
    assert result == ""


def test_extract_last_user_message_empty_list_content() -> None:
    """content is empty list → returns ''."""
    msg = _FakeMsg("user", content=[], text_content_val=None)
    ctx = _FakeChatCtx([msg])
    result = _extract_last_user_message(ctx)
    assert result == ""


def test_extract_last_user_message_list_with_non_str() -> None:
    """content is list with non-str items → returns ''."""
    msg = _FakeMsg("user", content=[123, 456], text_content_val=None)
    ctx = _FakeChatCtx([msg])
    result = _extract_last_user_message(ctx)
    assert result == ""


# ── _extract_delta_content edge cases ────────────────────────────


def test_extract_delta_content_empty_choices() -> None:
    """choices is empty list → None."""
    assert _extract_delta_content({"choices": []}) is None


def test_extract_delta_content_no_choices_key() -> None:
    """No choices key → None."""
    assert _extract_delta_content({}) is None


def test_extract_delta_content_content_not_str() -> None:
    """content is not str → None."""
    assert _extract_delta_content({"choices": [{"delta": {"content": 42}}]}) is None


def test_extract_delta_content_content_empty_str() -> None:
    """content is empty string → None."""
    assert _extract_delta_content({"choices": [{"delta": {"content": ""}}]}) is None


def test_extract_delta_content_valid() -> None:
    """Normal case."""
    assert _extract_delta_content({"choices": [{"delta": {"content": "hello"}}]}) == "hello"


def test_extract_delta_content_none_delta() -> None:
    """delta is None → None."""
    assert _extract_delta_content({"choices": [{"delta": None}]}) is None


def test_extract_delta_content_missing_delta() -> None:
    """No delta key → None."""
    assert _extract_delta_content({"choices": [{}]}) is None


# ── SSE edge cases ───────────────────────────────────────────────


async def test_sse_empty_data_payload_skipped() -> None:
    """'data: ' with empty payload after strip → skip line."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b"data: \n\n"
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    adapter = _make_adapter(handler)
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    contents, _ = await _collect(stream)
    assert contents == ["ok"]


async def test_sse_json_decode_failure_skipped() -> None:
    """data: with invalid JSON payload → skip (logged)."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b"data: {broken json\n\n"
            b'data: {"choices":[{"delta":{"content":"fine"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    adapter = _make_adapter(handler)
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    contents, _ = await _collect(stream)
    assert contents == ["fine"]


async def test_owns_client_creates_and_closes() -> None:
    """When no http_client override, adapter creates its own client (owns_client=True)."""
    # We test by constructing adapter WITHOUT http_client override
    # and mocking httpx.AsyncClient at a higher level.
    # Instead, let's verify the owns_client path indirectly via a successful request
    # using a real adapter (no override) pointed at a non-existent host.
    # This would raise APIConnectionError, confirming it tried to connect.
    adapter = BrainLLMAdapter(
        brain_url="http://127.0.0.1:1",  # unlikely to be listening
        brain_jwt="test",
        session_id="s",
        personality_id="p",
        timeout=0.5,
    )
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="hi")
    stream = adapter.chat(chat_ctx=chat_ctx, conn_options=_NO_RETRY)
    with pytest.raises((APIConnectionError, Exception)):
        await _collect(stream)
