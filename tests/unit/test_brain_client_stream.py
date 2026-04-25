"""brain_client.chat_stream() unit tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from cozyvoice.bridge.brain_client import BrainClient


def _make_sse_response(chunks: list[str], *, status_code: int = 200) -> httpx.Response:
    lines = []
    for c in chunks:
        payload = json.dumps({"choices": [{"delta": {"content": c}}]})
        lines.append(f"data: {payload}")
    lines.append("data: [DONE]")
    body = "\n".join(lines)
    return httpx.Response(status_code=status_code, text=body)


async def test_chat_stream_yields_chunks() -> None:
    client = BrainClient(base_url="http://brain:8000")

    mock_http = AsyncMock(spec=httpx.AsyncClient)

    async def fake_stream(method, url, **kwargs):
        resp = _make_sse_response(["北京", "明天晴"])

        class FakeCtx:
            async def __aenter__(self_):
                return resp
            async def __aexit__(self_, *args):
                pass

        return FakeCtx()

    mock_http.stream = fake_stream
    client._client = mock_http

    collected = []
    async for chunk in client.chat_stream(
        jwt="test-jwt",
        session_id="s1",
        personality_id="p1",
        message="北京天气",
    ):
        collected.append(chunk)
    assert collected == ["北京", "明天晴"]


async def test_chat_stream_skips_empty_deltas() -> None:
    client = BrainClient(base_url="http://brain:8000")

    mock_http = AsyncMock(spec=httpx.AsyncClient)

    async def fake_stream(method, url, **kwargs):
        lines = [
            'data: {"choices":[{"delta":{}}]}',
            'data: {"choices":[{"delta":{"content":"有内容"}}]}',
            "data: [DONE]",
        ]

        class FakeCtx:
            async def __aenter__(self_):
                return httpx.Response(200, text="\n".join(lines))
            async def __aexit__(self_, *args):
                pass

        return FakeCtx()

    mock_http.stream = fake_stream
    client._client = mock_http

    collected = []
    async for chunk in client.chat_stream(jwt="t", session_id="s", personality_id="p", message="hi"):
        collected.append(chunk)
    assert collected == ["有内容"]


async def test_chat_stream_not_started_raises() -> None:
    client = BrainClient(base_url="http://brain:8000")
    with pytest.raises(RuntimeError, match="not started"):
        async for _ in client.chat_stream(jwt="t", session_id="s", personality_id="p", message="hi"):
            pass
