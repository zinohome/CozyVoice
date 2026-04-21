"""brain_client 单测（mock httpx）。"""

from __future__ import annotations

import httpx
import pytest

from cozyvoice.bridge.brain_client import BrainClient


def _sse_bytes(chunks: list[str]) -> bytes:
    out = []
    for c in chunks:
        out.append(f"data: {c}\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode()


async def test_chat_collect() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = request.read().decode()
        body = _sse_bytes([
            '{"choices":[{"delta":{"content":"你好"}}]}',
            '{"choices":[{"delta":{"content":"世界"}}]}',
            '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
        ])
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = BrainClient(base_url="http://brain", timeout=5.0)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://brain")

    result = await client.chat_collect(
        jwt="test-token",
        session_id="00000000-0000-0000-0000-000000000000",
        personality_id="00000000-0000-0000-0000-000000000001",
        message="hi",
    )
    assert result == "你好世界"
    assert captured["headers"]["authorization"] == "Bearer test-token"
    assert captured["headers"]["x-source-channel"] == "voice"


async def test_raises_if_not_started() -> None:
    client = BrainClient(base_url="http://brain")
    with pytest.raises(RuntimeError, match="not started"):
        await client.chat_collect(jwt="x", session_id="s", personality_id="p", message="m")
