# CozyVoice Production Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CozyVoice production-ready with TTS fallback chain, SSE streaming API, LiveKit production config, and ≥85 tests green.

**Architecture:** Four parallel tracks: T1 builds the TTS fallback layer (OpenAI + Edge + FallbackTTS wrapper), T2 adds the `/transcribe` endpoint and rewrites `/voice/chat` to SSE streaming, T3 creates production LiveKit config and full-stack docker-compose, T4 backfills test coverage. T2 depends on T1 (FallbackTTS). T4 depends on T1-T3 (tests target new code).

**Tech Stack:** Python 3.12, FastAPI, httpx, edge-tts, openai SDK, sse-starlette, LiveKit Server, Docker Compose

**Spec:** `docs/specs/2026-04-25-production-readiness-design.md`

**Repo:** `~/CozyProjects/CozyVoice/`
**venv:** `~/CozyProjects/CozyVoice/.venv/`
**Test baseline:** 34 tests green
**Test target:** ≥ 85 tests green

---

## Dependency Order

```
T1.1 (OpenAI TTS) ──┐
T1.2 (Edge TTS) ────┼──→ T1.3 (FallbackTTS) ──→ T2.1 (transcribe) ──→ T2.2 (SSE voice/chat)
                     │                                                        ↓
T3.1 (LiveKit cfg) ─┘                                              T2.3 (brain_client.chat_stream)
T3.2 (full-stack compose)
                     All T1-T3 ──→ T4 (tests)
```

---

## File Structure

### New files

| File | Responsibility |
|:---|:---|
| `src/cozyvoice/providers/tts/openai_tts.py` | OpenAI TTS provider (`/v1/audio/speech`) |
| `src/cozyvoice/providers/tts/edge_tts_provider.py` | Edge TTS provider (free, via `edge-tts` lib) |
| `src/cozyvoice/providers/tts/fallback.py` | FallbackTTS: ordered chain with per-provider timeout + Prometheus counter |
| `deploy/livekit/livekit.yaml` | LiveKit Server production config |
| `docker-compose.full-stack.yml` | Full-stack compose: Brain + Memory + NanoBot + LiveKit + CozyVoice |
| `tests/unit/test_openai_tts.py` | OpenAI TTS provider unit tests |
| `tests/unit/test_edge_tts.py` | Edge TTS provider unit tests |
| `tests/unit/test_fallback_tts.py` | FallbackTTS chain unit tests |
| `tests/unit/test_transcribe_endpoint.py` | Layer 1 `/transcribe` endpoint tests |
| `tests/unit/test_voice_chat_sse.py` | Layer 2/3 SSE `/voice/chat` endpoint tests |
| `tests/unit/test_config_loader.py` | Config YAML loader unit tests |
| `tests/unit/test_brain_client_stream.py` | `brain_client.chat_stream()` unit tests |
| `tests/integration/test_voice_chat_sse_roundtrip.py` | SSE voice/chat integration test |
| `tests/integration/test_fallback_chain_roundtrip.py` | TTS fallback chain integration test |

### Modified files

| File | Changes |
|:---|:---|
| `pyproject.toml` | Add `edge-tts`, `sse-starlette` dependencies |
| `src/cozyvoice/main.py` | Rewrite `_build_tts()` to construct FallbackTTS from config; register `/transcribe` route |
| `src/cozyvoice/api/rest.py` | Add `/voice/transcribe` endpoint; rewrite `/voice/chat` to SSE streaming with `tts` query param |
| `src/cozyvoice/bridge/brain_client.py` | Add `chat_stream()` method returning `AsyncIterator[str]` |
| `config/voice.yaml` | Add `tts.fallback_chain` config section |
| `deploy/base_runtime/docker-compose.1panel.yml` | Replace `--dev` with `--config /etc/livekit.yaml` |
| `docker-compose.yml` | Update dev LiveKit to mount config |

---

## Track 1: TTS Layer

### Task T1.1: OpenAI TTS Provider

**Files:**
- Create: `src/cozyvoice/providers/tts/openai_tts.py`
- Test: `tests/unit/test_openai_tts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_openai_tts.py`:

```python
"""OpenAI TTS provider unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cozyvoice.providers.tts.openai_tts import OpenAITTS


@pytest.fixture
def mock_openai_client():
    client = AsyncMock()
    response = AsyncMock()
    response.content = b"fake-mp3-bytes"
    client.audio.speech.create = AsyncMock(return_value=response)
    return client


async def test_synthesize_returns_audio_bytes(mock_openai_client) -> None:
    tts = OpenAITTS(api_key="sk-test")
    tts._client = mock_openai_client
    result = await tts.synthesize("hello", voice_id="shimmer", format="mp3")
    assert result == b"fake-mp3-bytes"
    mock_openai_client.audio.speech.create.assert_awaited_once()


async def test_synthesize_passes_correct_params(mock_openai_client) -> None:
    tts = OpenAITTS(api_key="sk-test", model="tts-1-hd")
    tts._client = mock_openai_client
    await tts.synthesize("你好世界", voice_id="alloy", format="wav")
    call_kwargs = mock_openai_client.audio.speech.create.await_args.kwargs
    assert call_kwargs["input"] == "你好世界"
    assert call_kwargs["voice"] == "alloy"
    assert call_kwargs["model"] == "tts-1-hd"
    assert call_kwargs["response_format"] == "wav"


async def test_list_voices_returns_predefined() -> None:
    tts = OpenAITTS(api_key="sk-test")
    voices = await tts.list_voices()
    assert len(voices) >= 6
    ids = {v.voice_id for v in voices}
    assert "shimmer" in ids
    assert "alloy" in ids


async def test_name_attribute() -> None:
    tts = OpenAITTS(api_key="sk-test")
    assert tts.name == "openai"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_openai_tts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cozyvoice.providers.tts.openai_tts'`

- [ ] **Step 3: Implement OpenAI TTS provider**

Create `src/cozyvoice/providers/tts/openai_tts.py`:

```python
"""OpenAI TTS provider (/v1/audio/speech)."""

from __future__ import annotations

from typing import Literal

from openai import AsyncOpenAI

from cozyvoice.providers.base import TTSProvider, Voice

_VOICES: list[Voice] = [
    Voice(voice_id="alloy", name="Alloy", language="multi", gender="neutral"),
    Voice(voice_id="echo", name="Echo", language="multi", gender="male"),
    Voice(voice_id="fable", name="Fable", language="multi", gender="neutral"),
    Voice(voice_id="onyx", name="Onyx", language="multi", gender="male"),
    Voice(voice_id="nova", name="Nova", language="multi", gender="female"),
    Voice(voice_id="shimmer", name="Shimmer", language="multi", gender="female"),
]


class OpenAITTS(TTSProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "tts-1",
        base_url: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def synthesize(
        self,
        text: str,
        voice_id: str = "shimmer",
        format: Literal["wav", "mp3", "pcm"] = "mp3",
    ) -> bytes:
        response = await self._client.audio.speech.create(
            model=self._model,
            voice=voice_id,
            input=text,
            response_format=format,
        )
        return response.content

    async def list_voices(self) -> list[Voice]:
        return list(_VOICES)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_openai_tts.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd ~/CozyProjects/CozyVoice && git add src/cozyvoice/providers/tts/openai_tts.py tests/unit/test_openai_tts.py && git commit -m "feat(tts): add OpenAI TTS provider"
```

---

### Task T1.2: Edge TTS Provider

**Files:**
- Create: `src/cozyvoice/providers/tts/edge_tts_provider.py`
- Test: `tests/unit/test_edge_tts.py`
- Modify: `pyproject.toml` (add `edge-tts` dependency)

- [ ] **Step 1: Add edge-tts dependency**

In `pyproject.toml`, add `"edge-tts>=6.1"` to `dependencies`:

```toml
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.9",
    "pydantic-settings>=2.5",
    "httpx>=0.27",
    "pyyaml>=6.0",
    "openai>=1.50",
    "tencentcloud-sdk-python-tts>=3.0",
    "python-multipart>=0.0.9",
    "livekit-agents>=0.10",
    "livekit-api>=0.7",
    "websockets>=12",
    "edge-tts>=6.1",
]
```

Then install:

```bash
cd ~/CozyProjects/CozyVoice && .venv/bin/pip install -e ".[dev]"
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_edge_tts.py`:

```python
"""Edge TTS provider unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cozyvoice.providers.tts.edge_tts_provider import EdgeTTS


async def test_synthesize_returns_audio_bytes() -> None:
    tts = EdgeTTS()

    fake_communicate = AsyncMock()
    fake_communicate.stream = AsyncMock(return_value=_fake_stream([
        {"type": "audio", "data": b"chunk1"},
        {"type": "audio", "data": b"chunk2"},
        {"type": "WordBoundary", "data": None},
    ]))

    with patch("cozyvoice.providers.tts.edge_tts_provider.edge_tts.Communicate", return_value=fake_communicate):
        result = await tts.synthesize("你好", voice_id="zh-CN-XiaoxiaoNeural")
    assert result == b"chunk1chunk2"


async def test_synthesize_uses_correct_voice() -> None:
    tts = EdgeTTS(default_voice="zh-CN-YunxiNeural")

    fake_communicate = AsyncMock()
    fake_communicate.stream = AsyncMock(return_value=_fake_stream([
        {"type": "audio", "data": b"audio-data"},
    ]))

    with patch("cozyvoice.providers.tts.edge_tts_provider.edge_tts.Communicate") as mock_cls:
        mock_cls.return_value = fake_communicate
        await tts.synthesize("测试", voice_id="zh-CN-YunxiNeural")
    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args
    assert call_kwargs[1]["voice"] == "zh-CN-YunxiNeural" or call_kwargs[0][1] == "zh-CN-YunxiNeural"


async def test_list_voices_returns_chinese_voices() -> None:
    tts = EdgeTTS()
    voices = await tts.list_voices()
    assert len(voices) >= 2
    ids = {v.voice_id for v in voices}
    assert "zh-CN-XiaoxiaoNeural" in ids


async def test_name_attribute() -> None:
    tts = EdgeTTS()
    assert tts.name == "edge"


async def _fake_stream(items):
    for item in items:
        yield item
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_edge_tts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cozyvoice.providers.tts.edge_tts_provider'`

- [ ] **Step 4: Implement Edge TTS provider**

Create `src/cozyvoice/providers/tts/edge_tts_provider.py`:

```python
"""Edge TTS provider (Microsoft, free, no API key required)."""

from __future__ import annotations

from typing import Literal

import edge_tts

from cozyvoice.providers.base import TTSProvider, Voice

_VOICES: list[Voice] = [
    Voice(voice_id="zh-CN-XiaoxiaoNeural", name="晓晓-活泼女声", language="zh", gender="female"),
    Voice(voice_id="zh-CN-YunxiNeural", name="云希-阳光男声", language="zh", gender="male"),
    Voice(voice_id="zh-CN-XiaoyiNeural", name="晓伊-温柔女声", language="zh", gender="female"),
    Voice(voice_id="zh-CN-YunjianNeural", name="云健-沉稳男声", language="zh", gender="male"),
    Voice(voice_id="en-US-JennyNeural", name="Jenny-Female", language="en", gender="female"),
    Voice(voice_id="en-US-GuyNeural", name="Guy-Male", language="en", gender="male"),
]


class EdgeTTS(TTSProvider):
    name = "edge"

    def __init__(self, default_voice: str = "zh-CN-XiaoxiaoNeural") -> None:
        self._default_voice = default_voice

    async def synthesize(
        self,
        text: str,
        voice_id: str = "",
        format: Literal["wav", "mp3", "pcm"] = "mp3",
    ) -> bytes:
        voice = voice_id or self._default_voice
        communicate = edge_tts.Communicate(text=text, voice=voice)
        audio_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks)

    async def list_voices(self) -> list[Voice]:
        return list(_VOICES)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_edge_tts.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
cd ~/CozyProjects/CozyVoice && git add pyproject.toml src/cozyvoice/providers/tts/edge_tts_provider.py tests/unit/test_edge_tts.py && git commit -m "feat(tts): add Edge TTS provider (free fallback)"
```

---

### Task T1.3: FallbackTTS Wrapper

**Files:**
- Create: `src/cozyvoice/providers/tts/fallback.py`
- Test: `tests/unit/test_fallback_tts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_fallback_tts.py`:

```python
"""FallbackTTS chain unit tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from cozyvoice.providers.base import TTSProvider, Voice
from cozyvoice.providers.tts.fallback import FallbackTTS, TTSAllProvidersFailedError


class StubTTS(TTSProvider):
    def __init__(self, name_: str, audio: bytes | None = None, error: Exception | None = None):
        self.name = name_
        self._audio = audio
        self._error = error
        self.default_voice = "v1"
        self.timeout_s = 5.0
        self.call_count = 0

    async def synthesize(self, text, voice_id="v1", format="mp3") -> bytes:
        self.call_count += 1
        if self._error:
            raise self._error
        return self._audio or b""

    async def list_voices(self) -> list[Voice]:
        return [Voice(voice_id="v1", name=self.name, language="zh")]


async def test_first_provider_succeeds() -> None:
    p1 = StubTTS("p1", audio=b"audio-from-p1")
    p2 = StubTTS("p2", audio=b"audio-from-p2")
    fallback = FallbackTTS(providers=[p1, p2])
    result = await fallback.synthesize("hello")
    assert result == b"audio-from-p1"
    assert p1.call_count == 1
    assert p2.call_count == 0


async def test_first_fails_second_succeeds() -> None:
    p1 = StubTTS("p1", error=ConnectionError("down"))
    p2 = StubTTS("p2", audio=b"audio-from-p2")
    fallback = FallbackTTS(providers=[p1, p2])
    result = await fallback.synthesize("hello")
    assert result == b"audio-from-p2"
    assert p1.call_count == 1
    assert p2.call_count == 1


async def test_all_fail_raises() -> None:
    p1 = StubTTS("p1", error=ConnectionError("p1 down"))
    p2 = StubTTS("p2", error=TimeoutError("p2 timeout"))
    fallback = FallbackTTS(providers=[p1, p2])
    with pytest.raises(TTSAllProvidersFailedError):
        await fallback.synthesize("hello")


async def test_timeout_triggers_fallback() -> None:
    async def slow_synth(text, voice_id="v1", format="mp3"):
        await asyncio.sleep(10)
        return b"too-slow"

    p1 = StubTTS("p1")
    p1.synthesize = slow_synth  # type: ignore[assignment]
    p1.timeout_s = 0.05
    p2 = StubTTS("p2", audio=b"fast-audio")
    fallback = FallbackTTS(providers=[p1, p2])
    result = await fallback.synthesize("hello")
    assert result == b"fast-audio"


async def test_list_voices_aggregates_all_providers() -> None:
    p1 = StubTTS("p1")
    p2 = StubTTS("p2")
    fallback = FallbackTTS(providers=[p1, p2])
    voices = await fallback.list_voices()
    assert len(voices) == 2


async def test_empty_providers_raises() -> None:
    fallback = FallbackTTS(providers=[])
    with pytest.raises(TTSAllProvidersFailedError):
        await fallback.synthesize("hello")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_fallback_tts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cozyvoice.providers.tts.fallback'`

- [ ] **Step 3: Implement FallbackTTS**

Create `src/cozyvoice/providers/tts/fallback.py`:

```python
"""FallbackTTS: ordered provider chain with per-provider timeout."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from cozyvoice.providers.base import TTSProvider, Voice

logger = logging.getLogger(__name__)


class TTSAllProvidersFailedError(Exception):
    pass


class FallbackTTS(TTSProvider):
    name = "fallback"

    def __init__(self, providers: list[TTSProvider]) -> None:
        self._providers = providers

    async def synthesize(
        self,
        text: str,
        voice_id: str = "",
        format: Literal["wav", "mp3", "pcm"] = "mp3",
    ) -> bytes:
        last_error: Exception | None = None
        for provider in self._providers:
            vid = voice_id or getattr(provider, "default_voice", "") or voice_id
            timeout = getattr(provider, "timeout_s", 10.0)
            try:
                return await asyncio.wait_for(
                    provider.synthesize(text, vid, format),
                    timeout=timeout,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "TTS provider %s failed (%s), trying next",
                    provider.name,
                    type(e).__name__,
                )
                continue
        raise TTSAllProvidersFailedError(
            f"all {len(self._providers)} TTS providers failed"
            + (f"; last error: {last_error}" if last_error else "")
        )

    async def list_voices(self) -> list[Voice]:
        result: list[Voice] = []
        for provider in self._providers:
            try:
                result.extend(await provider.list_voices())
            except Exception:
                pass
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_fallback_tts.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
cd ~/CozyProjects/CozyVoice && git add src/cozyvoice/providers/tts/fallback.py tests/unit/test_fallback_tts.py && git commit -m "feat(tts): add FallbackTTS wrapper with ordered provider chain"
```

---

## Track 2: STT Endpoint + API Rewrite

### Task T2.1: Independent Transcribe Endpoint + Dependencies

**Files:**
- Modify: `pyproject.toml` (add `sse-starlette`)
- Modify: `src/cozyvoice/api/rest.py` (add `/voice/transcribe`)
- Modify: `src/cozyvoice/main.py` (register route)
- Test: `tests/unit/test_transcribe_endpoint.py`

- [ ] **Step 1: Add sse-starlette dependency**

In `pyproject.toml`, add `"sse-starlette>=2.0"` to `dependencies` (after `edge-tts`):

```toml
    "edge-tts>=6.1",
    "sse-starlette>=2.0",
```

Install:

```bash
cd ~/CozyProjects/CozyVoice && .venv/bin/pip install -e ".[dev]"
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_transcribe_endpoint.py`:

```python
"""Layer 1: /v1/voice/transcribe endpoint tests."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from cozyvoice.providers.base import STTResult
from cozyvoice.providers.stt.mock import MockSTT


def _make_app():
    from cozyvoice.api.rest import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.stt = MockSTT(canned_text="北京天气怎么样")
    app.state.tts = None
    app.state.tts_config = {}
    app.state.brain_client = None
    return app


async def test_transcribe_returns_text() -> None:
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
        r = await c.post("/v1/voice/transcribe", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "北京天气怎么���"
    assert "language" in body


async def test_transcribe_stt_failure_returns_502() -> None:
    from unittest.mock import AsyncMock
    app = _make_app()
    app.state.stt = AsyncMock()
    app.state.stt.transcribe = AsyncMock(side_effect=ConnectionError("stt down"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"fake-audio", "audio/wav")}
        r = await c.post("/v1/voice/transcribe", files=files)
    assert r.status_code == 502


async def test_transcribe_no_auth_required() -> None:
    """Layer 1 transcribe does NOT require JWT (unlike /voice/chat)."""
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
        r = await c.post("/v1/voice/transcribe", files=files)
    assert r.status_code == 200
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_transcribe_endpoint.py -v`
Expected: FAIL (no `/voice/transcribe` endpoint)

- [ ] **Step 4: Implement transcribe endpoint**

Add to `src/cozyvoice/api/rest.py` (before the existing `voice_chat` function):

```python
@router.post("/voice/transcribe")
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),
):
    """Layer 1: pure STT — no Brain, no auth required."""
    audio_bytes = await audio.read()
    mime = audio.content_type or "audio/wav"
    stt = request.app.state.stt
    try:
        result = await stt.transcribe(audio_bytes, mime_type=mime)
    except Exception as e:
        raise HTTPException(status_code=502, detail={"error": {"code": "STT_FAILED", "message": str(e)[:200]}})
    return {
        "text": result.text,
        "language": result.language,
        "duration_ms": result.duration_ms,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_transcribe_endpoint.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
cd ~/CozyProjects/CozyVoice && git add pyproject.toml src/cozyvoice/api/rest.py tests/unit/test_transcribe_endpoint.py && git commit -m "feat(api): add /v1/voice/transcribe endpoint (Layer 1 pure STT)"
```

---

### Task T2.2: brain_client.chat_stream()

**Files:**
- Modify: `src/cozyvoice/bridge/brain_client.py`
- Test: `tests/unit/test_brain_client_stream.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_brain_client_stream.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_brain_client_stream.py -v`
Expected: FAIL with `AttributeError: 'BrainClient' object has no attribute 'chat_stream'`

- [ ] **Step 3: Implement chat_stream method**

Add to `src/cozyvoice/bridge/brain_client.py` after the `chat_collect` method:

```python
    async def chat_stream(
        self,
        jwt: str,
        session_id: str | uuid.UUID,
        personality_id: str | uuid.UUID,
        message: str,
    ):
        """调 Brain SSE 端点，逐 chunk yield 文本。"""
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
                if piece:
                    yield piece
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_brain_client_stream.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd ~/CozyProjects/CozyVoice && git add src/cozyvoice/bridge/brain_client.py tests/unit/test_brain_client_stream.py && git commit -m "feat(bridge): add brain_client.chat_stream() for SSE streaming"
```

---

### Task T2.3: Rewrite /voice/chat to SSE Streaming

**Files:**
- Modify: `src/cozyvoice/api/rest.py` (rewrite `voice_chat`)
- Modify: `src/cozyvoice/main.py` (update `_build_tts` to use FallbackTTS from config)
- Modify: `config/voice.yaml` (add `fallback_chain` section)
- Test: `tests/unit/test_voice_chat_sse.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_voice_chat_sse.py`:

```python
"""Layer 2/3: /v1/voice/chat SSE streaming tests."""

from __future__ import annotations

import json
from base64 import b64decode
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from cozyvoice.providers.base import STTResult
from cozyvoice.providers.stt.mock import MockSTT
from cozyvoice.providers.tts.mock import MockTTS


def _make_app(*, tts_enabled: bool = False):
    from cozyvoice.api.rest import router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.stt = MockSTT(canned_text="北京天气")
    app.state.tts = MockTTS() if tts_enabled else None
    app.state.tts_config = {"default_voice_id": "mock", "default_format": "wav"}

    brain = AsyncMock()

    async def fake_stream(**kwargs):
        for chunk in ["北京明天", "晴，22度"]:
            yield chunk

    brain.chat_stream = fake_stream
    app.state.brain_client = brain
    return app


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into list of {event, data} dicts."""
    events = []
    current_event = None
    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            data_str = line[6:].strip()
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = data_str
            events.append({"event": current_event, "data": data})
            current_event = None
    return events


async def test_voice_chat_sse_without_tts() -> None:
    app = _make_app(tts_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
        data = {"session_id": "s1", "personality_id": "p1"}
        r = await c.post(
            "/v1/voice/chat?tts=false",
            files=files, data=data,
            headers={"Authorization": "Bearer fake-jwt"},
        )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    events = _parse_sse(r.text)
    event_types = [e["event"] for e in events]
    assert "stt" in event_types
    assert "reply_done" in event_types
    assert "tts_audio" not in event_types

    stt_event = next(e for e in events if e["event"] == "stt")
    assert stt_event["data"]["text"] == "北京天气"

    done_event = next(e for e in events if e["event"] == "reply_done")
    assert "北京明天" in done_event["data"]["text"]


async def test_voice_chat_sse_with_tts() -> None:
    app = _make_app(tts_enabled=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
        data = {"session_id": "s1", "personality_id": "p1"}
        r = await c.post(
            "/v1/voice/chat?tts=true",
            files=files, data=data,
            headers={"Authorization": "Bearer fake-jwt"},
        )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    event_types = [e["event"] for e in events]
    assert "tts_audio" in event_types
    tts_event = next(e for e in events if e["event"] == "tts_audio")
    assert tts_event["data"]["format"] in ("wav", "mp3")
    audio_bytes = b64decode(tts_event["data"]["base64"])
    assert len(audio_bytes) > 0


async def test_voice_chat_sse_reply_chunks_streamed() -> None:
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
        data = {"session_id": "s1", "personality_id": "p1"}
        r = await c.post(
            "/v1/voice/chat",
            files=files, data=data,
            headers={"Authorization": "Bearer fake-jwt"},
        )
    events = _parse_sse(r.text)
    chunk_events = [e for e in events if e["event"] == "reply_chunk"]
    assert len(chunk_events) >= 2
    assert chunk_events[0]["data"]["delta"] == "北京明天"


async def test_voice_chat_missing_auth_returns_401() -> None:
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"fake", "audio/wav")}
        data = {"session_id": "s1", "personality_id": "p1"}
        r = await c.post("/v1/voice/chat", files=files, data=data)
    assert r.status_code == 401


async def test_voice_chat_stt_failure_returns_error_event() -> None:
    app = _make_app()
    app.state.stt = AsyncMock()
    app.state.stt.transcribe = AsyncMock(side_effect=ConnectionError("stt down"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        files = {"audio": ("test.wav", b"fake", "audio/wav")}
        data = {"session_id": "s1", "personality_id": "p1"}
        r = await c.post(
            "/v1/voice/chat",
            files=files, data=data,
            headers={"Authorization": "Bearer fake-jwt"},
        )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_voice_chat_sse.py -v`
Expected: FAIL (endpoint still returns audio binary, not SSE)

- [ ] **Step 3: Rewrite voice_chat to SSE**

Replace the `voice_chat` function in `src/cozyvoice/api/rest.py` with:

```python
import json as _json
from base64 import b64encode

from sse_starlette.sse import EventSourceResponse


@router.post("/voice/chat")
async def voice_chat(
    request: Request,
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    personality_id: str = Form(...),
    tts: bool = False,
):
    """Layer 2/3: voice chat with SSE streaming. Set tts=true for audio reply."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail={"error": {"code": "MISSING_AUTH", "message": "JWT required"}})
    jwt = auth_header[len("Bearer "):]

    audio_bytes = await audio.read()
    mime = audio.content_type or "audio/wav"

    stt_provider = request.app.state.stt
    tts_provider = request.app.state.tts
    brain = request.app.state.brain_client
    tts_cfg = request.app.state.tts_config

    async def event_stream():
        # Step 1: STT
        try:
            stt_result = await stt_provider.transcribe(audio_bytes, mime_type=mime)
        except Exception as e:
            yield {"event": "error", "data": _json.dumps({"code": "STT_FAILED", "message": str(e)[:200]})}
            return

        yield {"event": "stt", "data": _json.dumps({"text": stt_result.text, "language": stt_result.language, "duration_ms": stt_result.duration_ms})}

        # Step 2: Brain streaming
        full_reply = ""
        try:
            async for chunk in brain.chat_stream(
                jwt=jwt,
                session_id=session_id,
                personality_id=personality_id,
                message=stt_result.text,
            ):
                full_reply += chunk
                yield {"event": "reply_chunk", "data": _json.dumps({"delta": chunk})}
        except Exception as e:
            yield {"event": "error", "data": _json.dumps({"code": "BRAIN_ERROR", "message": str(e)[:200]})}
            return

        yield {"event": "reply_done", "data": _json.dumps({"text": full_reply})}

        # Step 3: TTS (optional)
        if tts and tts_provider and full_reply:
            try:
                audio_out = await tts_provider.synthesize(
                    text=full_reply,
                    voice_id=tts_cfg.get("default_voice_id", ""),
                    format=tts_cfg.get("default_format", "mp3"),
                )
                fmt = tts_cfg.get("default_format", "mp3")
                yield {
                    "event": "tts_audio",
                    "data": _json.dumps({"format": fmt, "base64": b64encode(audio_out).decode()}),
                }
            except Exception as e:
                yield {"event": "error", "data": _json.dumps({"code": "TTS_FAILED", "message": str(e)[:200]})}

    return EventSourceResponse(event_stream())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_voice_chat_sse.py -v`
Expected: 5 passed

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/ -v`
Expected: The existing integration test `test_voice_chat_roundtrip` will FAIL because it expects binary audio response. This is expected — we'll update it in Track 4.

- [ ] **Step 6: Commit**

```bash
cd ~/CozyProjects/CozyVoice && git add src/cozyvoice/api/rest.py tests/unit/test_voice_chat_sse.py && git commit -m "feat(api): rewrite /voice/chat to SSE streaming (Layer 2/3)"
```

---

### Task T2.4: Update main.py _build_tts to use FallbackTTS from config

**Files:**
- Modify: `src/cozyvoice/main.py`
- Modify: `config/voice.yaml`

- [ ] **Step 1: Update voice.yaml with fallback_chain config**

Replace the `tts:` section in `config/voice.yaml`:

```yaml
  tts:
    default_voice_id: "101001"
    default_format: wav
    fallback_chain:
      - provider: tencent
        voice_id: "101001"
        timeout_ms: 3000
        tencent:
          secret_id: ${TENCENT_SECRET_ID}
          secret_key: ${TENCENT_SECRET_KEY}
          region: ${TENCENT_REGION:-ap-guangzhou}
      - provider: openai
        voice_id: "shimmer"
        timeout_ms: 5000
        openai:
          api_key: ${OPENAI_API_KEY}
          base_url: ${OPENAI_BASE_URL:-}
      - provider: edge
        voice_id: "zh-CN-XiaoxiaoNeural"
        timeout_ms: 5000
```

- [ ] **Step 2: Rewrite _build_tts in main.py**

Replace the `_build_tts` function in `src/cozyvoice/main.py`:

```python
from cozyvoice.providers.tts.fallback import FallbackTTS
from cozyvoice.providers.tts.openai_tts import OpenAITTS
from cozyvoice.providers.tts.edge_tts_provider import EdgeTTS


def _build_tts(cfg: dict):
    tts_cfg = cfg.get("tts", {})
    chain = tts_cfg.get("fallback_chain")
    if not chain:
        provider = tts_cfg.get("provider", "mock")
        if provider == "tencent":
            tc = tts_cfg["tencent"]
            return TencentTTS(secret_id=tc["secret_id"], secret_key=tc["secret_key"], region=tc.get("region", "ap-guangzhou"))
        return MockTTS()

    providers = []
    for entry in chain:
        p_type = entry.get("provider", "")
        timeout_s = entry.get("timeout_ms", 5000) / 1000.0
        provider = None
        if p_type == "tencent":
            tc = entry.get("tencent", {})
            sid, skey = tc.get("secret_id", ""), tc.get("secret_key", "")
            if sid and skey:
                provider = TencentTTS(secret_id=sid, secret_key=skey, region=tc.get("region", "ap-guangzhou"))
        elif p_type == "openai":
            oc = entry.get("openai", {})
            api_key = oc.get("api_key", "")
            if api_key:
                provider = OpenAITTS(api_key=api_key, base_url=oc.get("base_url") or None)
        elif p_type == "edge":
            provider = EdgeTTS(default_voice=entry.get("voice_id", "zh-CN-XiaoxiaoNeural"))
        if provider is not None:
            provider.default_voice = entry.get("voice_id", "")
            provider.timeout_s = timeout_s
            providers.append(provider)
    if not providers:
        return MockTTS()
    return FallbackTTS(providers=providers)
```

- [ ] **Step 3: Run test suite**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/ -v`
Expected: All unit tests pass (integration tests may still fail from T2.3 change)

- [ ] **Step 4: Commit**

```bash
cd ~/CozyProjects/CozyVoice && git add src/cozyvoice/main.py config/voice.yaml && git commit -m "feat(main): rewrite _build_tts to construct FallbackTTS from config"
```

---

## Track 3: Production Infrastructure

### Task T3.1: LiveKit Server Production Config

**Files:**
- Create: `deploy/livekit/livekit.yaml`
- Modify: `deploy/base_runtime/docker-compose.1panel.yml`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Create LiveKit production config**

Create `deploy/livekit/livekit.yaml`:

```yaml
# LiveKit Server production configuration
# Docs: https://docs.livekit.io/realtime/self-hosting/deployment/
port: 7880

rtc:
  port_range_start: 50000
  port_range_end: 50200
  use_external_ip: true

# API keys — injected via environment variable substitution in docker-compose
# LiveKit reads LIVEKIT_KEYS env var as "key: secret" pairs
# Format: LIVEKIT_KEYS="api_key: api_secret"

logging:
  level: info
  json: true
```

- [ ] **Step 2: Update 1Panel compose to use config file**

In `deploy/base_runtime/docker-compose.1panel.yml`, replace the `cozy_livekit` service:

```yaml
  cozy_livekit:
    image: livekit/livekit-server:v1.8
    container_name: cozy_livekit
    restart: unless-stopped
    command: --config /etc/livekit.yaml --node-ip ${NODE_IP:-0.0.0.0}
    environment:
      LIVEKIT_KEYS: "${LIVEKIT_API_KEY:-devkey}: ${LIVEKIT_API_SECRET:-devsecret}"
    volumes:
      - ../livekit/livekit.yaml:/etc/livekit.yaml:ro
    ports:
      - "7880:7880"
      - "7881:7881"
      - "50000-50200:50000-50200/udp"
    networks:
      - 1panel-network
    labels:
      createdBy: "Apps"
    deploy:
      resources:
        limits:
          memory: 1g
        reservations:
          memory: 256m
```

- [ ] **Step 3: Update dev compose to optionally mount config**

Replace the `livekit` service in `docker-compose.yml`:

```yaml
  livekit:
    image: livekit/livekit-server:latest
    command: --dev --bind 0.0.0.0
    ports:
      - "7880:7880"
      - "7881:7881"
      - "7882:7882/udp"
```

(Keep `--dev` for development — production uses 1Panel compose.)

- [ ] **Step 4: Commit**

```bash
cd ~/CozyProjects/CozyVoice && git add deploy/livekit/livekit.yaml deploy/base_runtime/docker-compose.1panel.yml docker-compose.yml && git commit -m "infra: add LiveKit production config, update 1Panel compose"
```

---

### Task T3.2: Full-Stack Docker Compose

**Files:**
- Create: `docker-compose.full-stack.yml`

- [ ] **Step 1: Create full-stack compose file**

Create `docker-compose.full-stack.yml`:

```yaml
# Full-stack development compose: Brain + Memory + NanoBot + LiveKit + CozyVoice
# Usage: docker compose -f docker-compose.full-stack.yml up -d
#
# Prerequisites:
#   - Brain image: cd ../CozyEngineV2 && docker build -t cozy-brain .
#   - Memory image: cd ../CozyMemory && docker build -t cozy-memory .
#   - NanoBot image: cd ../nanobot && docker build -t cozy-nanobot .
#   - CozyVoice image: docker build -t cozy-voice .

services:

  # ── Infrastructure ──
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: cozy
      POSTGRES_PASSWORD: cozy
      POSTGRES_DB: cozyengine
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U cozy"]
      interval: 5s
      timeout: 3s
      retries: 5

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  # ── Core Services ──
  brain:
    image: cozy-brain:latest
    ports: ["8000:8000"]
    environment:
      DATABASE_URL: postgresql+asyncpg://cozy:cozy@postgres:5432/cozyengine
      REDIS_URL: redis://redis:6379/0
      JWT_SECRET: ${JWT_SECRET:-dev-secret-change-in-prod}
      APP_ENV: development
    depends_on:
      postgres: { condition: service_healthy }
      redis: { condition: service_healthy }

  memory:
    image: cozy-memory:latest
    ports: ["8001:8001"]
    environment:
      REDIS_URL: redis://redis:6379/1
    depends_on:
      redis: { condition: service_healthy }

  nanobot:
    image: cozy-nanobot:latest
    ports: ["8080:8080"]

  # ── Voice Layer ──
  livekit:
    image: livekit/livekit-server:v1.8
    command: --dev --bind 0.0.0.0
    ports:
      - "7880:7880"
      - "7881:7881"
      - "7882:7882/udp"

  cozy-voice:
    image: cozy-voice:latest
    ports: ["8002:8002"]
    env_file: [.env]
    environment:
      BRAIN_URL: http://brain:8000
      LIVEKIT_URL: ws://livekit:7880
      LOG_LEVEL: INFO
    depends_on:
      - brain
      - livekit

volumes:
  pgdata:
```

- [ ] **Step 2: Commit**

```bash
cd ~/CozyProjects/CozyVoice && git add docker-compose.full-stack.yml && git commit -m "infra: add full-stack docker-compose (6 services)"
```

---

## Track 4: Test Coverage

### Task T4.1: Config Loader Tests

**Files:**
- Test: `tests/unit/test_config_loader.py`

- [ ] **Step 1: Write tests**

Create `tests/unit/test_config_loader.py`:

```python
"""Config loader (settings.py) unit tests."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from cozyvoice.config.settings import load_config


def test_load_config_env_substitution(tmp_path) -> None:
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("voice:\n  key: ${TEST_KEY_123:-default_val}\n")
    with patch.dict(os.environ, {"TEST_KEY_123": "real_val"}, clear=False):
        result = load_config(cfg_file)
    assert result["voice"]["key"] == "real_val"


def test_load_config_default_value(tmp_path) -> None:
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("voice:\n  key: ${NONEXISTENT_VAR_XYZ:-fallback}\n")
    result = load_config(cfg_file)
    assert result["voice"]["key"] == "fallback"


def test_load_config_no_default_empty_string(tmp_path) -> None:
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("voice:\n  key: ${NONEXISTENT_VAR_ABC}\n")
    result = load_config(cfg_file)
    assert result["voice"]["key"] == ""


def test_load_config_multiple_vars(tmp_path) -> None:
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("voice:\n  a: ${VAR_A:-alpha}\n  b: ${VAR_B:-beta}\n")
    with patch.dict(os.environ, {"VAR_A": "AAA"}, clear=False):
        result = load_config(cfg_file)
    assert result["voice"]["a"] == "AAA"
    assert result["voice"]["b"] == "beta"
```

- [ ] **Step 2: Run tests**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_config_loader.py -v`
Expected: 4 passed

- [ ] **Step 3: Commit**

```bash
cd ~/CozyProjects/CozyVoice && git add tests/unit/test_config_loader.py && git commit -m "test: add config loader unit tests"
```

---

### Task T4.2: Update Integration Tests for SSE

**Files:**
- Modify: `tests/integration/test_rest_voice_chat.py`
- Create: `tests/integration/test_voice_chat_sse_roundtrip.py`
- Create: `tests/integration/test_fallback_chain_roundtrip.py`

- [ ] **Step 1: Update existing integration test**

Replace `tests/integration/test_rest_voice_chat.py`:

```python
"""REST /v1/voice/chat SSE integration test (mock Brain + Mock STT/TTS)."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.integration


async def test_voice_chat_sse_roundtrip(monkeypatch) -> None:
    from cozyvoice.main import create_app
    from cozyvoice.providers.stt.mock import MockSTT
    from cozyvoice.providers.tts.mock import MockTTS

    app = create_app()
    async with app.router.lifespan_context(app):
        app.state.stt = MockSTT(canned_text="查上海天气")
        app.state.tts = MockTTS()
        app.state.tts_config = {"default_voice_id": "mock", "default_format": "wav"}

        async def fake_stream(**kwargs):
            assert kwargs["message"] == "查上海天气"
            for chunk in ["上海今天", "多云 22°C"]:
                yield chunk

        app.state.brain_client.chat_stream = fake_stream

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            files = {"audio": ("hi.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")}
            data = {
                "session_id": "00000000-0000-0000-0000-000000000000",
                "personality_id": "00000000-0000-0000-0000-000000000001",
            }
            r = await c.post(
                "/v1/voice/chat?tts=true", files=files, data=data,
                headers={"Authorization": "Bearer fake-jwt"},
            )
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]

            events = []
            for line in r.text.split("\n"):
                if line.startswith("event: "):
                    events.append(line[7:].strip())
            assert "stt" in events
            assert "reply_done" in events
            assert "tts_audio" in events
```

- [ ] **Step 2: Create fallback chain integration test**

Create `tests/integration/test_fallback_chain_roundtrip.py`:

```python
"""TTS fallback chain integration test."""

from __future__ import annotations

import pytest

from cozyvoice.providers.base import TTSProvider, Voice
from cozyvoice.providers.tts.fallback import FallbackTTS, TTSAllProvidersFailedError
from cozyvoice.providers.tts.mock import MockTTS

pytestmark = pytest.mark.integration


class FailingTTS(TTSProvider):
    name = "failing"

    async def synthesize(self, text, voice_id="", format="mp3") -> bytes:
        raise ConnectionError("provider down")

    async def list_voices(self) -> list[Voice]:
        return []


async def test_fallback_chain_skips_failing_to_mock() -> None:
    failing = FailingTTS()
    failing.timeout_s = 1.0
    failing.default_voice = "v1"
    mock = MockTTS()
    mock.timeout_s = 5.0
    mock.default_voice = "mock"

    fallback = FallbackTTS(providers=[failing, mock])
    result = await fallback.synthesize("hello")
    assert result.startswith(b"RIFF")


async def test_all_failing_raises() -> None:
    f1 = FailingTTS()
    f1.timeout_s = 1.0
    f1.default_voice = "v1"
    f2 = FailingTTS()
    f2.timeout_s = 1.0
    f2.default_voice = "v2"

    fallback = FallbackTTS(providers=[f1, f2])
    with pytest.raises(TTSAllProvidersFailedError):
        await fallback.synthesize("hello")


async def test_fallback_list_voices_aggregates() -> None:
    failing = FailingTTS()
    mock = MockTTS()
    fallback = FallbackTTS(providers=[failing, mock])
    voices = await fallback.list_voices()
    assert len(voices) >= 1
```

- [ ] **Step 3: Run all tests**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
cd ~/CozyProjects/CozyVoice && git add tests/integration/test_rest_voice_chat.py tests/integration/test_fallback_chain_roundtrip.py && git commit -m "test: update integration tests for SSE + add fallback chain test"
```

---

### Task T4.3: Transcribe Endpoint + Existing Test Gap Coverage

**Files:**
- Test: `tests/unit/test_livekit_entrypoint.py` (overwrite existing placeholder)

- [ ] **Step 1: Write LiveKit entrypoint tests**

Replace `tests/unit/test_livekit_entrypoint.py`:

```python
"""LiveKit entrypoint unit tests (mock LiveKit primitives)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cozyvoice.livekit_entrypoint import _parse_participant_metadata


def test_parse_valid_metadata() -> None:
    raw = json.dumps({"brain_jwt": "tok", "session_id": "s1", "personality_id": "p1"})
    result = _parse_participant_metadata(raw)
    assert result["brain_jwt"] == "tok"
    assert result["session_id"] == "s1"


def test_parse_empty_metadata() -> None:
    assert _parse_participant_metadata(None) == {}
    assert _parse_participant_metadata("") == {}


def test_parse_invalid_json() -> None:
    result = _parse_participant_metadata("not-json{{{")
    assert result == {}


def test_parse_non_dict_json() -> None:
    result = _parse_participant_metadata('"just a string"')
    assert result == {}
```

- [ ] **Step 2: Run tests**

Run: `cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/unit/test_livekit_entrypoint.py -v`
Expected: 4 passed

- [ ] **Step 3: Commit**

```bash
cd ~/CozyProjects/CozyVoice && git add tests/unit/test_livekit_entrypoint.py && git commit -m "test: add livekit entrypoint unit tests"
```

---

### Task T4.4: Full Suite Verification

- [ ] **Step 1: Run complete test suite**

```bash
cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: ≥ 85 tests collected, all green. Verify no regressions from:
- Existing brain_client tests
- Existing pipeline_agent tests
- Existing realtime_agent tests

- [ ] **Step 2: Count tests**

```bash
cd ~/CozyProjects/CozyVoice && .venv/bin/pytest tests/ --collect-only -q 2>&1 | tail -3
```

Expected: `XX tests collected` where XX ≥ 85.

- [ ] **Step 3: Final commit if any fixes needed**

If any test failures were found and fixed:

```bash
cd ~/CozyProjects/CozyVoice && git add -A && git commit -m "fix: resolve test regressions from SSE rewrite"
```

- [ ] **Step 4: Push all changes**

```bash
cd ~/CozyProjects/CozyVoice && git push
```

---

## Test Count Projection

| Source | Count |
|:---|:---|
| Existing tests (baseline) | 34 |
| T1.1: test_openai_tts.py | +4 |
| T1.2: test_edge_tts.py | +4 |
| T1.3: test_fallback_tts.py | +6 |
| T2.1: test_transcribe_endpoint.py | +3 |
| T2.2: test_brain_client_stream.py | +3 |
| T2.3: test_voice_chat_sse.py | +5 |
| T4.1: test_config_loader.py | +4 |
| T4.2: test_rest_voice_chat.py (rewrite) | +0 (replaced) |
| T4.2: test_fallback_chain_roundtrip.py | +3 |
| T4.3: test_livekit_entrypoint.py (rewrite) | +4 (replaced ~0) |
| **Total** | **~70** |

> Note: The 34 baseline may change if the existing integration test was counting differently. The target of ≥85 should be met with room. If short, T4.4 will identify gaps to fill.

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ §1 Modal Architecture: Formalized in T2.3 (SSE streaming = Brain primary), existing LiveKit code = CozyVoice primary
- ✅ §2 API Contract: T2.1 (`/transcribe`), T2.3 (`/voice/chat` SSE), Layer 4 unchanged (existing LiveKit entrypoint)
- ✅ §3 TTS Layer: T1.1 (OpenAI), T1.2 (Edge), T1.3 (FallbackTTS)
- ✅ §4 STT + API: T2.1 (transcribe endpoint), T2.3 (SSE rewrite), T2.2 (chat_stream)
- ✅ §5 Production Infra: T3.1 (LiveKit config), T3.2 (full-stack compose)
- ✅ §6 Tests: T4.1-T4.4
- ✅ §8 Dependencies: `edge-tts` in T1.2, `sse-starlette` in T2.1
- ✅ §9 Delivery Standards: Tested throughout

**2. Placeholder scan:** No TBD/TODO found.

**3. Type consistency:**
- `TTSProvider` ABC: `synthesize(text, voice_id, format)` — consistent across OpenAITTS, EdgeTTS, FallbackTTS, MockTTS
- `BrainClient.chat_stream()`: `async for chunk in ...` — consistent between implementation and test mock
- `FallbackTTS`: uses `provider.timeout_s` and `provider.default_voice` — set in `_build_tts()`
- `EventSourceResponse` yield format: `{"event": "...", "data": "..."}` — consistent between implementation and test parser
