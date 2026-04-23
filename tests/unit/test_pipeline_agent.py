"""Unit tests for `cozyvoice.pipeline_agent.run_cozy_pipeline`。

策略
----

`run_cozy_pipeline` 不应真连 LiveKit / OpenAI / Brain：mock 掉
`AgentSession` / `Agent` / `silero.VAD.load` / `lk_openai.STT` / `lk_openai.TTS`，
验证:
  1. 各组件正确构造（STT/TTS 传入了正确 env 来的 api_key/base_url）
  2. BrainLLMAdapter 作为 llm 传入
  3. 生命周期：session.start → wait disconnect → session.aclose
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cozyvoice import pipeline_agent
from cozyvoice.providers.brain_llm import BrainLLMAdapter


class _FakeRoom:
    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}

    def on(self, event: str, cb) -> None:
        self._handlers.setdefault(event, []).append(cb)

    def fire(self, event: str, *args) -> None:
        for cb in self._handlers.get(event, []):
            cb(*args)


def _make_ctx() -> SimpleNamespace:
    return SimpleNamespace(room=_FakeRoom())


@pytest.fixture
def patches(monkeypatch):
    """Mock plugins and capture kwargs."""
    captured: dict = {}

    fake_session = MagicMock(name="AgentSession_instance")
    fake_session.start = AsyncMock()
    fake_session.aclose = AsyncMock()

    def fake_session_ctor(*args, **kwargs):
        captured["session_kwargs"] = kwargs
        return fake_session

    fake_agent = MagicMock(name="Agent_instance")

    def fake_agent_ctor(*args, **kwargs):
        captured["agent_kwargs"] = kwargs
        return fake_agent

    vad_obj = MagicMock(name="vad")
    vad_load = MagicMock(return_value=vad_obj)

    def fake_stt(**kwargs):
        captured["stt_kwargs"] = kwargs
        return MagicMock(name="stt", **{"__class__": type("STT", (), {})})

    def fake_tts(**kwargs):
        captured["tts_kwargs"] = kwargs
        return MagicMock(name="tts")

    # Build fake modules
    fake_voice_mod = SimpleNamespace(Agent=fake_agent_ctor, AgentSession=fake_session_ctor)
    fake_silero = SimpleNamespace(VAD=SimpleNamespace(load=vad_load))
    fake_lk_openai = SimpleNamespace(STT=fake_stt, TTS=fake_tts)

    import sys

    monkeypatch.setitem(sys.modules, "livekit.agents.voice", fake_voice_mod)
    # Patch the plugins namespace (already real but we want to inject fakes)
    import livekit.plugins as _lp

    monkeypatch.setattr(_lp, "silero", fake_silero, raising=False)
    monkeypatch.setattr(_lp, "openai", fake_lk_openai, raising=False)

    return captured, fake_session


@pytest.mark.asyncio
async def test_run_cozy_pipeline_builds_all_components(monkeypatch, patches):
    captured, fake_session = patches
    monkeypatch.setenv("OPENAI_API_KEY", "proxy-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.delenv("OPENAI_REAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_REAL_BASE_URL", raising=False)

    ctx = _make_ctx()

    async def _driver():
        task = asyncio.create_task(
            pipeline_agent.run_cozy_pipeline(
                ctx,
                brain_url="http://brain.test",
                brain_jwt="jwt-abc",
                session_id="sess-1",
                personality_id="p-1",
                voice="nova",
                instructions="hi",
            )
        )
        # let it reach await disconnect_event
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        ctx.room.fire("participant_disconnected", SimpleNamespace(identity="u1"))
        await asyncio.wait_for(task, timeout=2.0)

    await _driver()

    # STT constructed with proxy key/base
    assert captured["stt_kwargs"]["api_key"] == "proxy-key"
    assert captured["stt_kwargs"]["base_url"] == "https://proxy.example/v1"
    assert captured["stt_kwargs"]["model"] == "whisper-1"

    # TTS falls back to proxy key when no REAL_* set
    assert captured["tts_kwargs"]["api_key"] == "proxy-key"
    assert captured["tts_kwargs"]["base_url"] == "https://proxy.example/v1"
    assert captured["tts_kwargs"]["voice"] == "nova"

    # AgentSession got BrainLLMAdapter as llm
    llm = captured["session_kwargs"]["llm"]
    assert isinstance(llm, BrainLLMAdapter)
    assert llm._brain_url == "http://brain.test"
    assert llm._brain_jwt == "jwt-abc"
    assert llm._session_id == "sess-1"
    assert llm._personality_id == "p-1"

    # lifecycle
    fake_session.start.assert_awaited_once()
    fake_session.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_cozy_pipeline_prefers_real_tts_env(monkeypatch, patches):
    captured, _ = patches
    monkeypatch.setenv("OPENAI_API_KEY", "proxy-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("OPENAI_REAL_API_KEY", "real-key")
    monkeypatch.setenv("OPENAI_REAL_BASE_URL", "https://api.openai.com/v1")

    ctx = _make_ctx()

    async def _driver():
        task = asyncio.create_task(
            pipeline_agent.run_cozy_pipeline(
                ctx,
                brain_url="http://brain.test",
                brain_jwt="jwt",
                session_id="s",
                personality_id="p",
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        ctx.room.fire("participant_disconnected", SimpleNamespace(identity="u1"))
        await asyncio.wait_for(task, timeout=2.0)

    await _driver()

    # STT: proxy
    assert captured["stt_kwargs"]["api_key"] == "proxy-key"
    assert captured["stt_kwargs"]["base_url"] == "https://proxy.example/v1"
    # TTS: real
    assert captured["tts_kwargs"]["api_key"] == "real-key"
    assert captured["tts_kwargs"]["base_url"] == "https://api.openai.com/v1"
    # default voice
    assert captured["tts_kwargs"]["voice"] == "alloy"


@pytest.mark.asyncio
async def test_run_cozy_pipeline_missing_openai_key_raises(monkeypatch, patches):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_REAL_API_KEY", raising=False)

    ctx = _make_ctx()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await pipeline_agent.run_cozy_pipeline(
            ctx,
            brain_url="http://brain.test",
            brain_jwt="jwt",
            session_id="s",
            personality_id="p",
        )


@pytest.mark.asyncio
async def test_run_cozy_pipeline_aclose_on_exception(monkeypatch, patches):
    """Even if start succeeds and disconnect fires, aclose must run."""
    captured, fake_session = patches
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_REAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_REAL_BASE_URL", raising=False)

    ctx = _make_ctx()

    async def _driver():
        task = asyncio.create_task(
            pipeline_agent.run_cozy_pipeline(
                ctx,
                brain_url="http://brain.test",
                brain_jwt="jwt",
                session_id="s",
                personality_id="p",
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        ctx.room.fire("participant_disconnected", SimpleNamespace(identity="u1"))
        await asyncio.wait_for(task, timeout=2.0)

    await _driver()
    fake_session.start.assert_awaited_once()
    fake_session.aclose.assert_awaited_once()
    # base_url for STT was empty env → should pass NOT_GIVEN sentinel
    from livekit.agents import NOT_GIVEN

    assert captured["stt_kwargs"]["base_url"] is NOT_GIVEN
    assert captured["tts_kwargs"]["base_url"] is NOT_GIVEN
