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

    def fake_dg_stt(**kwargs):
        captured["dg_stt_kwargs"] = kwargs
        return MagicMock(name="dg_stt")

    def fake_dg_tts(**kwargs):
        captured["dg_tts_kwargs"] = kwargs
        return MagicMock(name="dg_tts")

    fake_lk_deepgram = SimpleNamespace(STT=fake_dg_stt, TTS=fake_dg_tts)

    import sys

    monkeypatch.setitem(sys.modules, "livekit.agents.voice", fake_voice_mod)
    # Ensure `from livekit.plugins import deepgram` resolves to the fake
    # (livekit-plugins-deepgram may not be installed in CI)
    monkeypatch.setitem(sys.modules, "livekit.plugins.deepgram", fake_lk_deepgram)
    # Patch the plugins namespace (already real but we want to inject fakes)
    import livekit.plugins as _lp

    monkeypatch.setattr(_lp, "silero", fake_silero, raising=False)
    monkeypatch.setattr(_lp, "openai", fake_lk_openai, raising=False)
    monkeypatch.setattr(_lp, "deepgram", fake_lk_deepgram, raising=False)

    return captured, fake_session


@pytest.mark.asyncio
async def test_run_cozy_pipeline_builds_all_components(monkeypatch, patches):
    captured, fake_session = patches
    monkeypatch.setenv("OPENAI_API_KEY", "proxy-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.delenv("OPENAI_REAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_REAL_BASE_URL", raising=False)
    # Force OpenAI backend for legacy expectations
    monkeypatch.setenv("COZYVOICE_STT_BACKEND", "openai")
    monkeypatch.setenv("COZYVOICE_TTS_BACKEND", "openai")
    monkeypatch.delenv("COZYVOICE_STT_MODEL", raising=False)
    monkeypatch.delenv("COZYVOICE_STT_LANGUAGE", raising=False)

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
        for _ in range(5):
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        ctx.room.fire("participant_disconnected", SimpleNamespace(identity="u1"))
        await asyncio.wait_for(task, timeout=3.0)

    await _driver()

    # STT constructed with proxy key/base
    assert captured["stt_kwargs"]["api_key"] == "proxy-key"
    assert captured["stt_kwargs"]["base_url"] == "https://proxy.example/v1"
    # default STT model: gpt-4o-mini-transcribe（流式，首字节比 whisper-1 快）
    assert captured["stt_kwargs"]["model"] == "gpt-4o-mini-transcribe"
    assert captured["stt_kwargs"]["language"] == "zh"

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
    monkeypatch.setenv("COZYVOICE_STT_BACKEND", "openai")
    monkeypatch.setenv("COZYVOICE_TTS_BACKEND", "openai")
    monkeypatch.delenv("COZYVOICE_TTS_VOICE", raising=False)

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
        for _ in range(5):
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        ctx.room.fire("participant_disconnected", SimpleNamespace(identity="u1"))
        await asyncio.wait_for(task, timeout=3.0)

    await _driver()

    # STT: proxy
    assert captured["stt_kwargs"]["api_key"] == "proxy-key"
    assert captured["stt_kwargs"]["base_url"] == "https://proxy.example/v1"
    # TTS: real
    assert captured["tts_kwargs"]["api_key"] == "real-key"
    assert captured["tts_kwargs"]["base_url"] == "https://api.openai.com/v1"
    # default voice
    # default voice: shimmer（友好女声，中文合成稳定；可通过 env COZYVOICE_TTS_VOICE 覆盖）
    assert captured["tts_kwargs"]["voice"] == "shimmer"


@pytest.mark.asyncio
async def test_run_cozy_pipeline_missing_openai_key_raises(monkeypatch, patches):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_REAL_API_KEY", raising=False)
    monkeypatch.setenv("COZYVOICE_STT_BACKEND", "openai")
    monkeypatch.setenv("COZYVOICE_TTS_BACKEND", "openai")

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
    monkeypatch.setenv("COZYVOICE_STT_BACKEND", "openai")
    monkeypatch.setenv("COZYVOICE_TTS_BACKEND", "openai")

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
        for _ in range(5):
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        ctx.room.fire("participant_disconnected", SimpleNamespace(identity="u1"))
        await asyncio.wait_for(task, timeout=3.0)

    await _driver()
    fake_session.start.assert_awaited_once()
    fake_session.aclose.assert_awaited_once()
    # base_url for STT was empty env → should pass NOT_GIVEN sentinel
    from livekit.agents import NOT_GIVEN

    assert captured["stt_kwargs"]["base_url"] is NOT_GIVEN
    assert captured["tts_kwargs"]["base_url"] is NOT_GIVEN


# ---------- Deepgram backend ----------


async def _drive(ctx, **kwargs):
    """Helper: start pipeline, fire disconnect, await."""
    task = asyncio.create_task(pipeline_agent.run_cozy_pipeline(ctx, **kwargs))
    # Enough yields for the task to reach disconnect_event.wait()
    for _ in range(5):
        await asyncio.sleep(0)
    await asyncio.sleep(0.05)
    ctx.room.fire("participant_disconnected", SimpleNamespace(identity="u1"))
    await asyncio.wait_for(task, timeout=3.0)


@pytest.mark.asyncio
async def test_default_stt_backend_is_openai(monkeypatch, patches):
    """env 不设 COZYVOICE_STT_BACKEND → 默认走 OpenAI Whisper。"""
    captured, _ = patches
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-key")
    monkeypatch.setenv("OPENAI_API_KEY", "proxy-key")
    monkeypatch.delenv("COZYVOICE_STT_BACKEND", raising=False)
    monkeypatch.delenv("COZYVOICE_TTS_BACKEND", raising=False)
    monkeypatch.delenv("COZYVOICE_STT_LANGUAGE", raising=False)
    monkeypatch.delenv("COZYVOICE_STT_MODEL", raising=False)

    ctx = _make_ctx()
    await _drive(
        ctx,
        brain_url="http://brain.test",
        brain_jwt="jwt",
        session_id="s",
        personality_id="p",
    )

    # Default is OpenAI STT (not Deepgram)
    assert "stt_kwargs" in captured
    assert captured["stt_kwargs"]["api_key"] == "proxy-key"
    # Deepgram STT NOT constructed
    assert "dg_stt_kwargs" not in captured


@pytest.mark.asyncio
async def test_stt_backend_openai_preserves_legacy(monkeypatch, patches):
    """COZYVOICE_STT_BACKEND=openai 保持向后兼容。"""
    captured, _ = patches
    monkeypatch.setenv("OPENAI_API_KEY", "proxy-key")
    monkeypatch.setenv("COZYVOICE_STT_BACKEND", "openai")
    monkeypatch.setenv("COZYVOICE_TTS_BACKEND", "openai")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    ctx = _make_ctx()
    await _drive(
        ctx,
        brain_url="http://brain.test",
        brain_jwt="jwt",
        session_id="s",
        personality_id="p",
    )

    assert "stt_kwargs" in captured
    assert captured["stt_kwargs"]["model"] == "gpt-4o-mini-transcribe"
    # No Deepgram constructed
    assert "dg_stt_kwargs" not in captured
    assert "dg_tts_kwargs" not in captured


@pytest.mark.asyncio
async def test_deepgram_missing_key_raises(monkeypatch, patches):
    """显式 COZYVOICE_STT_BACKEND=deepgram，缺 DEEPGRAM_API_KEY → RuntimeError。"""
    monkeypatch.setenv("COZYVOICE_STT_BACKEND", "deepgram")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "proxy-key")

    ctx = _make_ctx()
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        await pipeline_agent.run_cozy_pipeline(
            ctx,
            brain_url="http://brain.test",
            brain_jwt="jwt",
            session_id="s",
            personality_id="p",
        )


@pytest.mark.asyncio
async def test_tts_backend_deepgram_with_english_uses_aura(monkeypatch, patches):
    """TTS=deepgram + language=en → Aura TTS（不回退）。"""
    captured, _ = patches
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-key")
    monkeypatch.setenv("COZYVOICE_STT_BACKEND", "deepgram")
    monkeypatch.setenv("COZYVOICE_TTS_BACKEND", "deepgram")
    monkeypatch.setenv("COZYVOICE_STT_LANGUAGE", "en")
    monkeypatch.setenv("COZYVOICE_TTS_VOICE", "aura-2-zeus-en")

    ctx = _make_ctx()
    await _drive(
        ctx,
        brain_url="http://brain.test",
        brain_jwt="jwt",
        session_id="s",
        personality_id="p",
    )

    assert "dg_tts_kwargs" in captured
    assert captured["dg_tts_kwargs"]["model"] == "aura-2-zeus-en"
    assert captured["dg_tts_kwargs"]["api_key"] == "dg-key"
    # Did NOT fall back to OpenAI TTS
    assert "tts_kwargs" not in captured


@pytest.mark.asyncio
async def test_tts_backend_deepgram_falls_back_for_chinese(monkeypatch, patches):
    """TTS=deepgram + language=zh → Aura 不支持，自动回退 OpenAI TTS。"""
    captured, _ = patches
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-key")
    monkeypatch.setenv("OPENAI_API_KEY", "proxy-key")
    monkeypatch.setenv("COZYVOICE_STT_BACKEND", "deepgram")
    monkeypatch.setenv("COZYVOICE_TTS_BACKEND", "deepgram")
    monkeypatch.setenv("COZYVOICE_STT_LANGUAGE", "zh")
    monkeypatch.delenv("COZYVOICE_TTS_VOICE", raising=False)

    ctx = _make_ctx()
    await _drive(
        ctx,
        brain_url="http://brain.test",
        brain_jwt="jwt",
        session_id="s",
        personality_id="p",
    )

    # Fell back to OpenAI TTS
    assert "tts_kwargs" in captured
    assert captured["tts_kwargs"]["voice"] == "shimmer"
    assert "dg_tts_kwargs" not in captured
