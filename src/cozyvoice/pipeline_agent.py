"""自建 Realtime Pipeline（VAD → STT → Brain LLM → TTS）实装（M5.5）。

实施概览
========

* **VAD**：`livekit-plugins-silero` 的 `silero.VAD.load()`。
* **STT**：`livekit-plugins-openai` 的 `openai.STT(model="whisper-1")`。走代理 key
  （`OPENAI_API_KEY` + `OPENAI_BASE_URL`）—— Whisper 普通 HTTP，代理兼容。
* **LLM**：`BrainLLMAdapter` —— 走 Brain `/v1/chat/voice` SSE。
* **TTS**：`livekit-plugins-openai` 的 `openai.TTS`。优先用 `OPENAI_REAL_*`
  （OpenAI 官方 key / base），回落到 `OPENAI_*`（代理）。
* **生命周期**：`AgentSession.start()` 后监听 `participant_disconnected` 事件，
  收到后 `session.aclose()` 优雅收尾。

边界 / 注意
-----------

* `silero.VAD.load()` 会首次下载 ONNX 模型（~几 MB），冷启动慢一点。
* 若代理 key 不支持 `/v1/audio/speech`（TTS）或 `/v1/audio/transcriptions`
  （Whisper STT），对应 plugin 实例化不会立刻报错，会在第一次推理时抛。
  我们在构造阶段不做假设，交给运行时由 `AgentSession` 冒泡。
* `tools` 目前未接入（BrainLLMAdapter 走 Brain 端 tool 调度），保留形参。

edge cases
-----------

* `chat_ctx.items` 为空：`BrainLLMAdapter` 不发任何 chunk，`AgentSession` 当空回答。
* Brain SSE 仅心跳 `{"choices":[{"delta":{}}]}`：`_extract_delta_content` 返回
  None 被跳过。
* participant 直接强退：`participant_disconnected` 触发 event，session 走收尾。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from livekit.agents import NOT_GIVEN

from cozyvoice.providers.brain_llm import BrainLLMAdapter

logger = logging.getLogger("cozyvoice.pipeline_agent")


def _optional(value: str | None) -> Any:
    """把可能为空的 env 变量转为 NOT_GIVEN 或原值，匹配 livekit plugin NotGivenOr 签名。"""
    return value if value else NOT_GIVEN


async def run_cozy_pipeline(
    ctx: Any,  # livekit.agents.JobContext
    *,
    brain_url: str,
    brain_jwt: str,
    session_id: str,
    personality_id: str,
    voice: str | None = None,
    instructions: str = "",
    tools: list[Any] | None = None,
) -> None:
    """自建 Realtime Pipeline 入口。

    构造 `AgentSession(vad=silero, stt=lk_openai.STT, llm=BrainLLMAdapter, tts=lk_openai.TTS)`
    并等待 participant disconnect 后关闭。
    """
    # lazy import：没装 [pipeline] 时，其他模式（openai realtime）仍可跑
    try:
        from livekit.agents.voice import Agent, AgentSession
        from livekit.plugins import openai as lk_openai
        from livekit.plugins import silero
    except ImportError as e:  # pragma: no cover - 依赖缺失时走这里
        raise RuntimeError(
            "run_cozy_pipeline: livekit-plugins-silero / livekit-plugins-openai "
            "not installed. Run `pip install -e .[pipeline]`."
        ) from e

    # STT 走代理 key（Whisper 是普通 HTTP，oneapi 兼容）
    stt_api_key = os.environ.get("OPENAI_API_KEY") or ""
    stt_base_url = os.environ.get("OPENAI_BASE_URL") or ""
    # TTS 优先真 key，回落代理（/v1/audio/speech 代理大概率也 OK）
    tts_api_key = (
        os.environ.get("OPENAI_REAL_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    tts_base_url = (
        os.environ.get("OPENAI_REAL_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    )

    if not stt_api_key:
        raise RuntimeError(
            "run_cozy_pipeline: OPENAI_API_KEY required for Whisper STT"
        )
    if not tts_api_key:
        raise RuntimeError(
            "run_cozy_pipeline: OPENAI_REAL_API_KEY or OPENAI_API_KEY required for TTS"
        )

    try:
        vad = silero.VAD.load()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"silero VAD load failed: {e}") from e

    try:
        stt = lk_openai.STT(
            model="whisper-1",
            api_key=stt_api_key,
            base_url=_optional(stt_base_url),
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"openai STT init failed: {e}") from e

    llm = BrainLLMAdapter(
        brain_url=brain_url,
        brain_jwt=brain_jwt,
        session_id=session_id,
        personality_id=personality_id,
    )

    try:
        tts = lk_openai.TTS(
            voice=voice or "alloy",
            api_key=tts_api_key,
            base_url=_optional(tts_base_url),
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"openai TTS init failed: {e}") from e

    session = AgentSession(vad=vad, stt=stt, llm=llm, tts=tts)
    agent = Agent(instructions=instructions or "")

    # participant disconnect → 停 session
    disconnect_event = asyncio.Event()

    def _on_disconnected(p: Any) -> None:
        logger.info(
            "cozy_pipeline: participant_disconnected %s",
            getattr(p, "identity", "?"),
        )
        disconnect_event.set()

    room = getattr(ctx, "room", None)
    if room is not None:
        room.on("participant_disconnected", _on_disconnected)

    logger.info(
        "cozy_pipeline: starting session (session_id=%s personality=%s voice=%s)",
        session_id,
        personality_id,
        voice or "alloy",
    )
    await session.start(agent, room=room)
    logger.info("cozy_pipeline: session started; awaiting disconnect")

    try:
        await disconnect_event.wait()
    finally:
        logger.info("cozy_pipeline: closing session")
        try:
            await session.aclose()
        except Exception:  # pragma: no cover
            logger.exception("cozy_pipeline: session.aclose() raised")


__all__ = ["run_cozy_pipeline"]
