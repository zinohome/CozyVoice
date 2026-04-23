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

    # Deepgram plugin 为可选（COZYVOICE_*_BACKEND=deepgram 时才需要）
    try:
        from livekit.plugins import deepgram as lk_deepgram
    except ImportError:  # pragma: no cover
        lk_deepgram = None  # type: ignore[assignment]

    # ---- 共用 OpenAI env（openai 后端 / Deepgram TTS 降级 均会用到）----
    # STT 走代理 key（Whisper 是普通 HTTP，oneapi 兼容）
    openai_stt_api_key = os.environ.get("OPENAI_API_KEY") or ""
    openai_stt_base_url = os.environ.get("OPENAI_BASE_URL") or ""
    # TTS 优先真 key，回落代理（/v1/audio/speech 代理大概率也 OK）
    openai_tts_api_key = (
        os.environ.get("OPENAI_REAL_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    openai_tts_base_url = (
        os.environ.get("OPENAI_REAL_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    )

    dg_api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()

    # ---- backend 选择 ----
    stt_backend = os.environ.get("COZYVOICE_STT_BACKEND", "openai").lower()
    tts_backend = os.environ.get("COZYVOICE_TTS_BACKEND", "deepgram").lower()
    stt_language = os.environ.get("COZYVOICE_STT_LANGUAGE", "zh")

    # ---- 前置校验 ----
    if stt_backend == "deepgram":
        if not dg_api_key:
            raise RuntimeError(
                "COZYVOICE_STT_BACKEND=deepgram but DEEPGRAM_API_KEY not set"
            )
        if lk_deepgram is None:
            raise RuntimeError(
                "COZYVOICE_STT_BACKEND=deepgram but livekit-plugins-deepgram not installed. "
                "Run `pip install -e .[pipeline]`."
            )
    else:
        # openai STT
        if not openai_stt_api_key:
            raise RuntimeError(
                "run_cozy_pipeline: OPENAI_API_KEY required for Whisper STT"
            )

    # Deepgram Aura 2（2026-01）支持 en/es/nl/fr/de/it/ja —— **不支持中文**。
    # 若 backend=deepgram 且语言是中文，自动降级 OpenAI TTS（shimmer）。
    # 详见：https://developers.deepgram.com/docs/tts-models
    aura_supports_language = stt_language.lower() in {
        "en", "es", "nl", "fr", "de", "it", "ja",
    }
    if tts_backend == "deepgram" and not aura_supports_language:
        logger.warning(
            "COZYVOICE_TTS_BACKEND=deepgram but Aura does not support language=%s; "
            "falling back to OpenAI TTS. Set COZYVOICE_TTS_BACKEND=openai to silence.",
            stt_language,
        )
        tts_backend = "openai"

    if tts_backend == "deepgram":
        if not dg_api_key:
            raise RuntimeError(
                "COZYVOICE_TTS_BACKEND=deepgram but DEEPGRAM_API_KEY not set"
            )
        if lk_deepgram is None:
            raise RuntimeError(
                "COZYVOICE_TTS_BACKEND=deepgram but livekit-plugins-deepgram not installed."
            )
    else:
        if not openai_tts_api_key:
            raise RuntimeError(
                "run_cozy_pipeline: OPENAI_REAL_API_KEY or OPENAI_API_KEY required for TTS"
            )

    try:
        vad = silero.VAD.load()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"silero VAD load failed: {e}") from e

    # ---- STT 构造 ----
    if stt_backend == "deepgram":
        # Deepgram 中文支持错位：Nova-3 只 GA 了 zh-HK（粤语），普通话要用 Nova-2。
        # 规则：中文（zh*）→ 默认 nova-2 + zh-CN；其他语言/multi → 默认 nova-3。
        lang_lower = stt_language.lower()
        is_chinese = lang_lower.startswith("zh")
        default_model = "nova-2" if is_chinese else "nova-3"
        dg_stt_model = os.environ.get("COZYVOICE_STT_MODEL", default_model)
        # Nova-2 要求显式语种码；把裸 "zh" 归一成 "zh-CN"
        dg_language = "zh-CN" if lang_lower == "zh" else stt_language
        try:
            stt = lk_deepgram.STT(
                api_key=dg_api_key,
                model=dg_stt_model,
                language=dg_language,
                interim_results=True,
                smart_format=True,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"deepgram STT init failed: {e}") from e
    else:
        # OpenAI：默认 gpt-4o-mini-transcribe（流式，首字节比 whisper-1 快 ~1s）
        # 通过 COZYVOICE_STT_MODEL 可回落 whisper-1
        openai_stt_model = os.environ.get(
            "COZYVOICE_STT_MODEL", "gpt-4o-mini-transcribe"
        )
        try:
            stt = lk_openai.STT(
                model=openai_stt_model,
                api_key=openai_stt_api_key,
                base_url=_optional(openai_stt_base_url),
                # 强制中文：gpt-4o-transcribe 系列也看 language hint
                language=stt_language if stt_language != "multi" else "zh",
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"openai STT init failed: {e}") from e

    llm = BrainLLMAdapter(
        brain_url=brain_url,
        brain_jwt=brain_jwt,
        session_id=session_id,
        personality_id=personality_id,
    )

    # ---- TTS 构造 ----
    if tts_backend == "deepgram":
        # Aura（仅英/西/欧洲/日语）——到这里说明语言受支持
        aura_voice = (
            voice
            or os.environ.get("COZYVOICE_TTS_VOICE")
            or "aura-2-asteria-en"
        )
        try:
            tts = lk_deepgram.TTS(
                api_key=dg_api_key,
                model=aura_voice,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"deepgram TTS init failed: {e}") from e
    else:
        # TTS：voice 参数优先级 personality.voice > env > 硬编码 "shimmer"
        # shimmer = 友好女声，中文合成最稳定（alloy 中性但个别中文字发音偏男）
        # 中文有声调，voice 切换会让用户感觉"一会儿男一会儿女"——固定 shimmer 避免
        openai_tts_voice = voice or os.environ.get("COZYVOICE_TTS_VOICE", "shimmer")
        try:
            tts = lk_openai.TTS(
                voice=openai_tts_voice,
                api_key=openai_tts_api_key,
                base_url=_optional(openai_tts_base_url),
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"openai TTS init failed: {e}") from e

    # LiveKit Agents 1.5.6：allow_interruptions/min_interruption_duration 已 deprecated，
    # 必须用新 `turn_handling` dict API，否则那两个参数被 warning 后忽略。
    # 关键配置：
    # - interruption.enabled=True：允许用户打断 agent
    # - discard_audio_if_uninterruptible=False：agent 说话时用户音频不丢弃（排队而非忽略）
    # - endpointing.min_delay=0.3：用户停顿 300ms 就判定说完（默认 0.5s 偏慢）
    # - preemptive_generation.enabled=True：LLM 可在 partial STT 时就开始，砍首字
    session = AgentSession(
        vad=vad, stt=stt, llm=llm, tts=tts,
        turn_handling={
            "interruption": {
                "enabled": True,
                "discard_audio_if_uninterruptible": False,
                "min_duration": 0.3,
            },
            "endpointing": {"min_delay": 0.3, "max_delay": 2.0},
            "preemptive_generation": {"enabled": True},
        },
    )

    # 订阅转写事件，日志里记录"用户说了什么 / agent 回什么"
    @session.on("user_input_transcribed")
    def _on_user_transcript(ev: Any) -> None:
        try:
            text = getattr(ev, "transcript", "") or ""
            is_final = getattr(ev, "is_final", None)
            if is_final:
                logger.info("USER ▸ %r (final)", text)
            elif text:
                logger.debug("user partial: %r", text)
        except Exception:  # noqa: BLE001
            logger.exception("user transcript log failed")

    @session.on("conversation_item_added")
    def _on_conv_item(ev: Any) -> None:
        try:
            item = getattr(ev, "item", None)
            if item is None:
                return
            role = getattr(item, "role", "?")
            content = getattr(item, "text_content", None) or ""
            if content and role == "assistant":
                logger.info("AGENT ▸ %r", content[:300])
            elif content and role == "user":
                logger.info("USER  ▸ %r", content[:300])
        except Exception:  # noqa: BLE001
            logger.exception("conversation_item log failed")

    # 订阅 metrics 事件，精确打印 STT/LLM/TTS 各自首字节延迟
    # 便于定位"10s 黑洞"在哪（STT / LLM / TTS / 还是别的）
    from livekit.agents import metrics as _lk_metrics

    @session.on("metrics_collected")
    def _on_metrics(ev: Any) -> None:
        try:
            m = ev.metrics
            # 不同 metric 类型字段不同，统一打关键时延字段
            name = type(m).__name__
            fields = {}
            for attr in ("ttfb", "duration", "audio_duration", "streamed", "end_of_utterance_delay", "on_interrupted"):
                v = getattr(m, attr, None)
                if v is not None:
                    fields[attr] = round(v, 3) if isinstance(v, float) else v
            if fields:
                logger.info("metrics[%s] %s", name, fields)
        except Exception:  # noqa: BLE001
            logger.exception("metrics log failed")
    # 中文优先指令：Brain 侧 personality 若无明确语言偏好，这里兜底让 Agent 走中文
    default_instructions = (
        "你是一个中文语音助手。请始终用**简体中文**回复用户，语气自然、口语化、简短。"
        "不要使用英文单词或夹带英文句子（除非是专有名词）。"
    )
    agent = Agent(instructions=instructions or default_instructions)

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
