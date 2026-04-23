"""LiveKit Agents Worker 入口 —— CozyVoice M5 γ。

设计要点
========

* 每次用户入房（`participant_connected`）触发一次 `handle_realtime_call` 协程。
* Participant.metadata（JSON 字符串）里携带：`brain_jwt` / `session_id` / `personality_id`。
* `LiveKit <-> Realtime`：
  - 下行（LiveKit → Realtime）：`rtc.AudioStream(track)` 迭代 `AudioFrameEvent`，
    把 `frame.data.tobytes()` 丢进 `audio_in` 队列；`realtime_agent._pump_in` 消费。
  - 上行（Realtime → LiveKit）：从 `audio_out` 队列取 PCM bytes，喂到
    `rtc.AudioSource` 并发布为 `LocalAudioTrack`。
* Participant disconnect 时：向 `audio_in` 放入 `None` sentinel、取消下行任务、
  关闭 AudioSource，从而让 `handle_realtime_call` 正常收尾（含调用
  Brain `/v1/chat/voice_summary`）。

运行
====

CLI：`python -m cozyvoice.livekit_entrypoint start`

环境变量
----

* `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`
* `OPENAI_API_KEY`
* `BRAIN_URL`（默认 `http://localhost:8000`）
* `OPENAI_REALTIME_MODEL`（默认 `gpt-4o-realtime-preview`）

参考
----
https://docs.livekit.io/agents/build/  — Agents 1.x
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli

from cozyvoice.pipeline_agent import run_cozy_pipeline
from cozyvoice.realtime_agent import RealtimeCallState, handle_realtime_call

logger = logging.getLogger("cozyvoice.livekit_entrypoint")

# Realtime 模式：
#   "openai"        — 走 OpenAI Realtime WS（默认，handle_realtime_call）
#   "cozy_pipeline" — 走自建 VAD+STT+BrainLLM+TTS 流水线（pipeline_agent）
COZYVOICE_REALTIME_MODE_ENV = "COZYVOICE_REALTIME_MODE"

# Realtime 默认 24kHz mono PCM16
REALTIME_SAMPLE_RATE = 24_000
REALTIME_NUM_CHANNELS = 1


def _parse_participant_metadata(raw: str | None) -> dict[str, str]:
    """从 participant.metadata 解析 brain_jwt / session_id / personality_id。"""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        logger.warning("participant metadata is not valid JSON: %s", raw[:120])
    return {}


async def _forward_track_to_queue(
    track: rtc.Track,
    audio_in: asyncio.Queue,
    *,
    sample_rate: int = REALTIME_SAMPLE_RATE,
    num_channels: int = REALTIME_NUM_CHANNELS,
) -> None:
    """把 LiveKit 订阅到的 audio track 重采样后推到 audio_in 队列。"""
    stream = rtc.AudioStream(
        track,  # type: ignore[arg-type]
        sample_rate=sample_rate,
        num_channels=num_channels,
    )
    try:
        async for event in stream:
            frame = event.frame
            await audio_in.put(bytes(frame.data))
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("audio forward task crashed")
    finally:
        # 通知 realtime_agent._pump_in 结束
        await audio_in.put(None)
        await stream.aclose()


async def _pump_queue_to_source(
    audio_out: asyncio.Queue,
    source: rtc.AudioSource,
    *,
    sample_rate: int = REALTIME_SAMPLE_RATE,
    num_channels: int = REALTIME_NUM_CHANNELS,
) -> None:
    """把 Realtime 回传的 PCM bytes 喂到 LocalAudioTrack。"""
    import numpy as np

    try:
        while True:
            pcm: bytes | None = await audio_out.get()
            if pcm is None:
                break
            if not pcm:
                continue
            # PCM16 → int16 frame
            samples = np.frombuffer(pcm, dtype=np.int16)
            if num_channels > 1:
                samples_per_channel = len(samples) // num_channels
            else:
                samples_per_channel = len(samples)
            frame = rtc.AudioFrame(
                data=samples.tobytes(),
                sample_rate=sample_rate,
                num_channels=num_channels,
                samples_per_channel=samples_per_channel,
            )
            await source.capture_frame(frame)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("audio out pump crashed")


async def _handle_participant(
    ctx: JobContext,
    participant: rtc.RemoteParticipant,
) -> None:
    """为一个 participant 建 Realtime 会话，串起 LiveKit ↔ OpenAI Realtime。"""
    meta = _parse_participant_metadata(participant.metadata)
    brain_jwt = meta.get("brain_jwt") or meta.get("jwt") or ""
    session_id = meta.get("session_id") or ""
    personality_id = meta.get("personality_id") or ""

    if not (brain_jwt and session_id and personality_id):
        logger.warning(
            "participant %s missing required metadata (brain_jwt/session_id/personality_id); skip",
            participant.identity,
        )
        return

    mode = (os.environ.get(COZYVOICE_REALTIME_MODE_ENV) or "openai").strip().lower()
    if mode == "cozy_pipeline":
        logger.info("using cozy_pipeline mode for participant %s", participant.identity)
        brain_url = os.environ.get("BRAIN_URL", "http://localhost:8000")
        voice = meta.get("voice") or None
        try:
            await run_cozy_pipeline(
                ctx,
                brain_url=brain_url,
                brain_jwt=brain_jwt,
                session_id=session_id,
                personality_id=personality_id,
                voice=voice,
            )
        except NotImplementedError as e:
            logger.error(
                "cozy_pipeline not yet implemented (%s); fallback to openai mode",
                e,
            )
            # 继续走 openai 分支作为 fallback
        else:
            return

    logger.info("using openai realtime mode for participant %s", participant.identity)
    audio_in: asyncio.Queue = asyncio.Queue(maxsize=128)
    audio_out: asyncio.Queue = asyncio.Queue(maxsize=128)

    # 1) 发布 Agent 出向 audio track
    source = rtc.AudioSource(REALTIME_SAMPLE_RATE, REALTIME_NUM_CHANNELS)
    agent_track = rtc.LocalAudioTrack.create_audio_track("cozyvoice-agent", source)
    publication = await ctx.room.local_participant.publish_track(
        agent_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    logger.info("published agent audio track sid=%s", publication.sid)

    # 2) 订阅 participant 的 audio track（通过事件注册 forwarder）
    forwarder_tasks: list[asyncio.Task] = []

    def _on_track_subscribed(
        track: rtc.Track,
        pub: rtc.RemoteTrackPublication,
        p: rtc.RemoteParticipant,
    ) -> None:
        if p.identity != participant.identity:
            return
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        logger.info("subscribed audio track from %s", p.identity)
        forwarder_tasks.append(
            asyncio.create_task(
                _forward_track_to_queue(track, audio_in),
                name=f"fwd-{p.identity}",
            )
        )

    ctx.room.on("track_subscribed", _on_track_subscribed)

    # 如果 participant 已经有 published track，主动挂起
    for pub in participant.track_publications.values():
        if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.track is not None:
            _on_track_subscribed(pub.track, pub, participant)  # type: ignore[arg-type]

    # 3) 出向 pump
    out_task = asyncio.create_task(
        _pump_queue_to_source(audio_out, source),
        name=f"out-{participant.identity}",
    )

    # 4) disconnect → cleanup
    disconnect_event = asyncio.Event()

    def _on_disconnected(p: rtc.RemoteParticipant) -> None:
        if p.identity == participant.identity:
            logger.info("participant %s disconnected", p.identity)
            disconnect_event.set()

    ctx.room.on("participant_disconnected", _on_disconnected)

    state = RealtimeCallState(
        jwt=brain_jwt,
        session_id=session_id,
        personality_id=personality_id,
    )

    brain_url = os.environ.get("BRAIN_URL", "http://localhost:8000")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    openai_base_url = os.environ.get("OPENAI_BASE_URL") or None
    openai_model = os.environ.get(
        "OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview"
    )

    call_task = asyncio.create_task(
        handle_realtime_call(
            audio_in=audio_in,
            audio_out=audio_out,
            state=state,
            brain_url=brain_url,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
        ),
        name=f"call-{participant.identity}",
    )

    try:
        done, _pending = await asyncio.wait(
            {call_task, asyncio.create_task(disconnect_event.wait())},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect_event.is_set() and not call_task.done():
            # 让 handle_realtime_call 走 voice_summary 收尾
            await audio_in.put(None)
            try:
                await asyncio.wait_for(call_task, timeout=10.0)
            except asyncio.TimeoutError:
                call_task.cancel()
    finally:
        for t in forwarder_tasks:
            t.cancel()
        await audio_out.put(None)
        out_task.cancel()
        for t in (*forwarder_tasks, out_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await source.aclose()
        except Exception:
            pass


async def entrypoint(ctx: JobContext) -> None:
    """Worker entrypoint —— 每个 LiveKit job 触发一次。"""
    logger.info("CozyVoice entrypoint joining room=%s", ctx.room.name if ctx.room else "<none>")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    tasks: dict[str, asyncio.Task] = {}

    def _on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        if participant.identity in tasks and not tasks[participant.identity].done():
            return
        logger.info("participant_connected: %s", participant.identity)
        tasks[participant.identity] = asyncio.create_task(
            _handle_participant(ctx, participant),
            name=f"participant-{participant.identity}",
        )

    ctx.room.on("participant_connected", _on_participant_connected)

    # 已在房间里的 participant 立刻挂起
    for p in ctx.room.remote_participants.values():
        _on_participant_connected(p)

    async def _cleanup() -> None:
        for t in tasks.values():
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    ctx.add_shutdown_callback(_cleanup)


def _main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # 内部健康检查/指标 HTTP server 默认 :8081，与 1Panel/Caddy 冲突；
    # 允许通过 LIVEKIT_AGENT_HTTP_PORT 覆盖（默认 8181）
    http_port = int(os.environ.get("LIVEKIT_AGENT_HTTP_PORT", "8181"))
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, port=http_port))


if __name__ == "__main__":
    _main()
