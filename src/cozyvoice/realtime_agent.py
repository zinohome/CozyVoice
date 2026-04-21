"""CozyVoice Realtime Voice Agent（基于 livekit-agents）。

独立进程：`python -m cozyvoice.realtime_agent`（通过 livekit-agents CLI 启动）
也可被其他入口导入 entrypoint 函数。

职责：
  1. 加入 LiveKit room，收到 participant 的 audio track
  2. 向 Brain 拉 voice context（人格+画像+记忆+工具）
  3. 建 OpenAI Realtime session，注入 instructions + tools
  4. 双向 pipe 音频（LiveKit ↔ Realtime）
  5. 拦截 function_call 事件 → 调 Brain /v1/tool_proxy → 结果回注 Realtime
  6. 离开时收集 turn transcript → POST /v1/chat/voice_summary
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field

from cozyvoice.bridge.brain_client import BrainClient
from cozyvoice.providers.realtime.openai_realtime import OpenAIRealtimeProvider

logger = logging.getLogger(__name__)


@dataclass
class RealtimeCallState:
    """单次通话的会话状态。"""

    jwt: str
    session_id: str
    personality_id: str
    turns: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)


async def handle_realtime_call(
    *,
    audio_in: asyncio.Queue,
    audio_out: asyncio.Queue,
    state: RealtimeCallState,
    brain_url: str,
    openai_api_key: str,
    openai_model: str = "gpt-4o-realtime-preview",
) -> None:
    """驱动一次完整的 Realtime 通话。参数 audio_in/out 是 asyncio.Queue 供上层（LiveKit room）喂帧。"""

    brain = BrainClient(base_url=brain_url)
    await brain.startup()

    try:
        # 1) 拉 context
        try:
            ctx = await brain.fetch_voice_context(
                jwt=state.jwt,
                session_id=state.session_id,
                personality_id=state.personality_id,
            )
        except Exception as e:
            logger.exception("fetch_voice_context failed: %s", e)
            ctx = {"system_prompt": "你是一位友好的助手。", "voice_id": "alloy", "allowed_tools": []}

        instructions = (ctx.get("system_prompt") or "") + "\n\n"
        if ctx.get("profile_context"):
            instructions += f"## 关于当前用户\n{ctx['profile_context']}\n\n"
        if ctx.get("memory_summary"):
            instructions += f"## 历史摘要\n{ctx['memory_summary']}"

        voice = ctx.get("voice_id") or "alloy"
        tools = ctx.get("allowed_tools") or []

        # 2) 建 Realtime
        provider = OpenAIRealtimeProvider(api_key=openai_api_key, model=openai_model)
        rt_session = await provider.open_session(instructions=instructions, voice=voice, tools=tools)

        try:
            # 3) 启动两条任务
            await asyncio.gather(
                _pump_in(audio_in, rt_session),
                _pump_events(rt_session, audio_out, brain, state),
                return_exceptions=False,
            )
        finally:
            await rt_session.close()

        # 4) 回写 summary
        try:
            await brain.voice_summary(
                jwt=state.jwt,
                session_id=state.session_id,
                turns=state.turns,
                tool_calls=state.tool_calls,
            )
        except Exception:
            logger.exception("voice_summary failed")

    finally:
        await brain.shutdown()


async def _pump_in(audio_in: asyncio.Queue, rt_session) -> None:
    """把 LiveKit room 输入帧送到 Realtime。"""
    while True:
        chunk = await audio_in.get()
        if chunk is None:  # sentinel 表示结束
            break
        await rt_session.send_audio(chunk)


async def _pump_events(rt_session, audio_out: asyncio.Queue, brain: BrainClient,
                       state: RealtimeCallState) -> None:
    """处理 Realtime 事件：音频→队列；function_call→Brain tool_proxy；transcript→state。"""
    pending_calls: dict[str, dict] = {}  # call_id → {"name", "args_buf"}
    current_assistant_text = ""

    async for event in rt_session.receive_events():
        etype = event.get("type", "")

        if etype == "response.audio.delta":
            audio_b64 = event.get("delta") or ""
            if audio_b64:
                await audio_out.put(base64.b64decode(audio_b64))

        elif etype == "response.audio_transcript.delta":
            current_assistant_text += event.get("delta", "")
        elif etype == "response.audio_transcript.done":
            if current_assistant_text:
                state.turns.append({"role": "assistant", "content": current_assistant_text})
                current_assistant_text = ""

        elif etype == "conversation.item.input_audio_transcription.completed":
            user_txt = event.get("transcript") or ""
            if user_txt:
                state.turns.append({"role": "user", "content": user_txt})

        elif etype == "response.function_call_arguments.delta":
            call_id = event.get("call_id")
            name = event.get("name")
            delta = event.get("delta", "")
            if call_id:
                slot = pending_calls.setdefault(call_id, {"name": name, "args_buf": ""})
                if name:
                    slot["name"] = name
                slot["args_buf"] += delta

        elif etype == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            name = event.get("name")
            full_args = event.get("arguments") or (pending_calls.get(call_id, {}).get("args_buf", ""))
            tool_name = name or pending_calls.get(call_id, {}).get("name") or "unknown"
            try:
                args_dict = json.loads(full_args) if isinstance(full_args, str) else (full_args or {})
            except json.JSONDecodeError:
                args_dict = {"raw": full_args}

            try:
                result = await brain.tool_proxy(
                    jwt=state.jwt,
                    session_id=state.session_id,
                    tool_name=tool_name,
                    arguments=args_dict,
                )
                state.tool_calls.append({
                    "tool_name": tool_name,
                    "arguments": args_dict,
                    "result": result.get("result") or result,
                    "duration_ms": 0,
                })
                await rt_session.submit_tool_result(call_id, json.dumps(result.get("result") or result))
            except Exception as e:
                logger.exception("tool_proxy failed: %s", e)
                state.tool_calls.append({
                    "tool_name": tool_name,
                    "arguments": args_dict,
                    "result": {"error": str(e)[:200]},
                    "duration_ms": 0,
                })
                await rt_session.submit_tool_result(call_id, json.dumps({"error": "tool_proxy failed"}))
            pending_calls.pop(call_id, None)

        elif etype == "response.done":
            continue

        elif etype == "error":
            logger.warning("Realtime error: %s", event)

    return
