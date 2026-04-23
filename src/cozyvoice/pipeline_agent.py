"""自建 Realtime Pipeline（VAD → STT → Brain LLM → TTS）骨架。

现状（M5.5 骨架）
=================

* `BrainLLMAdapter` 已就绪（`providers/brain_llm.py`），可独立单测。
* **livekit-agents VoicePipelineAgent / AgentSession 依赖的 plugin 包未安装**：
  - `livekit-plugins-silero`（VAD）
  - `livekit-plugins-openai`（或 deepgram）STT
  - `livekit-plugins-openai` TTS（或包装现有 TencentTTSProvider）
  当前 `.venv` 里只有 `livekit.agents` + `livekit.rtc`，**没有 `livekit.plugins.*`**。
* 因此 `run_cozy_pipeline` 目前抛 `NotImplementedError`；调用方改回 `openai` 模式即可
  跑现有 Realtime。

启用步骤（TODO）
================

1. `pyproject.toml` 增加 extras：
   ```
   [project.optional-dependencies]
   pipeline = [
       "livekit-plugins-silero>=0.7",
       "livekit-plugins-openai>=0.10",
       # or deepgram/azure/elevenlabs 等
   ]
   ```
   `pip install -e ".[pipeline]"` 装好。

2. 打开下面的 import 并实现 `run_cozy_pipeline`：
   ```python
   from livekit.agents.voice import Agent, AgentSession
   from livekit.plugins import silero, openai as lk_openai

   async def run_cozy_pipeline(ctx, *, brain_url, brain_jwt, session_id,
                               personality_id, voice, instructions=""):
       await ctx.wait_for_participant()
       session = AgentSession(
           vad=silero.VAD.load(),
           stt=lk_openai.STT(model="whisper-1"),
           llm=BrainLLMAdapter(
               brain_url=brain_url,
               brain_jwt=brain_jwt,
               session_id=session_id,
               personality_id=personality_id,
           ),
           tts=lk_openai.TTS(voice=voice or "alloy"),
       )
       agent = Agent(instructions=instructions)
       await session.start(agent, room=ctx.room)
       # cleanup：disconnect 事件 → session.aclose()；见 livekit_entrypoint 同款套路
   ```

3. 腾讯 TTS 接入（放第二步）：写 `TencentTTSWrapper(livekit.agents.tts.TTS)`，
   `_run()` 里调用现有 `providers/tts/tencent.py` 的 `synthesize()` 返回 PCM
   frame。MVP 先用 `lk_openai.TTS`。

4. Brain `/v1/chat/voice` 契约落地后跑 e2e；chunk 粒度过粗/过细都影响 TTS
   流式体验，必要时在 adapter 侧做 sentence splitter。

edge cases 记录
----------------

* `chat_ctx.items` 为空：`BrainLLMAdapter` 不发任何 chunk（`_run` early return），
  LLMStream 自然 close，`AgentSession` 会把这轮当空回答处理。
* chat_ctx 只有 system/assistant、无 user：同上。
* Brain SSE 返回 `{"choices":[{"delta":{}}]}`（仅心跳）：`_extract_delta_content`
  返回 None，被跳过。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("cozyvoice.pipeline_agent")


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
    """自建 Realtime Pipeline 入口（骨架）。

    Raises:
        NotImplementedError: livekit-plugins-* 依赖未安装，详见模块 docstring。
    """
    raise NotImplementedError(
        "run_cozy_pipeline: requires livekit-plugins-silero + livekit-plugins-openai "
        "(or equivalent STT/TTS plugins). Install via `pip install -e .[pipeline]` "
        "after adding the extras; then fill this function per module docstring. "
        "For now, use COZYVOICE_REALTIME_MODE=openai to route to handle_realtime_call."
    )


__all__ = ["run_cozy_pipeline"]
