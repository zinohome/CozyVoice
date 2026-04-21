"""WS /v1/voice/stream - 半双工音频流。"""

from __future__ import annotations

import base64
import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket, token: str = Query(default="")) -> None:
    if not token:
        await websocket.close(code=4401, reason="missing token")
        return

    await websocket.accept()

    stt = websocket.app.state.stt
    tts = websocket.app.state.tts
    brain = websocket.app.state.brain_client
    tts_cfg = websocket.app.state.tts_config

    session_id: str | None = None
    personality_id: str | None = None
    audio_buf = bytearray()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "code": "BAD_JSON"}))
                continue

            mtype = msg.get("type")
            if mtype == "start":
                session_id = msg.get("session_id")
                personality_id = msg.get("personality_id")
                audio_buf.clear()
                await websocket.send_text(json.dumps({"type": "ready"}))
            elif mtype == "audio_chunk":
                try:
                    chunk = base64.b64decode(msg.get("data", ""))
                    audio_buf.extend(chunk)
                except Exception:
                    await websocket.send_text(json.dumps({"type": "error", "code": "BAD_AUDIO"}))
            elif mtype == "end":
                if not session_id or not personality_id:
                    await websocket.send_text(json.dumps({"type": "error", "code": "NO_SESSION"}))
                    continue
                try:
                    stt_result = await stt.transcribe(bytes(audio_buf), mime_type="audio/wav")
                    await websocket.send_text(json.dumps({"type": "transcript", "text": stt_result.text}))
                    reply = await brain.chat_collect(
                        jwt=token,
                        session_id=session_id,
                        personality_id=personality_id,
                        message=stt_result.text,
                    )
                    audio_out = await tts.synthesize(
                        text=reply,
                        voice_id=tts_cfg.get("default_voice_id", "101001"),
                        format=tts_cfg.get("default_format", "wav"),
                    )
                    await websocket.send_text(json.dumps({
                        "type": "audio_chunk",
                        "data": base64.b64encode(audio_out).decode(),
                    }))
                    await websocket.send_text(json.dumps({"type": "done"}))
                    audio_buf.clear()
                except Exception as e:
                    logger.exception("voice stream pipeline failed")
                    await websocket.send_text(json.dumps({"type": "error", "code": "INTERNAL", "message": str(e)[:200]}))
            else:
                await websocket.send_text(json.dumps({"type": "error", "code": "UNKNOWN_TYPE"}))
    except WebSocketDisconnect:
        pass
