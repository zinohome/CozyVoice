"""POST /v1/voice/ endpoints — transcribe + chat (SSE)."""

from __future__ import annotations

import json as _json
from base64 import b64encode
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from sse_starlette.sse import EventSourceResponse

router = APIRouter()


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
