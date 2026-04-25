"""POST /v1/voice/chat - multipart 上传音频 → 返回音频。"""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

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
):
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail={"error": {"code": "MISSING_AUTH", "message": "JWT required"}})
    jwt = auth_header[len("Bearer "):]

    audio_bytes = await audio.read()
    mime = audio.content_type or "audio/wav"

    stt = request.app.state.stt
    tts = request.app.state.tts
    brain = request.app.state.brain_client
    tts_cfg = request.app.state.tts_config

    try:
        stt_result = await stt.transcribe(audio_bytes, mime_type=mime)
    except Exception as e:
        raise HTTPException(status_code=502, detail={"error": {"code": "STT_FAILED", "message": str(e)[:200]}})

    try:
        reply_text = await brain.chat_collect(
            jwt=jwt,
            session_id=session_id,
            personality_id=personality_id,
            message=stt_result.text,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail={"error": {"code": "BRAIN_UNREACHABLE", "message": str(e)[:200]}})

    try:
        audio_out = await tts.synthesize(
            text=reply_text,
            voice_id=tts_cfg.get("default_voice_id", "101001"),
            format=tts_cfg.get("default_format", "wav"),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail={"error": {"code": "TTS_FAILED", "message": str(e)[:200], "transcript": reply_text}})

    fmt = tts_cfg.get("default_format", "wav")
    mime_out = {"wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "audio/pcm"}[fmt]

    return Response(
        content=audio_out,
        media_type=mime_out,
        headers={
            "X-Brain-Transcript-In": quote(stt_result.text[:500]),
            "X-Brain-Transcript-Out": quote(reply_text[:500]),
        },
    )
