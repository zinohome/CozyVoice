"""FastAPI 入口 + lifespan 装配。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

from cozyvoice import __version__
from cozyvoice.api import rest, ws
from cozyvoice.bridge.brain_client import BrainClient
from cozyvoice.config.settings import load_config
from cozyvoice.providers.stt.mock import MockSTT
from cozyvoice.providers.stt.openai_whisper import OpenAIWhisperSTT
from cozyvoice.providers.tts.mock import MockTTS
from cozyvoice.providers.tts.tencent import TencentTTS

logger = logging.getLogger(__name__)

CFG_PATH = Path(__file__).parent.parent.parent / "config" / "voice.yaml"


def _build_stt(cfg: dict):
    stt_cfg = cfg.get("stt", {})
    provider = stt_cfg.get("provider", "mock")
    if provider == "openai_whisper":
        return OpenAIWhisperSTT(api_key=stt_cfg["openai"]["api_key"], model=stt_cfg["openai"].get("model", "whisper-1"))
    return MockSTT()


def _build_tts(cfg: dict):
    tts_cfg = cfg.get("tts", {})
    provider = tts_cfg.get("provider", "mock")
    if provider == "tencent":
        tc = tts_cfg["tencent"]
        return TencentTTS(secret_id=tc["secret_id"], secret_key=tc["secret_key"], region=tc.get("region", "ap-guangzhou"))
    return MockTTS()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("CozyVoice starting, v%s", __version__)
    cfg = load_config(CFG_PATH).get("voice", {})

    app.state.stt = _build_stt(cfg)
    app.state.tts = _build_tts(cfg)
    app.state.tts_config = cfg.get("tts", {})

    brain_cfg = cfg.get("brain", {})
    brain_client = BrainClient(
        base_url=brain_cfg.get("base_url", "http://localhost:8000"),
        timeout=float(brain_cfg.get("timeout", 60.0)),
    )
    await brain_client.startup()
    app.state.brain_client = brain_client

    yield

    await brain_client.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="CozyVoice", version=__version__, lifespan=lifespan)
    app.include_router(rest.router, prefix="/v1", tags=["voice"])
    app.include_router(ws.router, prefix="/v1", tags=["voice-ws"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
