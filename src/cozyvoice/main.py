"""FastAPI 入口 + lifespan 装配。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from cozyvoice import __version__
from cozyvoice.api import rest, ws
from cozyvoice.middleware.rate_limit import RateLimiter
from cozyvoice.bridge.brain_client import BrainClient
from cozyvoice.config.settings import load_config
from cozyvoice.providers.stt.mock import MockSTT
from cozyvoice.providers.stt.openai_whisper import OpenAIWhisperSTT
from cozyvoice.providers.tts.mock import MockTTS
from cozyvoice.providers.tts.tencent import TencentTTS
from cozyvoice.providers.tts.fallback import FallbackTTS
from cozyvoice.providers.tts.openai_tts import OpenAITTS
from cozyvoice.providers.tts.edge_tts_provider import EdgeTTS

logger = logging.getLogger(__name__)

CFG_PATH = Path(__file__).parent.parent.parent / "config" / "voice.yaml"


def _build_stt(cfg: dict):
    stt_cfg = cfg.get("stt", {})
    provider = stt_cfg.get("provider", "mock")
    if provider == "openai_whisper":
        return OpenAIWhisperSTT(
            api_key=stt_cfg["openai"]["api_key"],
            model=stt_cfg["openai"].get("model", "whisper-1"),
            base_url=stt_cfg["openai"].get("base_url") or None,
        )
    return MockSTT()


def _build_tts(cfg: dict):
    tts_cfg = cfg.get("tts", {})
    chain = tts_cfg.get("fallback_chain")
    if not chain:
        provider = tts_cfg.get("provider", "mock")
        if provider == "tencent":
            tc = tts_cfg["tencent"]
            return TencentTTS(secret_id=tc["secret_id"], secret_key=tc["secret_key"], region=tc.get("region", "ap-guangzhou"))
        return MockTTS()

    providers = []
    for entry in chain:
        p_type = entry.get("provider", "")
        timeout_s = entry.get("timeout_ms", 5000) / 1000.0
        provider = None
        if p_type == "tencent":
            tc = entry.get("tencent", {})
            sid, skey = tc.get("secret_id", ""), tc.get("secret_key", "")
            if sid and skey:
                provider = TencentTTS(secret_id=sid, secret_key=skey, region=tc.get("region", "ap-guangzhou"))
        elif p_type == "openai":
            oc = entry.get("openai", {})
            api_key = oc.get("api_key", "")
            if api_key:
                provider = OpenAITTS(api_key=api_key, base_url=oc.get("base_url") or None)
        elif p_type == "edge":
            provider = EdgeTTS(default_voice=entry.get("voice_id", "zh-CN-XiaoxiaoNeural"))
        if provider is not None:
            provider.default_voice = entry.get("voice_id", "")
            provider.timeout_s = timeout_s
            providers.append(provider)
    if not providers:
        return MockTTS()
    return FallbackTTS(providers=providers)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("CozyVoice starting, v%s", __version__)
    cfg = load_config(CFG_PATH).get("voice", {})

    app.state.stt = _build_stt(cfg)
    app.state.tts = _build_tts(cfg)
    app.state.tts_config = cfg.get("tts", {})
    app.state.rate_limiter = RateLimiter(requests_per_minute=30)

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

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
