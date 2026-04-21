"""腾讯云 TTS（短音频合成 API）。"""

from __future__ import annotations

import asyncio
import base64
from typing import Literal

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tts.v20190823 import models, tts_client

from cozyvoice.providers.base import TTSProvider, Voice


_VOICES: list[Voice] = [
    Voice(voice_id="101001", name="智瑜-温暖女声", language="zh", gender="female"),
    Voice(voice_id="101002", name="智聆-通用女声", language="zh", gender="female"),
    Voice(voice_id="101003", name="智美-标准女声", language="zh", gender="female"),
    Voice(voice_id="101004", name="智云-温和男声", language="zh", gender="male"),
    Voice(voice_id="101010", name="智莎-情感女声", language="zh", gender="female"),
]


class TencentTTS(TTSProvider):
    name = "tencent"

    def __init__(self, secret_id: str, secret_key: str, region: str = "ap-guangzhou") -> None:
        cred = credential.Credential(secret_id, secret_key)
        http_profile = HttpProfile(endpoint="tts.tencentcloudapi.com")
        client_profile = ClientProfile(httpProfile=http_profile)
        self._client = tts_client.TtsClient(cred, region, client_profile)

    async def synthesize(
        self,
        text: str,
        voice_id: str = "101001",
        format: Literal["wav", "mp3", "pcm"] = "wav",
    ) -> bytes:
        codec = {"wav": "wav", "mp3": "mp3", "pcm": "pcm"}[format]
        req = models.TextToVoiceRequest()
        req.Text = text
        req.SessionId = f"cozy-{hash(text) & 0xffffffff:x}"
        req.VoiceType = int(voice_id)
        req.Codec = codec
        req.SampleRate = 16000
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, self._client.TextToVoice, req)
        return base64.b64decode(resp.Audio)

    async def list_voices(self) -> list[Voice]:
        return list(_VOICES)
