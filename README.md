# CozyVoice

**CozyEngineV2 语音通道**：非实时 STT + TTS；Provider 抽象；Whisper + 腾讯 TTS；FastRTC 式 WS。

## 快速开始
```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env    # 填 OPENAI_API_KEY / TENCENT_* / BRAIN_URL
.venv/bin/uvicorn cozyvoice.main:app --host 0.0.0.0 --port 8002
```

## 端点
- `POST /v1/voice/chat`  multipart 上传音频 → 返回音频
- `WS /v1/voice/stream?token=<JWT>`  半双工流
- `GET /health`

## 架构
```
Client ─[JWT + audio]→ CozyVoice ─→ STT ─→ Brain /v1/chat/completions (SSE)
                                                    ↓
                                                TTS ─→ audio ─→ Client
```

CozyVoice 透传 client 的 JWT 给 Brain，并加 `X-Source-Channel: voice`。

## 关联
- [CozyEngineV2](https://github.com/zinohome/CozyEngineV2) Brain
- [CozyNanoBot](https://github.com/zinohome/CozyNanoBot) Cerebellum
