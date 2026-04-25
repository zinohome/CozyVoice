# CozyVoice Production Readiness Design

> CozyVoice 从"能跑"到"生产就绪"：模态架构形式化 + TTS 三级降级 + Pipeline 补全 + 部署配置 + 测试覆盖。

**日期**: 2026-04-25
**状态**: Draft
**仓库**: ~/CozyProjects/CozyVoice/

---

## 1. 模态架构

CozyVoice 与 Brain (CozyEngineV2) 的关系是**模态的**——谁是主体取决于场景。

### 1.1 非实时模态（Brain 主体）

CozyVoice 是 Brain 的"语音外挂"，只做 audio↔text 转换，不做业务决策。

```
Client ── audio ──→ CozyVoice
                      ├─ STT(audio→text)
                      ├─ ← 立即返回 stt_text
                      ├─ brain_client.chat_stream(stt_text) ──→ Brain(主体)
                      │                                         ├→ Memory
                      │                                         ├→ NanoBot
                      │                                         └→ LLM
                      ├─ ← 流式返回 reply_text
                      ├─ TTS(reply_text→audio, fallback chain)  [可选]
                      └─ ← 返回 reply_audio                    [可选]
```

### 1.2 实时模态（CozyVoice 主体）

CozyVoice Agent 掌控实时音频循环，Brain + Memory + NanoBot 退为"顾问团"。

```
Client ── WebRTC ──→ LiveKit Server ←──→ CozyVoice Agent(主体)
                                          ├─ OpenAI Realtime 模式
                                          └─ Cozy Pipeline 模式
                                               ↑
                              Brain 推送人格/画像（顾问角色）
                              Brain 异步接收 transcript（记忆存储）
```

### 1.3 三层能力模型

非实时模态下，三层能力逐层叠加，客户端按需选用：

| 层级 | 端点 | 输入 | 输出 | Brain 参与 |
|:---|:---|:---|:---|:---|
| Layer 1 纯转写 | `POST /v1/voice/transcribe` | audio | stt_text | 否 |
| Layer 2 语音对话(文本) | `POST /v1/voice/chat?tts=false` | audio + session/personality | stt_text + reply_text (SSE) | 是 |
| Layer 3 语音对话(语音) | `POST /v1/voice/chat?tts=true` | audio + session/personality | stt_text + reply_text + reply_audio (SSE) | 是 |
| Layer 4 实时语音 | LiveKit Room (WebRTC) | 实时音频流 | 实时音频流 | 顾问角色 |

---

## 2. API 契约

### 2.1 Layer 1: 纯转写

```
POST /v1/voice/transcribe
Content-Type: multipart/form-data
Body: audio (file)

Response 200 (JSON):
{
  "text": "北京天气怎么样",
  "language": "zh",
  "duration_ms": 2300
}
```

### 2.2 Layer 2 & 3: 语音对话 (SSE)

```
POST /v1/voice/chat?tts=false
Content-Type: multipart/form-data
Body: audio (file) + session_id + personality_id

Response 200 (SSE stream):

event: stt
data: {"text": "北京天气怎么样"}

event: reply_chunk
data: {"delta": "北京明天"}

event: reply_chunk
data: {"delta": "晴，气温20-28度"}

event: reply_done
data: {"text": "北京明天晴，气温20-28度"}
```

当 `tts=true` 时，额外追加：

```
event: tts_audio
data: {"format": "mp3", "base64": "<base64-encoded-audio>"}
```

### 2.3 Layer 4: 实时语音

通过 LiveKit Room 协议，不走 HTTP API：

1. Client 请求 Brain: `POST /v1/voice/token` → `{token, livekit_url, room_name}`
2. Client 用 token 加入 LiveKit Room (WebRTC)
3. CozyVoice Agent 自动加入同一 Room，实时音频双向流
4. 会话结束后 Agent 调 Brain: `POST /v1/chat/voice_summary` 异步存记忆

### 2.4 WebSocket 变体 (Layer 2/3)

```
WS /v1/voice/stream
Client → binary audio frames (持续发送)
Server → JSON messages:
  {"event": "stt", "text": "..."}
  {"event": "reply_chunk", "delta": "..."}
  {"event": "reply_done", "text": "..."}
  {"event": "tts_audio", "format": "mp3", "base64": "..."}
```

---

## 3. Track 1: TTS 层 — 三 Provider + Fallback

### 3.1 Provider 架构

```
TTSProvider (ABC)                    ← 已有
  ├─ TencentTTS      (主力，中文)    ← 已有
  ├─ OpenAITTS        (备选，多语言)  ← 新增
  ├─ EdgeTTS          (兜底，免费)    ← 新增
  └─ MockTTS          (测试用)       ← 已有

FallbackTTS (新增)
  ├─ providers: list[TTSProvider]
  ├─ synthesize(): 按序尝试，失败跳下一个
  └─ 每次失败: log.warning + prometheus counter
```

### 3.2 新增文件

| 文件 | 用途 |
|:---|:---|
| `src/cozyvoice/providers/tts/openai_tts.py` | OpenAI Audio API TTS |
| `src/cozyvoice/providers/tts/edge_tts.py` | Microsoft Edge TTS (免费) |
| `src/cozyvoice/providers/tts/fallback.py` | FallbackTTS 包装类 |

### 3.3 配置

```yaml
# config/voice.yaml
tts:
  fallback_chain:
    - provider: tencent
      voice_id: "101001"
      timeout_ms: 3000
    - provider: openai
      voice_id: "shimmer"
      timeout_ms: 5000
    - provider: edge
      voice_id: "zh-CN-XiaoxiaoNeural"
      timeout_ms: 5000
```

### 3.4 FallbackTTS 核心逻辑

```python
class FallbackTTS(TTSProvider):
    def __init__(self, providers: list[TTSProvider]):
        self._providers = providers

    async def synthesize(self, text, voice_id=None, format="mp3") -> bytes:
        for provider in self._providers:
            try:
                return await asyncio.wait_for(
                    provider.synthesize(text, voice_id or provider.default_voice, format),
                    timeout=provider.timeout_s,
                )
            except Exception:
                tts_fallback_total.labels(provider=provider.name).inc()
                logger.warning("TTS provider %s failed, trying next", provider.name)
                continue
        raise TTSAllProvidersFailedError("all TTS providers failed")
```

### 3.5 依赖

```
edge-tts>=6.1       # Edge TTS (异步，免费)
openai>=1.0         # OpenAI TTS (已有依赖)
```

---

## 4. Track 2: STT 层 + API 改造

### 4.1 新增独立转写端点

在 `api/rest.py` 新增 `POST /v1/voice/transcribe`，调用 STT provider 后直接返回 JSON，不经过 Brain。

### 4.2 改造 voice/chat 为 SSE 流式

现有 `/v1/voice/chat` 从一次性 JSON 响应改为 SSE 流式响应：

```python
@router.post("/v1/voice/chat")
async def voice_chat(audio: UploadFile, session_id, personality_id, tts: bool = False):
    async def stream():
        # Step 1: STT — 立即推送
        stt_result = await stt.transcribe(audio)
        yield sse_event("stt", {"text": stt_result.text})

        # Step 2: Brain 流式回复
        full_reply = ""
        async for chunk in brain_client.chat_stream(stt_result.text, session_id, personality_id):
            full_reply += chunk
            yield sse_event("reply_chunk", {"delta": chunk})
        yield sse_event("reply_done", {"text": full_reply})

        # Step 3: TTS（仅 tts=true 时）
        if tts:
            audio_bytes = await fallback_tts.synthesize(full_reply)
            yield sse_event("tts_audio", {"format": "mp3", "base64": b64encode(audio_bytes).decode()})

    return EventSourceResponse(stream())
```

### 4.3 brain_client 流式改造

`brain_client` 新增 `chat_stream()` 方法（保留 `chat_collect()` 供 realtime_agent 使用），返回 `AsyncIterator[str]`：

```python
async def chat_stream(self, text, session_id, personality_id) -> AsyncIterator[str]:
    async with self._client.stream("POST", f"{self._base}/v1/chat/completions", json=body) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                chunk = json.loads(line[6:])
                yield chunk.get("delta", "")
```

---

## 5. Track 3: 生产基础设施

### 5.1 LiveKit Server 生产配置

```yaml
# deploy/livekit/livekit.yaml
port: 7880
rtc:
  port_range_start: 50000
  port_range_end: 50200
  use_external_ip: true
keys:
  cozy_api_key: ${LIVEKIT_API_SECRET}
logging:
  level: info
```

去掉 `--dev`，改为 `--config /etc/livekit.yaml`。RTC 端口显式声明，API key 通过环境变量注入。

### 5.2 1Panel compose 更新

`deploy/base_runtime/docker-compose.1panel.yml` 中 LiveKit 服务改为：

```yaml
cozy_livekit:
  image: livekit/livekit-server:v1.8
  command: --config /etc/livekit.yaml --node-ip ${NODE_IP}
  volumes:
    - ../livekit/livekit.yaml:/etc/livekit.yaml:ro
  ports:
    - "7880:7880"
    - "50000-50200:50000-50200/udp"
```

### 5.3 全栈联调 compose

新增 `docker-compose.full-stack.yml`，包含全部 6 个服务：

| 服务 | 端口 | 依赖 |
|:---|:---|:---|
| postgres | 5432 | - |
| redis | 6379 | - |
| brain | 8000 | postgres, redis |
| memory | 8001 | redis |
| nanobot | 8080 | - |
| livekit | 7880 + UDP 50000-50200 | - |
| cozy-voice | 8002 | brain, livekit |

### 5.4 外部暴露端口

- `brain:8000` — 文本 API + voice token 签发
- `livekit:7880` — WebRTC 信令
- `cozy-voice:8002` — 非实时语音 API

其余服务仅 docker 内部网络通信。

---

## 6. Track 4: 测试补齐

### 6.1 新增测试文件

| 文件 | 覆盖目标 | 数量(估) |
|:---|:---|:---|
| `tests/unit/test_openai_tts.py` | OpenAI TTS provider | ~4 |
| `tests/unit/test_edge_tts.py` | Edge TTS provider | ~4 |
| `tests/unit/test_fallback_tts.py` | FallbackTTS 全部降级路径 | ~6 |
| `tests/unit/test_transcribe_endpoint.py` | Layer 1 纯转写 | ~3 |
| `tests/unit/test_voice_chat_sse.py` | Layer 2/3 SSE 三阶段 | ~6 |
| `tests/unit/test_pipeline_agent.py` | 补全 Pipeline Agent | ~5 |
| `tests/unit/test_livekit_entrypoint.py` | 补全 LiveKit 入口 | ~4 |
| `tests/unit/test_config_loader.py` | 配置解析 + env 插值 | ~4 |
| `tests/integration/test_voice_chat_roundtrip.py` | 非实时全链路 | ~3 |
| `tests/integration/test_fallback_chain.py` | TTS 降级全链路 | ~3 |

### 6.2 测试目标

- 当前: ~48 个测试
- 新增: ~42 个测试
- **目标: ≥ 85 个测试全绿**

### 6.3 测试原则

- 外部 API (OpenAI / Tencent / Edge) 全部 mock，不依赖真实密钥
- Brain 调用 mock httpx 响应
- LiveKit Room/Agent 用 mock 对象
- FallbackTTS 重点测试各种失败组合（主力成功、主力失败切备选、全挂报错）
- SSE 流式响应验证事件顺序和数据格式

---

## 7. 不在本次范围内

- CozyChat 前端集成（不是真实前端）
- 多语言 STT/TTS 切换策略（后续迭代）
- 负载测试 / 压力测试
- LiveKit TURN server 配置（内网部署不需要）
- Prometheus/Grafana 面板搭建（仅埋点，不建面板）

---

## 8. 依赖清单

```toml
# pyproject.toml 新增
edge-tts = ">=6.1"
sse-starlette = ">=1.6"    # SSE 响应支持（如未有）
```

现有依赖已包含：`openai`, `livekit-agents`, `livekit-api`, `httpx`, `prometheus-client`

---

## 9. 交付标准

| 维度 | 标准 |
|:---|:---|
| 功能 | Layer 1-3 端点可用，Layer 4 LiveKit Agent 可连 |
| TTS | 三级 fallback 链工作，任一 provider 故障自动切换 |
| 部署 | `docker-compose.full-stack.yml` 一键起全栈 |
| 测试 | ≥ 85 个测试全绿 |
| 可观测 | TTS fallback / STT 延迟 / Brain 调用延迟有 Prometheus 指标 |
