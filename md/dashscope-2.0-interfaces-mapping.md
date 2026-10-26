# dashscope 2.0 接口映射核对表（M3 3.9 / 3.10）

对照 [`md/dashscope-interfaces.md`](dashscope-interfaces.md) 第一部分（SDK 接口），
核对 `dashscope/v2/` 资源树的覆盖情况。

状态图例：✅ 已迁移 ｜ 🔀 收敛合并 ｜ ❌ 确定下线 ｜ ⚠️ 缺口（未迁移）

## 文本 / 对话

| 1.x 接口 | 2.0 资源 / 方法 | 状态 |
| --- | --- | --- |
| `Generation.call` | `client.chat.completions.create` | ✅ |
| `AioGeneration.call` | `AsyncClient.chat.completions.create` | ✅ |
| `Conversation.call` | `client.chat.completions.create`（history 转 messages） | 🔀 |
| `CodeGeneration.call` | `client.chat.completions.create`（同一端点，model 区分） | 🔀 |
| `Understanding.call` | `client.chat.completions.create`（同一端点，model 区分） | 🔀 |
| `Application.call` | `client.apps.completions.create(app_id=..., prompt=...)` | ✅ |

## 多模态

| 1.x 接口 | 2.0 资源 / 方法 | 状态 |
| --- | --- | --- |
| `MultiModalConversation.call` | `client.multimodal.chat.create`（content parts） | ✅ |
| `AioMultiModalConversation.call` | `AsyncClient.multimodal.chat.create` | ✅ |
| `ImageSynthesis.call/async_call/fetch/wait` | `client.images.generate` + `client.images.jobs.create/fetch/wait` | ✅ |
| `ImageGeneration.call/async_call/fetch/wait`（wan2.6-t2i / wan2.6-image） | `client.images.generate`（同步打 multimodal-generation 端点）+ `client.images.jobs.create/fetch/wait`（异步任务打 image-generation 端点）；`client.images` 按模型名自动路由协议 | ✅ |
| `VideoSynthesis.call/async_call/fetch/wait` | `client.videos.generate` + `client.videos.jobs.create/fetch/wait` | ✅ |
| `MultiModalEmbedding.call` | `client.multimodal.embeddings.create` | ✅ |
| `AioMultiModalEmbedding.call` | `AsyncClient.multimodal.embeddings.create` | ✅ |

> `client.chat.completions.create` 会检查 messages，含 image/audio/video part 时
> 自动改走多模态端点（签名与返回类型不变）；`client.multimodal.chat.create` 保留
> 为显式逃生口——无论有无媒体 part 都打到多模态端点（如给 qwen-vl 发纯文本）。
> 兼容协议（`openai`/`anthropic`）无独立多模态端点，两者是同一次调用。
>
> `client.images` 同样按模型名路由：旧 text2image 端点仅支持 wan2.5 及以下模型；
> wan2.6-t2i / wan2.6-image 只应答 multimodal-generation 新协议（`messages` 入参 /
> `choices` 出参），同步结果与任务结果都归一化为统一 `Job`（`results` / `image_urls`）。

## 向量

| 1.x 接口 | 2.0 资源 / 方法 | 状态 |
| --- | --- | --- |
| `TextEmbedding.call` | `client.embeddings.create` | ✅ |
| `BatchTextEmbedding`（call/fetch/wait） | `client.embeddings.jobs.create/fetch/wait`（复用 Job 基建，M3 3.10） | ✅ |

## 排序

| 1.x 接口 | 2.0 资源 / 方法 | 状态 |
| --- | --- | --- |
| `TextReRank.call` | `client.rerank.create` | ✅ |
| `AioTextReRank.call` | `AsyncClient.rerank.create` | ✅ |

## 分词

| 1.x 接口 | 2.0 资源 / 方法 | 状态 |
| --- | --- | --- |
| `Tokenization.call` | `client.tokenization.count` | ✅ |

## 语音

| 1.x 接口 | 2.0 资源 / 方法 | 状态 |
| --- | --- | --- |
| `Recognition.start/send_audio_frame/stop` | `client.audio.asr.recognize(...)` 上下文管理器（`send_audio_frame/stop/events`） | ✅ |
| `Transcription.async_call/fetch/wait` | `client.audio.asr.transcribe(file_urls)` + `client.audio.asr.jobs.*` | ✅ |
| `audio.qwen_asr.QwenTranscription` | 同上（`transcribe(model=...)` 按模型区分） | 🔀 |
| `SpeechSynthesizer.call`（sambert/cosyvoice） | `client.audio.tts.synthesize/stream`（ws，`model/voice` 区分） | ✅ |
| `audio.tts_v2.SpeechSynthesizer`（CosyVoice2 流式） | `client.audio.tts.stream` | 🔀 |
| `HttpSpeechSynthesizer.call` | `client.audio.tts.http.synthesize/stream`（M4，HTTP 形态；ws 统一入口亦可） | ✅ |
| `audio.qwen_tts.SpeechSynthesizer` | `client.audio.tts.http.*`（M4 按模型名前缀路由）/ `client.audio.tts.synthesize`（ws） | ✅ |
| `TranslationRecognizerRealtime` / `TranslationRecognizerChat` | `client.audio.translation.recognize(...)`（M3） | ✅ |
| `VocabularyService` / `AsrPhraseManager` | `client.audio.vocabularies` / `client.audio.phrases`（M5） | ✅ |
| `VoiceEnrollmentService`（声音复刻） | `client.audio.voices`（M5；qwen 声音设计另建模为 `client.audio.voice_design`，1.x 无对应类） | ✅ |
| `audio.qwen_omni.OmniRealtimeConversation` | `client.audio.realtime.connect(...)`（M2 起含 omni 类型化助手） | ✅ |

> audio 资源树的推理入口为 `asr` / `tts` / `realtime` / `translation`，管理面为
> `vocabularies` / `phrases` / `voices` / `voice_design`；`asr` 下直接是
> `transcribe` / `recognize` / `jobs.*`，不再有 `transcription` 中间层。
> 实时语音连接复用（一条 ws 多任务/顺序复用）见
> [`dashscope-2.0-connection-reuse-sdk.md`](dashscope-2.0-connection-reuse-sdk.md)（M6）。

## 微调

| 1.x 接口 | 2.0 资源 / 方法 | 状态 |
| --- | --- | --- |
| `FineTunes.call` | `client.fine_tunes.create` | ✅ |
| `FineTunes.cancel/list/get/delete` | `client.fine_tunes.cancel/list/get/delete` | ✅ |
| `FineTunes.stream_events` | `client.fine_tunes.stream_events` | ✅ |
| `FineTunes.wait` | `client.fine_tunes.wait`（退避轮询 + 超时） | ✅ |
| `FineTunes.logs` | `client.fine_tunes.logs`（rl 侧同一能力为 `client.rl.jobs.logs`） | ✅ |

## 部署

| 1.x 接口 | 2.0 资源 / 方法 | 状态 |
| --- | --- | --- |
| `Deployments.call/list/get/delete/scale` | `client.deployments.create/list/get/delete/scale` | ✅ |

## 资源

| 1.x 接口 | 2.0 资源 / 方法 | 状态 |
| --- | --- | --- |
| `Models.list/get` | `client.models.list/get` | ✅ |
| `Files.upload/list/get/delete` | `client.files.upload/list/get/delete` | ✅ |

## Agentic RL

| 1.x 接口（CLI `dashscope rl`） | 2.0 资源 / 方法 | 状态 |
| --- | --- | --- |
| `rl register_functions` | `client.rl.register_functions`（复用 1.x 实现） | ✅ |
| `rl test_functions` | `client.rl.test_functions` | ✅ |
| `rl upload_data` | `client.rl.upload_data` | ✅ |
| `rl run` | `client.rl.jobs.create` | ✅ |
| `rl get/cancel/logs/list` | `client.rl.jobs.fetch/cancel/logs/list`（v2 HTTP/错误模型） | ✅ |
| `rl wait`（CLI 轮询） | `client.rl.jobs.wait`（退避轮询 + 超时） | ✅ |

> `client.reinforcement_learning` 是 `client.rl` 的长名别名（同一实例）。
> rl 资源面分两层：`client.rl.*` 是本地准备动作，`client.rl.jobs.*` 是训练任务
> 管理（与 `images.jobs` / `fine_tunes` 同一套 Job 语义）。
> **`jobs.create` / `register_functions` / `test_functions` / `upload_data`
> 这四个 1.x 直通方法不接受 `timeout=`**（本地打包上传与 YAML 解析的超时由 1.x
> 内部控制）；其余 `jobs.*` 走 v2 传输层，`timeout=` 可用。

## Assistants API【确定下线】

| 1.x 接口 | 状态 |
| --- | --- |
| `Assistants.create/retrieve/list/update/delete` | ❌ |
| `Threads.create/retrieve/update/delete/create_and_run` | ❌ |
| `Messages.create/retrieve/list/update/delete` | ❌ |
| `Runs.*` | ❌ |
| `Steps.list/retrieve` | ❌ |

托管 Agent 场景由 `client.agentstudio` 承接（1.x 子 SDK 懒挂载，逻辑不动）。

## AgentStudio

| 接口 | 2.0 挂载 | 状态 |
| --- | --- | --- |
| `agentstudio.Client.agents/sessions/events/files/skills/vaults/environments/deployments` | `client.agentstudio.*`（懒挂载 1.x `Client`/`AsyncClient`） | ✅ |

## 全局配置

| 1.x | 2.0 | 状态 |
| --- | --- | --- |
| `dashscope.api_key` / `save_api_key()` | `Client(api_key=...)` + 配置文件（`resolve_config`） | 🔀 |
| `dashscope.api_key` / `base_*` 模块变量运行时修改 | `dashscope.settings` 全局门面（赋值即时生效，懒读 env，M3 3.11） | ✅ |
| `DASHSCOPE_LOGGING_LEVEL` / `LOG_LEVEL` 两套日志级别 | `DASHSCOPE_LOG` 统一（完整级别）+ `settings.log_level` setter | 🔀 |
| `base_http_api_url` / `base_compatible_api_url` | `Client(base_url=..., protocol=...)` | 🔀 |
| `base_websocket_api_url` | `DASHSCOPE_WEBSOCKET_BASE_URL` 环境变量 + 资源 `ws_url=` 参数 | 🔀 |
| `close_shared_aio_session()` | `client.close()` / `await client.aclose()` | 🔀 |

## 缺口清单

| 缺口 | 1.x 对应 | 说明 |
| --- | --- | --- |
| ~~`client.embeddings` 多模态向量化~~ | `MultiModalEmbedding` / `AioMultiModalEmbedding` | 已补齐（M3 3.10）：`multimodal.embeddings.create`，端点 `/services/embeddings/multimodal-embedding/multimodal-embedding` |
| ~~`client.embeddings` 批量任务~~ | `BatchTextEmbedding` | 已补齐（M3 3.10）：`embeddings.jobs.create/fetch/wait`，复用 3.2 Job 基建 |

> M3 3.10 后核对表无遗留缺口（除确定下线的 Assistants API 族）。
