# DashScope 1.x → 2.0 迁移指南

> 2.0 是一次有意的重设计，不是 1:1 移植：接口形状向 OpenAI/Anthropic 习惯收敛，
> 管理面与批量任务对齐并增强，推理面差异集中在错误处理与等待语义（§3）。
> 本文汇总迁移时必须知道的**接口对照、行为差异与不在 2.0 范围的能力**。
> 逐接口完整核对表见 [`dashscope-2.0-interfaces-mapping.md`](dashscope-2.0-interfaces-mapping.md)；
> 2.0 各域用法见 [`dashscope-2.0-interfaces-examples.md`](dashscope-2.0-interfaces-examples.md)。

## 1. 共存与分层

- `dashscope>=2.0` 同一个发布包内含 1.x 与 2.0 两套接口，不做双包分发；1.x 进入
  冻结维护（只修缺陷、不增功能），存量代码可以先不动，按模块逐步迁移。
- 2.0 实现直接位于 `dashscope` 根包，`Client` / `AsyncClient` / `chat` /
  `settings` / `WsConnection` / `errors` 等均从根包导入；原 `dashscope.v2`
  命名空间已移除。
- 包根的 `Generation` / `ImageSynthesis` 等 1.x 导出保持不变，两套接口可在
  同一进程混用（共用 `DASHSCOPE_API_KEY` 等环境变量）。旧
  `dashscope.aigc`、`dashscope.client.base_api`、`dashscope.files` 等 deep import
  通过懒加载别名解析到 `dashscope.legacy_1x`，存量代码无需修改，也不新增警告。
- AgentStudio 与 FineTune/Agentic RL 的物理实现归入 `dashscope.applications`；
  原 `dashscope.agentstudio`、`dashscope.finetune` 公开路径通过别名保持不变。

## 2. 三分钟对照

### 2.1 鉴权与全局配置

| 1.x | 2.0 |
| --- | --- |
| `dashscope.api_key = "sk-..."` / `save_api_key()` | `Client(api_key=...)`；缺省读 `DASHSCOPE_API_KEY`；支持 `~/.dashscope/config.toml` |
| 运行时改 `dashscope.api_key` 等模块变量 | `dashscope.settings` 全局门面，赋值即时生效 |
| `base_http_api_url` / `base_compatible_api_url` | `Client(base_url=..., protocol=...)` |
| `base_websocket_api_url` | `DASHSCOPE_WEBSOCKET_BASE_URL` 环境变量或资源级 `ws_url=` 参数 |
| `DASHSCOPE_LOGGING_LEVEL` / `LOG_LEVEL` 两套级别 | `DASHSCOPE_LOG` 统一 + `settings.log_level` |
| `close_shared_aio_session()` | `client.close()` / `await aclient.aclose()`（或上下文管理器） |

### 2.2 各域速查

| 域 | 1.x | 2.0 |
| --- | --- | --- |
| 文本生成 | `Generation.call` / `AioGeneration.call` | `client.chat.completions.create`（AsyncClient 同名） |
| 多模态对话 | `MultiModalConversation.call` | 同上（含媒体 part 自动路由）或显式 `client.multimodal.chat.create` |
| 图像 | `ImageSynthesis.call/async_call/fetch/wait` | `client.images.generate` + `client.images.jobs.create/fetch/wait` |
| 视频 | `VideoSynthesis.call/async_call/fetch/wait` | `client.videos.generate` + `client.videos.jobs.*` |
| 向量 | `TextEmbedding.call` / `BatchTextEmbedding` | `client.embeddings.create` / `client.embeddings.jobs.*` |
| 多模态向量 | `MultiModalEmbedding.call` | `client.multimodal.embeddings.create` |
| 排序 | `TextReRank.call` | `client.rerank.create` |
| 分词 | `Tokenization.call` | `client.tokenization.count`（本地分词 `dashscope.tokenizers.get_tokenizer`） |
| 百炼应用 | `Application.call` | `client.apps.completions.create(app_id=...)` |
| 实时识别 | `Recognition.start/send_audio_frame/stop` | `client.audio.asr.recognize(...)` 上下文管理器 |
| 批量转写 | `Transcription.async_call/fetch/wait` | `client.audio.asr.transcribe(file_urls)` + `client.audio.asr.jobs.*` |
| 语音合成 | `SpeechSynthesizer.call`（sambert/cosyvoice/qwen） | `client.audio.tts.synthesize/stream`（ws）；`client.audio.tts.http.*`（HTTP） |
| 实时翻译 | `TranslationRecognizerRealtime` | `client.audio.translation.recognize(...)` |
| Omni 实时 | `OmniRealtimeConversation` | `client.audio.realtime.connect(...)`（omni 类型化助手） |
| 词表/热词/声纹 | `VocabularyService` / `AsrPhraseManager` / `VoiceEnrollmentService` | `client.audio.vocabularies` / `phrases` / `voices`（+ `voice_design`） |
| 微调 | `FineTunes.*` | `client.fine_tunes.*` |
| 部署 | `Deployments.call/list/get/delete/scale` | `client.deployments.create/list/get/delete/scale` |
| 模型/文件 | `Models.list/get`、`Files.*` | `client.models.*`、`client.files.*` |
| Agentic RL | CLI `dashscope rl ...` | `client.rl.*` + `client.rl.jobs.*` |

```python
# 1.x
from dashscope import Generation
resp = Generation.call(model="qwen3-max", prompt="你好")
if resp.status_code == 200:
    print(resp.output.text)

# 2.0
import dashscope
client = dashscope.Client()          # 缺省读 DASHSCOPE_API_KEY
resp = client.chat.completions.create(model="qwen3-max", messages="你好")
print(resp.choices[0].message.content)   # 失败直接抛异常，见 §3.1
```

## 3. 行为差异（迁移前必读）

### 3.1 错误处理：返回信封 → 抛异常

1.x 所有调用返回响应对象，由调用方自查 `status_code` / `code`；2.0 默认
`raise_on_error=True`，失败直接抛异常族（`DashScopeError` 及其子类，按错误目录
从线缆码映射）：

```python
# 1.x
resp = Generation.call(...)
if resp.status_code != 200:
    print(resp.code, resp.message)

# 2.0
from dashscope import DashScopeError
try:
    resp = client.chat.completions.create(...)
except DashScopeError as e:
    print(e.code, e.message, e.status_code, e.request_id)

# 兼容模式：恢复信封风格，自查 is_error / code / message
client = dashscope.Client(raise_on_error=False)
```

### 3.2 异步任务等待语义

| 维度 | 1.x（`wait_timeout=-1`） | 2.0（`Jobs.wait`） |
| --- | --- | --- |
| 等待超时 | 默认无限等待；超时**返回**错误响应 | 默认 `timeout=86400`（24h）；超时**抛** `DashScopeError` |
| 任务 FAILED | 返回响应，自查 `output.task_status` | `raise_on_error=True` 时按错误目录抛映射异常；`=False` 时返回终态 `Job` |
| 任务 CANCELED | 返回响应 | `raise_on_error=True` 时抛 `DashScopeError`；`=False` 时返回 `Job` |
| 轮询节奏 | 固定间隔 | 指数退避（`poll_interval` → `max_interval`） |

### 3.3 视频 `prompt_extend` 默认值

1.x `VideoSynthesis.call` 的 deprecated 别名参数 `extend_prompt` **默认 `True`**
（`dashscope/legacy_2x/aigc/video_synthesis.py:63`）：不显式传参时默认开启提示词扩展。
2.0 只保留 `prompt_extend=None`（缺省由服务端决定），不恢复 `extend_prompt`
别名。**依赖 1.x 默认开启的调用，迁移后需显式传 `prompt_extend=True`。**

### 3.4 流式输出形态

1.x `Generation.call(stream=True)` 默认每帧**重放全量**文本（需
`incremental_output=True` 才给增量）；2.0 流式一律**增量**（OpenAI 形态），
自行拼接：

```python
stream = client.chat.completions.create(model="qwen3-max", messages="你好", stream=True)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
    # 思考模型另有 delta.reasoning_content
```

百炼应用（`client.apps.completions`）保留 `incremental_output` 开关，帧形态由它决定。

### 3.5 根 `save_api_key` 的实现来源

包根 `save_api_key` 现由 `dashscope._config` 提供，签名仍为
`save_api_key(api_key, api_key_file_path=None)`，默认写入
`~/.dashscope/api_key`。与旧 1.x writer 相比，新实现显式使用 UTF-8、展开
`~`，并为自定义路径创建缺失的父目录；`DASHSCOPE_API_KEY_FILE_PATH` 仍只
影响读取，不改变默认写入位置。需要旧 writer 时可从
`dashscope.legacy_1x.common.api_key` 显式导入。

### 3.6 日志名称与隔离

`DASHSCOPE_LOG` 和 `settings.log_level` 的环境变量、级别及脱敏语义不变。
当前客户端使用独立的 `dashscope._logging` logger 层级，不再以 `v2` 命名，
也不会通过配置共享的 `dashscope` 根 logger 改变 1.x 的 level、handlers 或
filters。按名称获取当前客户端 logger 时应使用 `dashscope._logging`。

## 4. 2.0 新能力（迁移后可选用）

- 全资源 `AsyncClient` 异步孪生（1.x 仅零散 aio 入口）。
- pydantic 强类型响应（`extra="allow"` 兜底，字段可点出）。
- 连接复用：`client.ws_connection()` 一条 ws 承载多任务/顺序复用，全模型 HTTP
  keep-alive 连接池——见 [`dashscope-2.0-connection-reuse-sdk.md`](dashscope-2.0-connection-reuse-sdk.md)。
- `client.with_options(...)` 派生不同配置的客户端，共享连接池。
- 音频专项：HTTP 合成 `client.audio.tts.http`、实时翻译 `client.audio.translation`、
  词表/热词/声纹管理面、omni 类型化助手（1.x 均无对应或形态不同）。

## 5. 不在 2.0 范围的能力

以下能力 **v2 不提供**；均已评审拍板，不再视为缺口。

| 能力 | 1.x 位置 | 处置 | 迁移建议 |
| --- | --- | --- | --- |
| Assistants / Threads / Runs | `dashscope/legacy_2x/assistants/`、`dashscope/legacy_2x/threads/` | **随 1.x 下线，2.0 有意不包含** | 勿用于新代码；托管 Agent 场景评估 `client.agentstudio` 或自建编排 |
| CodeGeneration | `legacy_2x/aigc/code_generation.py` | hold，2.0 先不支持 | 继续用 1.x；代码生成场景可直接走 `chat.completions` 同一端点 |
| nlp Understanding | `legacy_2x/nlp/understanding.py` | hold，2.0 先不支持 | 继续用 1.x |
| MultiModalDialog | `legacy_2x/aigc/multimodal_dialog.py` | hold，2.0 先不支持 | 继续用 1.x；或改 `client.multimodal.chat.create` |
| Conversation / History | `dashscope/legacy_2x/aigc/conversation/` | hold，2.0 先不支持 | 继续用 1.x；多轮对话以 messages 数组自行维护（`chat.completions`） |
| `dashscope` CLI（v2 版） | acli（自包含，走 1.x） | v2 无 CLI；计划项，后续再加 | 命令行场景继续用 1.x CLI |

## 6. 相关文档

- [DashScope 2.0 设计规范](dashscope-2.0-design.md)
- [2.0 接口示例总览](dashscope-2.0-interfaces-examples.md)
- [2.0 接口映射核对表](dashscope-2.0-interfaces-mapping.md)
- [2.0 vs 1.x 对齐评审](dashscope-2.0-vs-1.x-alignment-review.md)
- 语音分册：[ASR](dashscope-2.0-asr-sdk.md) / [TTS](dashscope-2.0-tts-sdk.md) /
  [实时翻译](dashscope-2.0-translation-sdk.md) / [Realtime](dashscope-2.0-realtime-sdk.md) /
  [连接复用](dashscope-2.0-connection-reuse-sdk.md) / [音频管理面](dashscope-2.0-audio-management-sdk.md)
