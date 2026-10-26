# dashscope 2.0 设计方案

> 基于 V2 分支现状（见 `md/dashscope-interfaces.md` 全量接口清单），规划 dashscope 2.0。

## 0. 目标与原则

1. **用户接口友好**：一个现代化的 SDK + CLI（`dashscope`），以及一个 Agentic CLI（`dashscope expert`）。
2. **协议兼容**：通过配置或传入 config，兼容 Anthropic / OpenAI 协议（既可作为客户端消费这两种协议，也保证 dashscope 自身能力可按这两种协议表达）。
3. **接口自动化生成**（后置）：SDK / CLI / 文档 / 测试从单一 API 元数据源自动生成，人工只维护元数据与少量手写覆盖。

现状问题（为什么要 2.0）：

- SDK 接口形态不统一：`Generation.call`、`AioGeneration.call`、`ImageSynthesis.async_call/fetch/wait` 三套风格并存，参数散落在 `**kw`，无统一 Client 对象。
- CLI 与 SDK 一一映射、命令冗长（`dashscope multimodal_conversation create`），缺少面向场景的友好命令。
- 协议层只有自有 `protocol/websocket.py`，OpenAI/Anthropic 兼容散落在 acli 的 providers 中，未沉淀为 SDK 能力。
- 接口靠手写，SDK/CLI/文档三处重复劳动，易漂移。

---

## 1. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│  dashscope expert（Agentic CLI，演进自 dashscope.acli）       │
│  TUI/REPL、session、memory、mcp、tools、skills               │
├─────────────────────────────────────────────────────────────┤
│  dashscope CLI（dashscope.cli 重写）                         │
│  命令树 = SDK 资源树的自动映射 + 场景化快捷命令                  │
├─────────────────────────────────────────────────────────────┤
│  dashscope SDK 2.0（dashscope 包根）                         │
│  Client 资源式 API（chat/images/audio/files/finetunes/...）  │
├──────────────────────┬──────────────────────────────────────┤
│  Protocol 适配层      │  Transport 层                        │
│  openai / anthropic  │  HTTP(SSE) / WebSocket / 任务轮询      │
│  / dashscope-native  │                                      │
├──────────────────────┴──────────────────────────────────────┤
│  API 元数据（specs/*.yaml）→ codegen → SDK/CLI/文档/测试      │
└─────────────────────────────────────────────────────────────┘
```

分层原则：expert 只用 SDK 公开 API；CLI 是 SDK 的薄壳；协议与传输在 SDK 内部隔离，上层无感。

---

## 2. SDK 2.0 设计

### 2.1 统一入口：资源式 Client

对齐 OpenAI/Anthropic 的用户心智，一个 `Client` 对象挂所有资源；同步 + 异步双版本：

```python
import dashscope

client = dashscope.Client(api_key=..., protocol="openai" | "anthropic" | "native", base_url=...)

resp = client.chat.completions.create(model="qwen-plus", messages=[...], stream=True, tools=[...])
img  = client.images.generate(model="wanx2.1-t2i", prompt="...", n=1, size="1024*1024")
aclient = dashscope.AsyncClient(...)
```

### 2.2 资源树与现有接口映射

| 新资源 | 覆盖现有接口（见 interfaces 清单） | 关键方法 |
| --- | --- | --- |
| `client.chat` | `Generation` / `AioGeneration` / `Conversation` / `CodeGeneration` / `Understanding` | `completions.create`（messages 统一格式，tools/stream 一等参数） |
| `client.apps` | `Application` | `completions.create(app_id, prompt)` |
| `client.multimodal` | `MultiModalConversation` / `AioMultiModalConversation` | `chat.create`（content parts：text/image_url/audio/video） |
| `client.images` | `ImageSynthesis` | `generate`（同步封装）/ `jobs.create/fetch/wait`（异步任务） |
| `client.videos` | `VideoSynthesis` | 同上模式 |
| `client.embeddings` | `TextEmbedding` / `MultiModalEmbedding` / `BatchTextEmbedding` | `create` / `jobs.*`（批量任务） |
| `client.rerank` | `TextReRank` / `AioTextReRank` | `create(query, documents)` |
| `client.tokenization` | `Tokenization` | `count(prompt)` |
| `client.audio.asr` | `Recognition` / `Transcription` / `QwenTranscription` | `stream`（实时 ws）/ `transcribe(file_urls)` |
| `client.audio.tts` | `SpeechSynthesizer` / `tts_v2` / `HttpSpeechSynthesizer` / `qwen_tts` | `synthesize`（一次性）/ `stream`（流式） |
| `client.audio.realtime` | `OmniRealtimeConversation` | `session()`（ws 会话上下文管理器） |
| `client.fine_tunes` | `FineTunes` | `create/get/list/cancel/delete/stream_events/wait` |
| `client.deployments` | `Deployments` | `create/get/list/scale/delete` |
| `client.files` | `Files` | `upload/list/get/delete` |
| `client.models` | `Models` | `list/get` |
| `client.rl` | （CLI 的 `dashscope rl` 能力 SDK 化） | `register_functions/test_functions/upload_data/run/get/cancel/logs/list` |
| `client.beta.assistants` | `Assistants/Threads/Messages/Runs/Steps` | OpenAI Assistants 风格 CRUD（保持 `beta` 命名空间） |

约定：

- **长任务统一模式**：`jobs.create → fetch/wait → result`，`wait` 内置退避轮询；所有任务资源返回统一的 `Job` 对象（`id/status/result/error`）。
- **流式统一模式**：`stream=True` 一律返回迭代器/异步迭代器，事件对象统一 `event_type + data`；ws 实时接口用上下文管理器。
- **类型化**：入参出参全部 dataclass/pydantic，`**kw` 仅保留 `extra_headers/timeout` 等逃逸舱；提供 `.pyi`。
- **错误模型**：统一 `DashScopeError` 层级（AuthError/RateLimitError/TaskFailedError...），带 `request_id`。

### 2.3 配置

```python
dashscope.Client(api_key=..., base_url=..., protocol=..., timeout=..., max_retries=...)
```

优先级：显式参数 > 环境变量（`DASHSCOPE_API_KEY` 等） > `~/.dashscope/config.toml`。保留 `dashscope.api_key` 等模块级全局量作为 1.x 兼容层（deprecate 标注）。

### 2.4 兼容性

- 包根继续导出旧接口（`Generation.call` 等），内部转调新 Client，打 DeprecationWarning；两个 minor 版本后移除。
- `dashscope 2.0` 以新 Client 为唯一推荐入口。

---

## 3. 协议兼容层（Anthropic / OpenAI）

`dashscope/protocols/` 下三个 adapter，实现同一内部接口 `ChatProtocol`：

| protocol | 请求/响应形态 | 说明 |
| --- | --- | --- |
| `native` | dashscope 自有 HTTP/ws | 全能力，默认值 |
| `openai` | `/v1/chat/completions`（含 SSE） | 对齐 OpenAI SDK 心智；tool_calls、多模态 content parts 直接可映射 |
| `anthropic` | `/v1/messages`（含 SSE） | system 独立字段、`tool_use/tool_result` block 映射 |

### 3.1 两个方向的兼容

1. **出站（客户端）**：`Client(protocol="openai", base_url=...)` 可直接对接任意 OpenAI/Anthropic 兼容端点（含百炼的 compatible-mode 与第三方网关）。参数翻译规则集中在 adapter：
   - messages：`role/content` 归一化 → 目标协议格式（system 提取、多模态 parts 转换）。
   - tools：OpenAI `functions` ↔ Anthropic `input_schema` ↔ native `tools`。
   - 采样参数：`temperature/top_p/max_tokens/stop` 直接映射；不支持的参数按 `strict=False`（默认）静默丢弃、`strict=True` 抛错。
   - 响应：统一翻译回 SDK 的 `ChatCompletion` 对象（含 usage、finish_reason）。
2. **入站（服务端，可选后置）**：`dashscope serve` 起一个本地 OpenAI/Anthropic 兼容端点，把请求翻译为 native 调用——让存量 OpenAI SDK 用户零改造用 dashscope。此阶段先留接口不实现。

### 3.2 与 acli providers 的关系

acli 现有 `providers/` 已有多协议接入逻辑：抽取其中的协议翻译代码下沉到 SDK `protocols/`，acli 改为复用 SDK，避免两处实现漂移。

---

## 4. dashscope CLI 2.0

### 4.1 命令树（自动映射 SDK 资源树）

```
dashscope chat          create [--model -m] [--prompt | --messages @file.jsonl] [--stream] [--tools @tools.json]
dashscope multimodal    create ...
dashscope images        generate | jobs list/get/wait
dashscope videos        generate | jobs ...
dashscope embeddings    create [--input @texts.txt]
dashscope rerank        create --query ... --documents @docs.json
dashscope audio         asr stream|transcribe / tts synthesize|stream
dashscope files         upload|list|get|delete
dashscope models        list|get
dashscope ft            create|get|list|stream|cancel|delete   （保留现习惯名）
dashscope deployments   create|get|list|scale|delete
dashscope rl            register-functions|test-functions|upload-data|run|get|list|cancel|logs
dashscope apps          call --app-id ...
dashscope assist        ...（beta）
dashscope config        set|get|list          （api_key/protocol/base_url 持久化）
dashscope expert        ...                    （转发到 Agentic CLI，见 §5）
```

### 4.2 体验改进点（相对现状）

- **全局参数**：`--api-key -k`、`--protocol`、`--base-url`、`--output json|table|yaml`、`--json`（机器可读输出，供 jq/脚本）。
- **输入友好**：`--prompt` 直接传文本；`--messages @file` 从文件读；支持管道 `cat msg.json | dashscope chat create -`；图片等本地文件自动经 `oss upload` 换 URL（复用现有 `cli/oss.py`）。
- **流式输出**：TTY 下默认渲染流式文本，`--no-stream` 关闭。
- **任务轮询内置**：`images generate --wait` 一条命令等结果。
- **隐藏别名保留**：`fine-tunes`→`ft`、`agentic-rl`→`rl`，不破坏存量脚本。

实现：Typer 保留；命令层由 codegen（§6）从元数据生成骨架，每个命令只写参数渲染与输出格式化。

---

## 5. dashscope expert（Agentic CLI）

基于现有 `dashscope.acli` 演进，不改交互面，只做三层整合：

1. **底座切换**：模型调用从 acli 自有 providers 切到 SDK 2.0 Client（§3 协议兼容自动生效，`/provider`、`/model` 及 `load_config(provider, base_url)` 语义不变）。
2. **入口与命名**：新增 `dashscope expert` 顶层命令（等价 `dashscope --tui` 等现有启动参数）；`run_interactive` SDK 入口保留。
3. **dashscope 能力即工具**：把 SDK 资源（文生图、ASR/TTS、embedding、文件上传等）注册为 expert 内置 tools/skills，让 agent 一句话调全量 dashscope 能力——这是 expert 相对通用 agentic CLI 的差异点。

现有命令面（`/session`、`/memory`、`/mcp`、`/tts`、`/cron` 等，见 interfaces 清单第三部分）全部保留，不在本期重构范围。

---

## 6. 接口自动化生成（阶段三）

### 6.1 元数据源

`specs/*.yaml`（OpenAPI 3 扩展）描述每个 API：路径、方法、参数 schema、任务型/流式/普通三种形态标记、协议映射注解（openai/anthropic 对应字段）、CLI 命令名与示例。

### 6.2 生成物

| 产物 | 说明 |
| --- | --- |
| SDK 资源方法 | 参数 dataclass + 请求/响应模型 + 同步/异步实现 |
| CLI 命令 | 参数定义 + help 文本 |
| 类型存根 / 文档 | `.pyi` 与 markdown 接口表（替代手写清单） |
| 冒烟测试 | 按元数据中的示例生成参数化用例 |

`scripts/codegen.py` 一键生成，CI 校验"生成物与元数据一致"。手写覆盖点：复杂流式/实时接口（ws）、需要特殊 UX 的命令，用 `# manual` 标记豁免生成。

---

## 7. 目录与里程碑

### 目标目录布局

```
dashscope/
  client.py            # Client / AsyncClient
  resources/           # chat, images, audio, files, finetunes, rl, assist...
  protocols/           # native / openai / anthropic adapter
  transport/           # http(sse) / websocket / task polling
  cli/                 # CLI 2.0（生成 + 手写渲染）
  acli/                # expert（改底座，不改交互面）
  compat/              # 1.x 旧接口兼容层
specs/                 # API 元数据（阶段三）
scripts/codegen.py
```

### 里程碑

| 阶段 | 内容 | 交付 |
| --- | --- | --- |
| M1 | SDK 2.0 骨架：Client、transport、chat/embeddings/rerank/files/models 先行，native 协议 | 新 Client 可用，旧接口并存 |
| M2 | 协议层：openai / anthropic adapter + 参数翻译 + 配置 | `protocol=` 生效，兼容端点可直连 |
| M3 | SDK 全量覆盖：多模态/任务型（images/videos）、audio（含 ws）、ft/deployments/rl/assistants | interfaces 清单 100% 映射 |
| M4 | CLI 2.0 + `dashscope config` | 新命令树，旧命令保留别名 |
| M5 | expert 整合：acli 底座切 SDK、dashscope 能力注册为 tools | `dashscope expert` 发布 |
| M6 | codegen：specs 元数据 + 生成 SDK/CLI/文档/测试，CI 校验 | 新增接口只改 yaml |
| M7 | 1.x 兼容层下线、文档迁移 | 2.0 正式 |

### 风险与决策点

- **多模态消息格式**：native 与 OpenAI content parts 差异最大的地方，M1 前需先定内部规范消息模型。
- **ws 实时接口与协议层的关系**：Recognition/OmniRealtime 无 OpenAI/Anthropic 对应物，协议层仅覆盖 HTTP 文本类接口，需在文档明示。
- **`dashscope serve`（入站兼容）**是否做、何时做，待 M2 后评估。
- **expert 的包归属**：保持 `dashscope.acli` 模块名不变（存量用户脚本兼容），仅在命令层加 `expert` 入口。
