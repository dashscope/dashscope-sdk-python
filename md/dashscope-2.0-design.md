# dashscope 2.0 设计方案

> 基于 V2 分支现状（见 `md/dashscope-interfaces.md` 全量接口清单），规划 dashscope 2.0。

## 0. 目标与原则

**目标**

1. **用户接口友好**：一个现代化的 SDK + CLI（`dashscope`），以及一个 Agentic CLI（`dashscope expert`）。
2. **协议兼容**：通过配置或传入 config，兼容 Anthropic / OpenAI 协议（既可作为客户端消费这两种协议，也保证 dashscope 自身能力可按这两种协议表达）。
3. **接口自动化生成**：SDK / CLI / 文档 / 测试从单一 API 元数据源自动生成，人工只维护元数据与少量手写覆盖。

**原则**（对比调研后确立/修订）

- **P1 对齐行业基线，不做发明**：Client 实例化、Pydantic 类型化、异常驱动、自动重试、httpx——OpenAI/Anthropic（同为 Stainless 生成）已验证的现代 SDK 基线，直接对齐，不自创风格。
- **P2 契约正确性优先于功能数量**：错误被吞、退出码撒谎、流中出错不报——这类"静默错误"是调研发现的最高危问题，修复优先级高于任何新能力。
- **P3 渐进迁移，不破坏存量**：1.x 代码冻结、2.0 物理隔离新建（§2.4 零接触方案），不建兼容层；DeprecationWarning 引导迁移；1.x 快修（M0）在维护分支单独实施，不等 2.0。
- **P4 保留自有优势**：异步长任务（async_call/fetch/wait）、多模态 content 格式（`{"text":..., "image":...}`）、RL/微调/部署全流程是调研确认的领先项，2.0 统一化但**不削弱、不下线**。
- **P5 差异化定位**：CLI 不跟随 Agent 赛道（expert 负责），主打两家已放弃的"AI 平台长流程管理"品类；SDK 不做厂商专属适配，靠协议层通吃（第三方模型已收敛到 OpenAI/Anthropic 兼容）。
- **P6 codegen 是主线不是后置**：调研证实手写 vs 流水线是代际差距（OpenAI/Anthropic 均为 spec 生成）。specs 元数据从 M1 起同步建设，M6 全面接管，而非等功能写完再补。

现状问题（为什么要 2.0）：

- SDK 接口形态不统一：`Generation.call`、`AioGeneration.call`、`ImageSynthesis.async_call/fetch/wait` 三套风格并存，参数散落在 `**kw`，无统一 Client 对象。
- CLI 与 SDK 一一映射、命令冗长（`dashscope multimodal_conversation create`），缺少面向场景的友好命令。
- 协议层只有自有 `protocol/websocket.py`，OpenAI/Anthropic 兼容散落在 acli 的 providers 中，未沉淀为 SDK 能力。
- 接口靠手写，SDK/CLI/文档三处重复劳动，易漂移。
- （调研新增）错误契约弱：返回值式错误处理易漏检、核心路径无自动重试、流式出错不退出/不可解析、日志缺 request_id 且无敏感信息过滤。
- （调研新增）战略窗口期：OpenAI/Anthropic 已放弃"API 管理 CLI"品类、其 SDK 全面 spec 生成化——DashScope 若不在 2.0 补齐契约与流水线，差距将从"体验落后"固化为"代际落后"。

### 对比调研共识（→ 设计硬需求）

对 OpenAI/Anthropic SDK+CLI 的八份对比调研（见 `md/` 目录）收敛出的共识，全部纳入本方案：

| 调研结论 | 出处 | 2.0 落点 | 预期收益 |
| --- | --- | --- | --- |
| Client 实例化 + 链式资源导航（类方法 + 全局变量是最大槽点） | sdk-* 对比 ×4 | §2.1 Client 资源式 API | 易用性：多租户/多环境安全，IDE 补全全链路可用 |
| 异常驱动错误处理（按 HTTP 状态细分异常类）+ 错误码枚举 | sdk-source-level-comparison-part2 | §2.7 错误模型（§2.2 摘要） | 稳定性：错误不再被漏检（返回值模式漏写 if 即静默失败）；重试/降级逻辑可收敛 |
| 内置自动重试（默认 2 次 + Retry-After + 指数退避），DashScope 核心路径目前无重试 | sdk-source-level-comparison-part2 §2.4 | §2.2 传输层 | 稳定性：限流/抖动场景成功率直接提升，用户最痛点 |
| Pydantic 类型化响应 + `@overload` 精确签名（流式/非流式类型层面可区分） | sdk-source-level-comparison-part2 §1.3 | §2.2 约定 | 易用性：字段拼写错误静态期暴露，杜绝运行时 KeyError |
| 流式人体工学：text_stream 纯文本迭代器、final message 聚合、流内错误抛异常 | sdk-deep-comparison §3.3 | §2.2 流式统一模式 | 易用性 + 稳定性：用户少写累积/检查样板；流中错误不再被吞 |
| 便捷属性（`resp.output_text` 一行取文本）+ Quickstart 极简入口 | sdk-comparison-openai-vs-dashscope | §2.5 | 易用性：上手代码从 6 行降到 2 行，对齐 OpenAI 首屏体验 |
| httpx 统一同步/异步，消除 requests/aiohttp 双栈 | sdk-source-level-comparison-part2 §3.4 | §2.2 传输层 | 性能：连接池复用减少每请求 TCP/TLS 握手；包体积与双栈漂移风险同降 |
| 日志：request_id 入日志、敏感头过滤、重试原因/等待时间 | sdk-deep-comparison §6.7 | §2.2 日志 | 可运维性：排障可定位到工单级；消除 API Key 泄露入日志的安全隐患 |
| OpenAI/Anthropic SDK 均为 Stainless 从 OpenAPI spec 生成，手写 vs 流水线是代际差距 | conclusion-sdk-architecture-stainless | §6 codegen 升为战略项 | 研发效率：新参数上线从"人肉改签名+文档+测试"（周级）降到改 spec 自动生成（天级），消除漂移 |
| OpenAI 已删除 API CLI、Anthropic 从未有过；"API 管理 CLI"品类只剩 DashScope，独特价值在长流程管理（ft/部署/RL） | cli-source-level-comparison-2026-08 §0 | §4 定位 | 战略：占据两家放弃的品类空位，差异化而非跟随 |
| CLI 立即修：流式出错 exit 0 bug、shell 补全、auth whoami、`--output-format json` | cli-source-level-comparison-2026-08 §5 | §4.2 | 稳定性（CI 误判成功）+ 易用性（补全/认证探测）+ 可集成性（jq 可解析） |
| MiniMax/DeepSeek/Baseten/Kimi 协议已收敛到 OpenAI/Anthropic 兼容，无需厂商专属 adapter | vendor-sdk-research-compat | §3.3 | 研发效率：避免 N 个厂商 adapter 的维护负担；第三方模型开箱即用 |

**收益汇总**（按维度）：

- **易用性**：上手 2 行代码、IDE 全补全、错误不用逐个 if 检查、流式无需手工累积——对齐 OpenAI/Anthropic 的现代基线（调研评分中 DashScope 差距最大的三项：类型提示、错误契约、流式人体工学）。
- **稳定性**：自动重试 + 异常驱动 + 流内错误抛出 + CLI 退出码修复，覆盖"请求失败被静默吞掉"的全部已知路径；这是当前 SDK/CLI 唯一会导致用户数据/CI 结果错误的类别。
- **性能**：httpx 连接池复用（现状每次请求新建 requests.Session）；高并发场景减少握手开销，异步路径与同步共用一套实现、无行为漂移。
- **研发效率**：codegen 使"新增一个 API 参数"的成本从周级手写降到天级自动生成，SDK/CLI/文档/测试四处不再漂移。
- **战略**：独占"AI 平台长流程管理 CLI"品类 + 第三方模型零成本接入，构成 2.0 的差异化护城河。

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
| `client.images` | `ImageSynthesis` / `ImageGeneration` | `generate`（同步封装）/ `jobs.create/fetch/wait`（异步任务）；按模型名路由协议——wan2.5 及以下走旧 text2image 端点（`/services/aigc/text2image/image-synthesis`，仅异步任务，`prompt` 入参 / `results` 出参）；wan2.6-t2i / wan2.6-image 只应答 multimodal-generation 新协议（`generate` 同步打 `/services/aigc/multimodal-generation/generation`，`jobs` 异步任务打 `/services/aigc/image-generation/generation`，`messages` 入参 / `choices` 出参，出参归一化为统一 `Job.results`） |
| `client.videos` | `VideoSynthesis` | `generate` / `jobs.create/fetch/wait`；同端点（`/services/aigc/video-generation/video-synthesis`，仅异步任务，`video_url` 出参）下按模型名路由输入协议——wan2.6 及以下走旧首帧协议（`img_url` 入参，公开名 `image_url`）；wan2.7-i2v 只应答新 i2v 协议（`media` 入参：`first_frame` / `last_frame` / `driving_audio` / `first_clip` 类型化素材列表、每种至多一个、必填，`image_url` 归一为单个 `first_frame` 素材项） |
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
| `client.rl` | （CLI 的 `dashscope rl` 能力 SDK 化） | `register_functions/test_functions/upload_data/run/get/cancel/logs/list`；**复用 1.x 既有实现，不重写**——自有逻辑多，为避免影响 1.x，2.0 仅在 errors 与底层 HTTP 调用上对齐 v2，业务逻辑一律不动（见 3.7） |
| ~~`client.beta.assistants`~~ | `Assistants/Threads/Messages/Runs/Steps` | **整个 Assistants API 族确定下线，不迁入 2.0**：1.x 原模块冻结保留至 3.0 随旧代码删除，2.0 资源树中无此项（托管 Agent 场景由 `client.agentstudio` 承接） |
| `client.agentstudio` | `dashscope.agentstudio`（agents/sessions/events，含流式事件） | 托管 Agent API；1.x 子 SDK 代码已较新、与 2.0 风格接近，**仅合并入统一 Client，不动原来的代码和逻辑**；errors 与底层 HTTP 调用与 2.0 的对齐由专门环节统一处理（见 M3 任务 3.12：挂载层映射包装，不触碰 1.x 代码，不与 §2.4 冲突） |

约定：

- **长任务统一模式**：`jobs.create → fetch/wait → result`，`wait` 内置退避轮询；所有任务资源返回统一的 `Job` 对象（`id/status/result/error`）。
- **同资源多协议按模型路由**：同一资源下不同模型落在不同服务端点/协议时，按模型名自动路由，不向用户暴露端点选择参数（与 `chat.completions` 含媒体 part 自动改走多模态端点同一约定）。当前应用点：`client.images`（见上表：wan2.6-t2i / wan2.6-image → multimodal-generation 协议，其余 → 旧 text2image 端点）、`client.videos`（wan2.7-i2v → media 新 i2v 协议，其余 → 旧 `img_url` 首帧协议）。新协议模型清单分别以 `v2/resources/images.py` 的 `MULTIMODAL_MODEL_PREFIXES`、`v2/resources/videos.py` 的 `MEDIA_MODEL_PREFIXES` 为准，新模型上线时前缀扩充。
- **流式统一模式**：`stream=True` 一律返回类型化迭代器（`Stream[Chunk]` / `AsyncStream[Chunk]`，用 `@overload` 让流式/非流式返回类型在静态层面可区分）；提供 `stream.text_stream` 纯文本迭代器与 `get_final_message()` 聚合助手；**流内错误统一抛异常，不 yield 错误对象**。ws 实时接口用上下文管理器。例外：`client.apps` 在 `raise_on_error=False` 兼容模式下以 `is_error=True` 终止事件透出流内错误（对齐 1.x `Application` 流式返回值语义），不回抛；chat/multimodal 无此例外。
- **类型化**：入参出参全部 Pydantic 模型，`**kw` 仅保留 `extra_headers/timeout` 等逃逸舱；响应提供便捷属性（如 `resp.output_text` 一行取文本）；内置 `.pyi`。
- **错误模型**（对齐 OpenAI/Anthropic，见对比调研；**规范正文见 §2.7**）：
  - 默认异常驱动：HTTP/业务失败抛 `DashScopeError` 层级，按状态码细分为 9 个扁平类 `BadRequestError(400)` / `AuthenticationError(401)` / `PermissionDeniedError(403)` / `NotFoundError(404)` / `RequestTooLargeError(413)` / `RateLimitError(429)` / `InternalServerError(500)` / `ServiceUnavailableError(503)` / `GatewayTimeoutError(504)`；另有传输层 `APIConnectionError` / `APITimeoutError`、内容审核 `ContentFilterError`、SDK 自报 `SdkError`；
  - 异常携带 `status_code / code / message / request_id / body / response.headers`；
  - 成功响应同样携带 `resp.headers`（网关原始 HTTP 响应头，1.x `GenerationResponse.headers` 对齐）：由 `types.common.with_headers` 在每个非流式 HTTP 成功返回点填充（`with_headers` 按结构化 Protocol 接受任何声明了 `headers` 字段的模型，不限于 `ResponseBase` 子类，如 audio `HttpSynthesis`）；流式经 `Stream.headers`（chat）、`AppStream.headers`、`HttpSynthesisStream.headers` 透出，chat 另由 `get_final_message()` 回填到聚合结果；ws 资源无 HTTP 信封，保持空 dict；成功侧 `resp.headers` 是小写键的普通 dict（同名多值头以 `", "` 拼接，保证 `model_dump()` 可序列化），而 `errors.response_error` 放在异常上的是活的 `httpx.Headers`（大小写不敏感）；"HTTP 200 但业务失败"（如任务 FAILED）抛出的异常同样携带 `headers`，且 `err.body` 只含 wire/作业快照，不会把 `headers` 字段带进去（调用方常直接记录 `err.body`）；
  - `"Throttling"`、`"InvalidApiKey"` 等字符串错误码收敛为可 import 的枚举常量（`dashscope.v2.errors.ErrorCode`，13 项；包根不导出），完整对照表见 §2.7；
  - 兼容开关：`Client(raise_on_error=False)` 退回 1.x 返回值模式；
  - 保留"HTTP 200 但业务失败"的双层检查（DashScope 网关特性，现有 `ensure_ok` 语义上移到 SDK）。
- **内置重试**：默认 `max_retries=2`，429/5xx/连接错误自动重试，解析 `Retry-After`，指数退避 + 抖动；`Client(max_retries=N)` / `with_options()` 可调。
- **传输层**：httpx 统一同步/异步（连接池复用），替换 requests + aiohttp 双栈；超时支持 connect/read 分离。
- **日志**：`DASHSCOPE_LOG=debug` 一键开启；记录 request_id、HTTP 状态码、重试原因与等待时间；SensitiveHeadersFilter 过滤 Authorization，防止密钥入日志。

### 2.3 配置

```python
dashscope.Client(api_key=..., base_url=..., protocol=..., timeout=..., max_retries=...)
```

优先级：显式参数 > 环境变量（`DASHSCOPE_API_KEY` 等） > `~/.dashscope/config.toml`。1.x 的 `dashscope.api_key` 等模块级全局量原样冻结保留（供存量代码使用，deprecate 标注）；2.0 新代码一律用 `Client(...)` 显式配置，两者互不影响。

### 2.4 兼容与长期共存策略（零接触物理隔离）

前提修正：原"两个 minor 版本后移除"不成立，1.x 预计 1–2 年内无法下线。曾评估"单代码库、双门面共享核心"方案，结论是共享核心的行为渗透防护（语义隔离 + 契约测试）成本高于收益——**2.0 采用零接触物理隔离：1.0 代码一行不动、整体冻结，2.0 完全独立新建，HTTP/传输不复用**。

```
用户代码（1.x 写法） ── dashscope.Generation.call / dashscope.cli（冻结，不动）
用户代码（2.0 写法） ── dashscope.Client / dashscope.chat（包根导出）
                                      │
                          dashscope/v2/（独立实现：httpx transport / protocols / resources）
```

**架构规则**

1. **1.x 代码冻结**：现有模块（`Generation` 等类方法接口、`dashscope/cli/` 旧命令树、requests/aiohttp 传输）保留但不再修改（仅安全补丁走 1.x 维护分支）；不建兼容层、不做转调——用户继续 `import dashscope` 用旧写法，行为逐字节不变。
2. **2.0 独立目录**：全部新实现放 `dashscope/v2/`（client / resources / protocols / transport / errors），不与 1.x 共享任何请求代码（例外见规则 4）；依赖共存（requests/aiohttp 留给 1.x，httpx 供 2.0），默认完整安装，不做 extras 拆分。
3. **包根只导新面**：`dashscope/__init__.py` 导出 2.0 公开 API——`dashscope.Client` / `dashscope.AsyncClient` / `dashscope.chat()`——用户 import 路径无 `v2` 字样；`v2` 只是内部目录名，不是公开命名空间。
4. **白名单式共享**：唯一允许的共享是零逻辑常量（端点 URL、模型名常量等），逐项白名单评审；任何含行为的代码一律不共享。**例外**：`rl` 与 `agentstudio` 两块直接复用 1.x 既有实现代码，不重写。**基本原则：两块的业务逻辑一律不动**，仅对齐 errors 与底层 HTTP 调用——`rl` 自有逻辑多，2.0 只建挂载层；`agentstudio` 已较新，仅合并入 Client（见 M3 任务 3.7/3.8）。**白名单登记**（2.0 期间对 1.x 的全部实际触碰，此外零触碰）：① 包根 `dashscope/__init__.py` 导出 2.0 公开面（规则 3 授权）；② `0c58a74` 将 `dashscope/finetune/reinforcement/common/constants.py` 的 `LOGGER_NAME` 默认值从 `dashscope` 改为 `dashscope.finetune.reinforcement`（rl 日志挂独立 logger 名，避免与包根 logger 配置互相干扰）。
5. **新需求只进 2.0**：新接口、新参数、新模型能力一律只进 `v2/` 与新命令树；1.x（含旧 CLI 命令）不新增支持，文档引导用户为新能力迁移。
6. **测试互不影响**：现有 1.x 测试套件原样保留作冻结回归（不改、不扩）；2.0 测试全部新建于 `tests/v2/`。
7. **下线即删目录**：3.0 时整目录删除 `dashscope/v2/` 之外全部旧代码并上移，无兼容层代码残留——删除动作机械、无风险。

**acli 独立于新旧隔离**：`dashscope/acli/` 代码量大，2.0 原样复用、不重写；它既不在 1.x 冻结集、也不进 `v2/`，作为第三条线独立演进（提交约束见 §7"开发与提交约束"），M5 仅做底座切换与 tools 注册。3.0 包结构重整时需同步迁移适配（见后续任务 8.1）。

**CLI 侧**

8. 旧命令（`generation create`、`multimodal_conversation create` 等）不动、继续调用旧模块；新命令树（`chat`/`config`/`auth`/`images`...）单独建，只 import `v2/`。
9. 新旧命令并存于同一 `dashscope` 入口。`ft`/`rl` 等习惯名由新命令树直接承接（参数/输出按 §4 改进，向后兼容旧参数名）；其余旧命令名（`generation` 等）原样冻结，3.0 随旧代码一并删除。

**下线流程（遥测驱动，不设死线）**

10. 旧接口挂 DeprecationWarning（Python 默认不显示，不扰民；此为唯一允许的 1.x 触碰），文档/官网推迁移指南；
11. 埋点统计旧接口调用占比（包版本 + 入口名，脱敏），每季度评估；
12. 移除条件：旧接口调用占比低于阈值（如 <5%）且无大客户依赖反馈 → 3.0 整目录删除旧代码，而非 2.x 中途砍掉；
13. 共存期内 1.x 与 2.0 接口使用同一版本号发布（`dashscope>=2.0` 即含两者），不做双包分发，避免用户选型困惑。

**成本与收益**：代价是包内短期存在两套 HTTP 依赖与部分重复实现（包体积略增、极个别 bug 需修两处）；收益是 1.x 回归风险在结构上归零（不存在共享代码，渗透无从发生）、无兼容层开发维护成本、新需求迭代不触碰旧代码、3.0 下线是一次机械删除。参考先例：Azure SDK track1/track2、boto/boto3、openai v0→v1 均为物理隔离式换代。

### 2.5 Quickstart 体验（对齐 OpenAI 的 5 行上手）

```python
import dashscope

resp = dashscope.chat("qwen-plus", "讲个笑话")   # 模块级极简入口（内部单例 Client）
print(resp.output_text)                            # 一行取文本
```

- `dashscope.chat(model, prompt, **kw)` 为官方 Quickstart 首推写法，文档首页即用此例；
- 进阶用户切到 `dashscope.Client()` 资源式 API，两者返回同一套 Pydantic 模型。

### 2.6 命名规范（长期约束）

**本节是硬约束**：`dashscope/v2/` 下新增任何资源、方法、参数或响应模型都必须遵守，评审时按此逐条核对。规范来自对全量 38 个响应模型 / 90+ 公开方法的横向评审（诊断过程与逐条改名清单见 [`dashscope-2.0-naming-review.md`](dashscope-2.0-naming-review.md)，本节只落结论）。

命名的唯一目标：**同一个语义在全 SDK 只有一个名字，客户不需要猜**。因此规范以"消灭同义词"为主线，而不是以"取个好名字"为主线。

**动词收敛到 8 类**（不允许出现表外动词；表内动词的阻塞语义固定）：

| 语义 | 动词 | 阻塞 | 适用 |
| --- | --- | --- | --- |
| 同步推理 | `create` | 是 | chat / multimodal.chat / multimodal.embeddings / embeddings / rerank / apps.completions |
| 长任务提交 | `jobs.create` | **否** | images / videos / audio.asr / embeddings / fine_tunes / rl |
| 长任务查询·等待 | `jobs.fetch` / `jobs.wait` | 否 / 是 | 同上 |
| 一步式语法糖 | `generate` / `transcribe` | 是 | images / videos / audio.asr（= `create` + `wait`） |
| 持久资源 CRUD | `get` / `list` / `delete` / `cancel` / `scale` | 否 | fine_tunes / deployments / models / files |
| 双工会话 | `recognize` / `connect` | ctx manager | audio.asr / audio.realtime |
| 纯计算·文件 | `count` / `upload` | 是 | tokenization / files |
| 流式 | `stream=True` 或 `.stream()` / `.events()` | — | 见下"流式形态" |

**阻塞规则（P0）**：`<res>.create()` / `jobs.create()` **永不阻塞**，只提交并返回任务快照；`wait()` 才阻塞。裸领域动词（`generate` / `transcribe`）是 `create + wait` 的语法糖，**必须在 docstring 里明写它会阻塞**，否则客户会把两种 `create` 当成一回事。

**入参规范**：

| 类别 | 规范 |
| --- | --- |
| 模型 | 一律 `model: str`，不允许 `model_name` 之类变体 |
| 任务主键 | 对外统一 `job_id`；服务端字段 `task_id` 保留为双向别名属性 |
| HTTP 超时 | `timeout: Optional[float]`，单次请求；**所有公开方法都必须有，且显式写进签名，不许藏在 `**kwargs` 里**。唯一例外见下"rl 例外" |
| 轮询预算 | `wait()` 上 `timeout` = 等待总预算；阻塞式一步法上 `timeout` = 总预算、提交那一次请求另用 `submit_timeout` |
| 服务端透传 | `parameters=`（并入线缆 parameters 段）/ `extra_input=`（并入 input 段）；**禁止** `extra_parameters` 之类第三、第四个同义名 |
| 分页 | `page_no` / `page_size`，默认值全线一致；禁止 `page=` |
| 文件 URL | 多文件端点 `file_urls: List[str]`，单文件端点 `file_url: str`；单复数跟随端点真实入参个数，禁止裸 `url=` |
| 必填主输入 | keyword-only（顶层 `dashscope.chat()` 一行式入口例外） |
| 低频服务端参数 | 收进 `parameters=`，在 docstring 里列常用键，不铺到签名上 |
| 测试注入钩子 | 不进公开签名：用 `_` 前缀（如 `wait(_sleep=...)`）或 `with_options()` |

**rl 例外**：`client.rl` 的四个 1.x 直通方法（`jobs.create` / `register_functions` / `test_functions` / `upload_data`）按 §2.4 规则 4 复用 1.x 业务逻辑，超时由 1.x 内部控制，**不接受 `timeout=`**。这是全 SDK 唯一的例外，必须在接口文档里显式写明，不能让"所有方法都支持 timeout"成为一句假承诺。

**输出字段规范**：

| 类别 | 规范 |
| --- | --- |
| 时间戳 | `created_at` / `updated_at` / `submitted_at` / `scheduled_at` / `finished_at`；禁止 `gmt_*`、`*_time` 混用 |
| 用量 | 单一 `Usage` 类（`types/common.py`）全 SDK 复用，禁止再造同构 Usage 类或退化成裸 dict |
| 分页 | `page_no` / `page_size` / `total_count`；禁止 `total` |
| 序号 | 一律 `index`；禁止 `text_index` |
| 模型标识 | 一律 `model`；微调产物 `finetuned_model`，部署产物 `deployed_model`；禁止 `model_name` / `finetuned_output` / `version` |
| 任务标识 | 对外 `job_id`，`task_id` 为服务端别名 |
| 错误字段 | 所有顶层响应继承 `ResponseBase`（`request_id` / `status_code` / `code` / `message` / `is_error`）。`is_error` 由 `ResponseBase` 从非空 `code` **自动推导**（与 `errors.response_error` 同一判据），任何资源都不可能忘记设置它 |
| 信封投递 | payload 是 dict 时用 `flatten_envelope(body)` 整体摊平；payload 是列表或 `data` 下的单条目时（`files` / `models`），用 `response_envelope(body)` 把信封字段单独挂到解析出的模型上 |
| 响应形状 | **一律扁平**：服务端的 `output` / `data` 信封在 `_parse` 里拆掉（`flatten_envelope`），客户不写 `.output.` / `.data.` 这一跳 |
| 服务端字段别名 | 对外名与服务端线缆名不同时，用 Pydantic `Field(validation_alias=AliasChoices(对外名, 服务端名))` 做双向可读，**不改线缆报文** |

**信封规则的三个例外**（都不是顶层 HTTP 响应，故不继承 `ResponseBase`）：

1. `FineTuneEvent` —— SSE 流帧，不是响应；自带 `is_error` 标记错误帧。
2. `RegisteredFunctions` / `UploadedData` —— `client.rl.register_functions` / `upload_data` 的返回，由 1.x 返回值元组构造，**背后不存在 HTTP 信封**（1.x 失败即抛异常，这也是这两个方法没有 `timeout=` 的同一原因）。给它们加 `request_id` / `code` 只会是永远为空的装饰。
3. 嵌套条目上的信封字段保持为空：`Deployment` / `File` / `Model` / `FineTune` 既是 `get` 的顶层返回、又是 `*List` 的条目类型（"一个形状学一次"），信封只在顶层那份上填充——服务端每次响应报一次，而不是每个条目报一次。

**两条硬禁止**：

1. 公开响应模型上不得出现裸 `Dict[str, Any]` / `Any` 字段。唯一豁免是服务端字段因服务而异、无法类型化的 `Job.output` 与 `FineTuneEvent.output`，且**必须同时提供类型化视图**（`Job.results` / `image_urls` / `video_url` / `result_url`）。
2. 不得把阿里内部命名习惯泄漏到公开 SDK：`gmt_*`、`is_deleted`、`biz_*` 一类内部 DB / 中台风命名一律改名。例外是**服务端本身就是公开字段名**的情况（如 `apps` 的 `biz_params`、多模态向量的 `res_level`）——此时改名会造成 SDK 与线缆的新不一致，处理方式是在 docstring 里解释语义，而不是改名。

**流式形态**：chat 增量文本用 `stream=True` → `ChatStream`（`ChatCompletionChunk`）；应用增量输出用 `stream=True` → `AppStream`（`AppCompletion`）；训练事件用 `fine_tunes.stream_events()` → `FineTuneEvent`；音频用 `audio.tts.stream()` → `bytes`。websocket 双工会话不属于流式响应，用 `.events()` 迭代服务端事件。流中出错不静默：默认抛异常族，`raise_on_error=False` 时产出一个 `is_error=True` 的对象后结束迭代。

**新增资源时的自检顺序**：① 动词是否在 8 类表内、阻塞语义是否与表一致 → ② `timeout=` 是否显式在签名上 → ③ 响应是否继承 `ResponseBase` 且已扁平化 → ④ 时间戳 / 分页 / 序号 / 标识符字段名是否与规范一致 → ⑤ 有没有引入表外同义词 → ⑥ 若按 §2.4 规则 4 复用 1.x 实现，1.x 异常是否已翻译进 v2 异常族（见 §2.7）。

### 2.7 错误模型（目录驱动，长期约束）

**本节是硬约束**：`dashscope/v2/errors.py` 里的异常类名、HTTP 状态与 SDK 侧错误元数据一律**不在 SDK 中手写**，而是从生成的错误目录 `dashscope/v2/_error_registry.py` 派生（`_` 前缀按 v2 约定标记内部模块：它不从 `dashscope/v2/__init__.py` 导出，只被 `errors.py` 与测试引用）。改写背景、逐条决策与验证数据见 [`dashscope-2.0-errors-registry-plan.md`](dashscope-2.0-errors-registry-plan.md)。

**目录是唯一事实源**（上游独立仓库 `dashscope-error-registry`，vendored 版本 `1.0.6` / commit `56b7208`）：

| 目录表 | 条数 | SDK 用途 |
| --- | --- | --- |
| `PUBLIC_ERRORS` | 16 | 每条的 `error_code` 就是一个异常类名，`status_code` 就是它的状态码 |
| `SDK_ERRORS` | 32 | 客户端自报错误的 `code` / `solution` / `allow_retry` |

`_error_registry.py` 是上游 `tools/generate.py` 的产物、文件头即写明 DO NOT EDIT。**这句话的含义是"不得手改目录内容"**——增删错误码、改状态码、改文案都必须回上游 registry 仓库改 `proto/error_registry.proto` 与 `data/*.textproto` 再重新生成，不能在 SDK 里就地打补丁。

但它**不是**上游原文的逐字节副本，vendored 这份带有**已登记的本地适配**，整份替换会把它们全部回退，升级时必须逐条重新打上（清单见 registry-plan §1 D1）：

1. 上游第三张表 `InternalError`（277 条服务端组件级错误）**没有 vendor 进来**——它对 SDK 没有投影（proto 里 `InternalError.external` 指回某个 `PublicError.key`，最终还是落到那 16 条），且整份 vendor 会让 `dashscope/v2/` 第一次出现 CJK（388 行）并带来 97 处 black 拆不开的 E501。
2. `vars` 形参改名 `substitutions` / `variables`——上游用 `vars` 遮蔽内建名，而仓库的 pylint disable 列表里**没有** W0622。
3. 隐式 Optional 补成显式 `Optional[Dict[str, str]]`。
4. 按仓库 pin 的 black 23.3.0 / `--line-length=79` 重排。
5. provenance 从上游独立的 `_version.py` 内联进文件头；文件名加 `_` 前缀。

`tests/v2/test_error_registry.py` 钉住其中的文件头、provenance 与裁剪三项（`test_generated_header_survives_vendoring` / `test_catalog_size_and_provenance` / `test_vendored_copy_is_pruned_to_the_tables_v2_consumes`），`format_msg(substitutions=...)` 系列用例钉住第 2 项——**误将上游原文整份覆盖进来会立刻红**。

**派生规则**：9 个状态类是 `APIStatusError` 的**扁平兄弟**（互为兄弟、不再是 `InternalServerError` 兜一堆 5xx），`status_code` 逐个从目录读出而非写字面量——目录改状态码，SDK 随之改变，无需改代码。未收录的状态码（418 / 502 / 507…）落 `APIStatusError` 本身，**不猜**成 `InternalServerError`：猜出来的类会让 `except InternalServerError` 抓到并非 500 的失败，且实例的 `status_code`（502）与类声明的状态码（500）互相矛盾。重试判定与此无关——`transport/_retry.py` 按**原始状态码**决定是否重试（408 / 409 / 429 / 500 / 502 / 503 / 504），不看异常类。

**三层来源，一个异常族**：

| 来源 | 判据 | 落到 | `status_code` |
| --- | --- | --- | --- |
| HTTP 层 | 非 2xx 状态码 | `_STATUS_MAP[status]` | 该状态码 |
| 业务层 | HTTP 200 但 body 带 `code` | `_CODE_MAP[code]` | `200` |
| SDK 层 | 根本没有响应 | `SdkError` | `None` |

`APIStatusError` 因此读作"**网关声明的失败**"（前两层都算），而不是字面的"非 2xx 响应"；实例的 `status_code` 说明是哪一层。这也是 `ContentFilterError` 一处 `except` 能同时覆盖两种上报方式的原因。

**crosswalk 与目录 key 是两个命名空间**：目录 key（`invalid_api_key`）是标准词表，网关线缆码（`InvalidApiKey`）是百炼的历史事实，两者不互相替代。`_WIRE_TO_PUBLIC` 是 SDK 自己维护的对照表，覆盖全部 13 个 `ErrorCode` 成员；它只决定"这次失败对应**哪条目录项**"，**不决定异常类**——类永远由该目录项的 `error_code` 给出。

| `ErrorCode`（线缆码） | 目录 key | 异常类 |
| --- | --- | --- |
| `InvalidApiKey` | `invalid_api_key` | `AuthenticationError` |
| `AccessDenied` / `AccessDenied.Unpurchased` / `Arrearage` | `permission_denied` | `PermissionDeniedError` |
| `InvalidParameter` | `invalid_request` | `BadRequestError` |
| `Model.NotExist` | `resource_not_found` | `NotFoundError` |
| `DataInspectionFailed` | `content_policy_violation` | `ContentFilterError` |
| `Throttling` / `.RateQuota` / `.AllocationQuota` | `rate_limit_exceeded` | `RateLimitError` |
| `InternalError` | `internal_error` | `InternalServerError` |
| `ServiceUnavailable` | `service_unavailable` | `ServiceUnavailableError` |
| `GatewayTimeout` | `request_timeout` | `GatewayTimeoutError` |

`Arrearage`（欠费）走 `permission_denied` 而不是 `insufficient_quota`：欠费是**请求被拒**，不是配额用尽，而目录里 `insufficient_quota` 是 429、意味着可重试。

**唯一的细化：`ContentFilterError(BadRequestError)`。** 内容审核在线缆上是 `DataInspectionFailed`、在目录里是 `content_policy_violation` → 400 → `BadRequestError`。V2 需要一个能被单独 catch 的类，做法是**子类化**而非新增目录外状态：`ContentFilterError` 是 `BadRequestError` 的子类，`except BadRequestError` 依然抓得到，目录不变式不破。该不变式由 `test_wire_code_never_leaves_the_catalogued_class` 对 crosswalk 每一行钉住。**将来任何细化都必须走同一条路**：子类化目录已命名的类，不得引入目录里没有的 `status_code`。

**`SdkError` 没有状态码，这不是偷懒。** 目录的客户端 SDK 子图里 `SdkError` 只有必填的 `solution`，**没有** `external` 字段指回 `PublicError`——标准本身就没给 SDK 侧错误任何公开投影。所以 `status_code` 保持 `None`；凭空造一个等于给标准里没有的东西加上含义。它的 `code` 是目录名（如 `sdk.agentic_rl.OSSUploadError`），`solution` 是随目录发布的修复建议，另有 `allow_retry` 供调用方自行判断（**不接进 `_retry.py`**：那是 HTTP 层，与 SDK 层不同层）。

**1.x 子系统挂载层的翻译义务**：按 §2.4 规则 4 复用 1.x 实现的资源，其 1.x 异常必须翻进 v2 异常族，否则用户写 `except DashScopeError` 抓不到。做法是 `error_from_sdk_exception(exc, family=...)` 两级查目录：`<family>.<ClassName>` 命中就用该条，否则退到 `<family>.Error` 家族兜底。**不嗅探 message 文本、也不嗅探 1.x 的数字 `error_code`**——两者都是 1.x 内部实现细节。捕获范围只取能覆盖该子系统全部异常类的最小基类元组，捕得更宽会把挂载层自己的 bug 贴成 `SdkError`。当前状态：`rl` 已接线（family `sdk.agentic_rl`，8 个直通方法 sync / async 全覆盖）；`agentstudio` 待 M3 任务 3.12（其挂载层直接返回 1.x client 对象，需要透明代理才能拦截）。

**两级查目录依赖 `<family>.Error` 兜底存在，而 `sdk.agentstudio` 没有这一条。**目录里 `sdk.agentstudio.*` 只有 4 条（`APIConnectionError` / `APITimeoutError` / `StreamError` / `StreamClosedError`），全是非 HTTP 的本地/流层错误；对这 4 条之外的类调用 `error_from_sdk_exception` 会 `KeyError`，反而盖掉原始异常。agentstudio 的 HTTP 层（1.x `APIStatusError` + 8 个子类）自带真实 `status_code`，**属于状态映射、不属于 `SDK_ERRORS`**，要走 `error_from_status`。所以 agentstudio 的翻译是**两条路**，不能照 `rl` 的单路写法。这两个事实已被 `tests/v2/test_errors.py` 与 `tests/v2/test_error_registry.py` 钉死，接线时要改的是用例、不是绕过它们。

**新增错误相关代码时的自检顺序**：① 类名与状态码是否从目录派生而非手写 → ② 目录里有没有这一条（没有就去上游 registry 仓库提，**不是**在 SDK 里造） → ③ 细化是否用子类化、有没有逃出目录命名的类 → ④ 有没有泄漏目录外的 `status_code` → ⑤ 新增的线缆码是否进了 crosswalk 且有对应用例 → ⑥ 复用 1.x 的路径是否已翻译。

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
   - 采样参数：`temperature/top_p/max_tokens/stop` 直接映射；无对应字段的参数默认透传到请求体（`strict=False`，保证厂商新参数如 reasoning/thinking 不被 SDK 发版节奏卡住），`strict=True` 时对未知参数抛错。
   - 响应：统一翻译回 SDK 的 `ChatCompletion` 对象（含 usage、finish_reason）。
2. **入站（服务端，可选后置）**：`dashscope serve` 起一个本地 OpenAI/Anthropic 兼容端点，把请求翻译为 native 调用——让存量 OpenAI SDK 用户零改造用 dashscope。此阶段先留接口不实现。

### 3.2 与 acli providers 的关系

acli 现有 `providers/` 已有多协议接入逻辑，目标仍是"协议翻译收敛到 SDK、避免两处漂移"，但**实现路径受同步机制约束**：`dashscope/acli/` 是 agenticCLI 仓库 `src/acli` 的**单向同步副本**（`scripts/sync_acli.sh`，同步时先 `rm -rf` 再拷贝），在副本上做本地修改会被下次同步整体覆盖，且副本一旦依赖 `dashscope.v2` 也无法回流独立仓库 agenticCLI。因此：

- 协议翻译代码**只下沉到 SDK `v2/protocols/`**（M2 任务 2.2/2.3），2.0 期间**不改同步副本 `dashscope/acli/providers/`**；
- acli 切换为复用 SDK（删除 providers 重复实现）推迟到 **M5 任务 5.1**（底座切换）一次性处理，该改动按"开发与提交约束"单独 commit 并及时 cherry-pick 回 agenticCLI，保持两 repo 一致；3.0 迁移（8.1）时最终收敛。

### 3.3 第三方厂商模型（接入百炼）的兼容结论

对 MiniMax / DeepSeek / Baseten / Kimi 的调研（详见 `md/vendor-sdk-research-compat.md`）结论：**不需要为任何厂商做专属 adapter**——四家对话协议已全部收敛到 OpenAI 兼容（MiniMax/Baseten 另支持 Anthropic），百炼网关再收敛为 compatible-mode / native 两条通路，§3 的三协议 adapter 天然覆盖。需要做的只是六个通用落点：

1. **reasoning/thinking 参数**：`extra_body` 透传逃生舱 + 高频参数（`enable_thinking`/`reasoning_effort`）显式化 + 响应模型含 `reasoning_content` 字段（流式单独事件化）；
2. **模型目录可发现性**：`client.models.list(provider=...)` 按来源过滤，文档给"厂商 → 百炼 model id"对照表；
3. **错误映射**：`ErrorCode` 枚举覆盖第三方模型常见失败（内容审核/配额/模型不存在）；
4. MiniMax 双 Key 形态：百炼统一 key，无需处理；
5. MiniMax 视频/声音克隆等私有能力：若上百炼走 native 异步任务（§2.2 长任务模式），不是协议兼容问题；
6. CLI 示例补厂商特有参数用法（`--extra-body '{"thinking":...}'`）与 compatible-mode/native 选择指引。

---

## 4. dashscope CLI 2.0

**定位**（来自 CLI 对比调研）：OpenAI 已于 2026-05 删除 API CLI、Anthropic 从未有过，"SDK 内嵌 API 管理 CLI"品类只剩 DashScope。对标物是 **Codex cloud + aws cli 的混合体**：不与 Agent 类 CLI 比对话体验（那是 expert/acli 的赛道），而是把"AI 平台长流程管理"（微调→部署→RL→视频批处理）做成无可替代的差异点；`chat create` 这类一条 curl 可替代的调用仅作便捷入口。

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
dashscope config        set|get|list          （api_key/protocol/base_url 持久化）
dashscope expert        ...                    （转发到 Agentic CLI，见 §5）
```

### 4.2 体验改进点（相对现状）

- **认证命令**：`dashscope auth login/whoami/logout`，whoami 验证 key 有效性并以退出码表达（0=有效），CI 可脚本化探测（对标 `codex login status`）。
- **Shell 补全**：Typer `add_completion=True`，提供 bash/zsh/fish 补全安装命令（现状 `cli/__init__.py` 显式关闭，一行可修）。
- **全局参数**：`--api-key -k`、`--protocol`、`--base-url`、`--output json|table|yaml`（全局化，所有命令含 `ft list`/`rl list` 支持 JSON + cursor 分页）。
- **输入友好**：`--prompt` 直接传文本；`--messages @file` 从文件读；支持管道 `cat msg.json | dashscope chat create -`；图片等本地文件自动经 `oss upload` 换 URL（复用现有 `cli/oss.py`）。
- **流式输出**：TTY 下渲染增量纯文本（修复现状打印 dict repr 的问题）；`--output json` 时每 chunk 一行合法 JSON（NDJSON），可被 jq 消费；**流式中途出错必须 exit 1**（修复现状 `cli/generation.py` 流式分支只打印错误仍 exit 0 的 bug）。
- **退出码分级**：0 成功 / 1 服务端错误 / 2 认证错误 / 3 参数错误 / 4 限流，脚本免解析 stderr。
- **任务轮询内置**：`images generate --wait` 一条命令等结果；所有长任务统一 `--wait` + Rich Progress。
- **配置文件**：`~/.dashscope/config.toml` 存默认 model/workspace/输出格式，`-c key=value` 单次覆盖（优先级：命令行 > 环境变量 > 配置文件）。
- **隐藏别名保留**：`fine-tunes`→`ft`、`agentic-rl`→`rl`，不破坏存量脚本。

实现：Typer 保留；命令层由 codegen（§6）从元数据生成骨架，每个命令只写参数渲染与输出格式化。

---

## 5. dashscope expert（Agentic CLI）

基于现有 `dashscope.acli` 演进，不改交互面，只做三层整合：

1. **底座切换**：acli 整体原样复用（代码量大，2.0 不重写、3.0 随包结构重整迁移，见 §2.4 与后续任务 8.1）；仅模型调用从 acli 自有 providers 切到 SDK 2.0 Client（§3 协议兼容自动生效，`/provider`、`/model` 及 `load_config(provider, base_url)` 语义不变）。
2. **入口与命名**：新增 `dashscope expert` 顶层命令（等价 `dashscope --tui` 等现有启动参数）；`run_interactive` SDK 入口保留。
3. **dashscope 能力即工具**：把 SDK 资源（文生图、ASR/TTS、embedding、文件上传等）注册为 expert 内置 tools/skills，让 agent 一句话调全量 dashscope 能力——这是 expert 相对通用 agentic CLI 的差异点。
4. **问答 prompt 的 1.x/2.0 区分**（待办，见 M5 任务 5.4）：expert 回答 dashscope 相关编码问题时识别代码语境版本，新代码推荐 2.0、存量 1.x 不强行迁移、新能力标注仅 2.0；片段可插拔、2.0 门面落地后才注入。

现有命令面（`/session`、`/memory`、`/mcp`、`/tts`、`/cron` 等，见 interfaces 清单第三部分）全部保留，不在本期重构范围。

---

## 6. 接口自动化生成（阶段三）

**依据**（见 `md/conclusion-sdk-architecture-stainless.md`）：OpenAI/Anthropic SDK 均由 Stainless 从 OpenAPI spec 自动生成（目录结构逐文件同构），spec 变更 → CI 自动重新生成 SDK 发 PR，天级送达。DashScope 手写实现的代际差距（30+ 个 `Any` 参数、手工 kwargs 搬运、SDK/CLI/文档漂移）正是手写 vs 流水线的差距——2.0 追平的路径不是继续手写补丁，而是把百炼 API 沉淀成 spec 接入生成器（Stainless 商业服务，或 oapi-codegen/自研模板）。

### 6.1 元数据源

`specs/*.yaml`（OpenAPI 3 扩展）描述每个 API：路径、方法、参数 schema、任务型/流式/普通三种形态标记、协议映射注解（openai/anthropic 对应字段）、CLI 命令名与示例。spec 格式**语言无关**：核心字段不含 Python 专属注解，语言特性走 `x-python`/`x-java`/`x-node` 等扩展命名空间，为后续多语言生成留基座（见后续任务 8.3）。

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
  __init__.py          # 包根导出 2.0 公开面：Client / AsyncClient / chat()
  v2/                  # 2.0 全部新实现（独立，不与 1.x 共享请求代码）
    client.py          #   Client / AsyncClient
    resources/         #   chat, images, audio, files, finetunes, rl, agentstudio...
    protocols/         #   native / openai / anthropic adapter
    transport/         #   http(sse) / websocket / task polling（httpx）
    errors.py          #   异常族；类名与状态码从 _error_registry 派生（§2.7）
    _error_registry.py #   生成的错误目录，vendored 上游制品，DO NOT EDIT
  cli/                 # 统一 CLI 命令树：2.0 新命令与 1.x 旧命令共存同一入口
                       #（实施调整：不设独立 cli2/ 目录，旧点式命令经映射翻译兼容）
  acli/                # expert（改底座，不改交互面）
  generation.py, assistants/, threads/, ...   # 1.x 旧模块，原样冻结
specs/                 # API 元数据（阶段三）
scripts/codegen.py
tests/
  （现有 1.x 测试原样保留，冻结回归）
  v2/                  # 2.0 测试
```

### 里程碑（总览）

| 阶段 | 内容 | 交付 |
| --- | --- | --- |
| M0 | **已移出 V2 范围**：1.x CLI 快修在 1.x 维护分支单独实施，清单见 [`md/m0-1x-cli-hotfix.md`](m0-1x-cli-hotfix.md) | — |
| M1 | SDK 2.0 骨架：Client、transport（httpx）、异常体系 + 错误码枚举 + 自动重试 + 日志改造、chat/embeddings/rerank/files/models 先行，native 协议 | 新 Client 可用，旧接口并存 |
| M2 | 协议层：openai / anthropic adapter + 参数翻译 + 配置 | `protocol=` 生效，兼容端点可直连 |
| M3 | SDK 全量覆盖：多模态/任务型（images/videos）、audio（含 ws）、ft/deployments/rl/agentstudio（Threads 除外：下线） | interfaces 清单 100% 映射（除下线项） |
| M4 | CLI 2.0 + `dashscope config` | 新命令合入 `dashscope/cli/` 单命令树，旧命令映射兼容保留 |
| M5 | expert 整合：acli 底座切 SDK、dashscope 能力注册为 tools | `dashscope expert` 发布 |
| M6 | codegen：specs 元数据 + 生成 SDK/CLI/文档/测试，CI 校验 | 新增接口只改 yaml |
| M7 | 2.0 GA：文档迁移 + 遥测埋点上线，进入 1–2 年新旧并存运营（1.x 冻结、2.0 独立演进）；旧代码删除作为遥测门槛项推迟到 3.0 | 2.0 正式，共存机制可运营 |

### 开发与提交约束

- **acli/expert 改动单独 commit**：所有涉及 Agentic CLI 的改动（M5 全部、expert prompt、acli tools）限定在 `dashscope/acli/**`（及其专属测试）范围内单独提交，不与其他模块混提，便于后续 cherry-pick 同步回 agenticCLI 仓库。
- **acli 是单向同步副本**：`dashscope/acli/` 由 `scripts/sync_acli.sh` 从 agenticCLI 仓库 `src/acli` 同步而来（先删后拷 + import 重写），本地对副本的修改会被下次同步整体覆盖。因此 2.0 期间原则上不改 `dashscope/acli/**`（providers 收敛见 §3.2 / M2 任务 2.5）；M5 的必要修改必须单独 commit 并及时 cherry-pick 回 agenticCLI，避免两 repo 长期分叉。
- 其余模块按任务表粒度提交，每项 ≈ 一次 PR。

### 里程碑任务分解（每项 ≈ 一次 PR）

#### M0 — 已移出本文档

1.x CLI 快修清单移至 [`md/m0-1x-cli-hotfix.md`](m0-1x-cli-hotfix.md)，在 1.x 维护分支单独实施，不属于 V2 分支工作。

#### M1 — SDK 2.0 骨架

| # | 任务 | 交付 |
| --- | --- | --- |
| 1.1 | `v2/` 目录骨架 + `Client`/`AsyncClient` 空壳 + 配置解析（参数 > 环境变量 > config.toml） | `dashscope/v2/client.py`、`dashscope/v2/_config.py` |
| 1.2 | transport 层：httpx 同步/异步、连接池、超时分级、自动重试（Retry-After + 指数退避 + 抖动） | `dashscope/v2/transport/` |
| 1.3 | 异常体系 + `ErrorCode` 枚举（含"HTTP 200 业务失败"双层检查）+ `raise_on_error=False` 兼容开关；异常类名与状态码一律从生成的错误目录派生，不手写（§2.7） | `dashscope/v2/errors.py` + vendored `dashscope/v2/_error_registry.py` |
| 1.4 | 日志改造：request_id/状态码/重试原因入日志 + SensitiveHeadersFilter + `DASHSCOPE_LOG` | `dashscope/v2/_logging.py` |
| 1.5 | 内部规范消息模型（messages/content parts 归一定义，多模态预留） | `dashscope/v2/types/messages.py` |
| 1.6 | `client.chat`：Pydantic 响应 + `@overload` + `output_text` + `stream.text_stream/get_final_message` + reasoning_content 字段 | `dashscope/v2/resources/chat/` |
| 1.7 | `client.embeddings` / `client.rerank` / `client.tokenization` | 同上模式 |
| 1.8 | `client.files` / `client.models`（含 `list(provider=...)` 过滤） | 同上模式 |
| 1.9 | 包根导出 2.0 公开面：`dashscope.Client` / `AsyncClient` / `dashscope.chat(model, prompt)`（1.x 旧导出保持不动） | `dashscope/__init__.py` |
| 1.10 | specs 元数据格式原型：chat 接口先行写 yaml（P6：codegen 主线从 M1 起步，M6 接管） | `specs/chat.yaml` |

#### M2 — 协议兼容层

| # | 任务 | 交付 |
| --- | --- | --- |
| 2.1 | `ChatProtocol` 内部接口 + native adapter 接入 v2 transport | `dashscope/v2/protocols/base.py` |
| 2.2 | openai adapter：messages/tools/采样参数翻译 + 未知参数透传（strict 开关） | `dashscope/v2/protocols/openai.py` |
| 2.3 | anthropic adapter：system 提取、tool_use/tool_result 映射 | `dashscope/v2/protocols/anthropic.py` |
| 2.4 | SSE 流式解析统一（error 事件抛异常） | `dashscope/v2/transport/sse.py` |
| 2.5 | acli providers 收敛：**不改同步副本 `dashscope/acli/providers/`**（会被 sync 覆盖且依赖无法回流，见 §3.2）——协议翻译只在 `v2/protocols/` 实现，acli 复用切换推迟到 M5 5.1 | `tests/v2/protocols/`（与 acli providers 行为一致性对照用例） |
| 2.6 | 兼容端点集成测试（百炼 compatible-mode + DeepSeek/Kimi 直连用例） | `tests/v2/protocols/` |

#### M3 — SDK 全量覆盖

| # | 任务 | 交付 |
| --- | --- | --- |
| 3.1 | `client.multimodal.chat`（多模态 content parts） | `v2/resources/multimodal/` |
| 3.2 | 长任务统一基建：`Job` 对象 + `jobs.create/fetch/wait`（退避轮询） | `dashscope/v2/resources/_jobs.py` |
| 3.3 | `client.images` / `client.videos`（generate 同步封装 + jobs 异步）；images 双协议路由：wan2.6-t2i / wan2.6-image 走 multimodal-generation 新协议（同步 + image-generation 异步任务），其余模型走旧 text2image 端点（见 §2.2 路由约定）；videos 同端点双输入协议路由：wan2.7-i2v 走 media 新 i2v 协议（`media` 类型化素材入参），其余模型走旧 `img_url` 首帧协议 | `v2/resources/images/`、`v2/resources/videos/` |
| 3.4 | `client.audio.asr`：Transcription 批转写 + Recognition 实时 ws（上下文管理器） | `v2/resources/audio/` |
| 3.5 | `client.audio.tts`（一次性 + 流式）+ `client.audio.realtime`（Omni ws）；ws 实时类资源（tts/realtime/asr.recognition）仅提供同步面，`AsyncAudio` 只挂 `asr.transcription` | 同上 |
| 3.6 | `client.fine_tunes` + `client.deployments` | `v2/resources/fine_tunes/`、`v2/resources/deployments/` |
| 3.7 | `client.rl`（现 CLI 能力 SDK 化）：**复用 1.x 既有实现，不重写**——自有逻辑多，为避免影响 1.x，2.0 只建挂载层并在 errors、底层 HTTP 调用上对齐 v2，业务逻辑一律不动 | `v2/resources/reinforcement_learning.py`（薄挂载层）+ 复用 1.x rl 实现 |
| 3.8 | `client.apps`（Application）+ `client.agentstudio` 并入统一 Client（agents/sessions/events 流式）：agentstudio 1.x 代码已较新、与 2.0 接近，**仅合并挂载入 Client，不动原来的代码和逻辑**；errors 与底层 HTTP 调用对齐 2.0 由专门环节统一处理（见任务 3.12） | `v2/resources/apps.py`、agentstudio 挂载入 Client |
| 3.9 | `interfaces 清单 100% 映射`核对表（排除下线项）+ 全量冒烟测试 | `tests/v2/smoke/` |
| 3.10 | `client.embeddings` 缺口补齐：多模态向量化 `create_multimodal`（MultiModalEmbedding）+ 批量任务 `jobs.create/fetch/wait`（BatchTextEmbedding，复用 3.2 Job 基建） | `v2/resources/embeddings.py` |
| 3.11 | 全局配置门面 `dashscope.settings`：单例收敛 api_key / base_url / websocket_base_url / workspace / protocol / timeout / max_retries / log_level，懒读环境变量（访问时读，不 import 时读），setter 即时生效（log_level 直接调 logger.setLevel）；`resolve_config` 优先级变为「显式参数 > settings 赋值 > env > config.toml」，覆盖 1.x 模块级变量（`dashscope.api_key` / `base_http_api_url` 等）的运行时配置需求 | `dashscope/v2/_config.py`（Settings）、`dashscope/v2/_logging.py`（set_log_level） |
| 3.12 | errors 对齐专项：2.0 全资源（含 agentstudio、rl 挂载层）的 errors 与底层 HTTP 调用统一对齐 `dashscope.v2.errors` 层级——在挂载层做错误映射包装，不动 1.x 业务逻辑（不与 §2.4 冲突）；并核查 `raise_on_error=False` 兼容模式在所有资源的行为一致性。**`rl` 半边已完成**：8 个 1.x 直通方法（sync + async）经 `error_from_sdk_exception` 翻成 `SdkError`，`except DashScopeError` 可捕获（见 §2.7）。**剩 `agentstudio`**：其挂载层直接返回 1.x client 对象，异常在 SDK 边界外抛出，需要透明代理才能拦截。**且只包调用层不够**——流式错误是在 `for event in stream` 迭代中途从生成器体里抛的（`StreamClosedError`、以及生产者线程 transport 异常的 `raise item` 重抛），`stream()` 早已返回，代理拦不到；而流式正是 agentstudio 的主要用法。**"只包调用层"比"完全不包"更糟**：用户会以为 `except DashScopeError` 已覆盖 agentstudio，却在最主要的路径上被漏网。完整成本清单（两条翻译路、无 `.Error` 兜底、11 个资源类 × sync/async 无共同基类、会被打破的现有断言、需要显式声明的 dunder 集合）见 errors registry plan 的 P2b 段 | `v2/resources/` agentstudio 挂载层 |

#### M4 — CLI 2.0

> 实施注记：方案由"新建独立 `cli2/` 目录"调整为"合入 `dashscope/cli/` 单命令树"（上游 1.27.x 已按单树形态发布并经 fd2519e 同步入 V2）。V2 现状 `cli/` 仍构建在 1.x 上，切 v2 底座及 chat/config 等新命令随本阶段任务实施。

| # | 任务 | 交付 |
| --- | --- | --- |
| 4.1 | `cli/` 统一命令树骨架：全局参数（-k/--protocol/--base-url/--output）+ 错误着色统一（只 import v2/） | `dashscope/cli/__init__.py` |
| 4.2 | `dashscope config set/get/list`（config.toml 持久化） | `cli/config.py` |
| 4.3 | 生成类命令：chat/multimodal/images/videos/audio（stdin/@file/管道/本地文件自动换 OSS URL） | `cli/` 各命令 |
| 4.4 | 工具类命令：embeddings/rerank/tokenization/files/models/oss | 同上 |
| 4.5 | 长流程命令：ft/deployments/rl（承接习惯名，--wait + Progress + cursor 分页 JSON） | 同上 |
| 4.6 | apps 命令 + `dashscope expert` 转发入口 | 同上 |
| 4.7 | 旧点式命令经 `_COMMAND_MAP`/`_PARAM_MAP` 翻译到新命令、隐藏别名（`ft`/`agentic-rl`）保留，与 2.0 命令并存于同一 `dashscope` 入口；新命令 `--help` 补场景化示例 | 全命令 |

#### M5 — expert 整合

> 实施注记（2026-10）：当前采用 **examples 先行** 的保守切片，只更新
> `agenticCLI-examples/dashscope-sdk-expert`，再由
> `scripts/sync_examples.sh` 单向同步到 `dashscope/acli/examples/`；不修改
> `agenticCLI`，也无需把本轮文档改动回流该仓库。2.0 实现已提升到包根，
> 后续涉及协议层的任务使用 `dashscope/protocols/`，不再引用 `dashscope/v2/`。
> 本轮仅交付 5.4 的 examples 问答策略与知识更新；5.1、5.2 未实施，5.3 的完整
> 发布和全斜杠命令回归也不因此视为完成。

| # | 任务 | 交付 |
| --- | --- | --- |
| 5.1 | acli 模型调用底座切 SDK Client（`load_config`/`/provider`/`/model` 语义不变）；providers 协议翻译切换复用包根 `protocols/`。后续实施时先修改并验证 agenticCLI，再用同步脚本更新 dashscope，单独提交 | `acli/agent.py` 等 |
| 5.2 | dashscope 能力注册为内置 tools（images/tts/asr/embeddings/files 等） | `acli/tools/dashscope_*` |
| 5.3 | `dashscope expert` 命令 + 文档；回归 acli 全量斜杠命令 | — |
| 5.4 | expert 问答 prompt 策略：识别 1.x/2.0 语境（`Generation.call`→1.x、`dashscope.Client`→2.0）；新代码推荐 2.0，存量 1.x 修 bug 保持原风格，新参数/新能力标注仅 2.0。当前以 system prompt 的版本选择规则和按需加载的领域 skills 实现，覆盖 2.0.3 包根接口、扁平响应及 `dashscope.errors`；1.x↔2.0 对照留在可检索文档而非常驻 prompt | `agenticCLI-examples/dashscope-sdk-expert/.acli/system-prompt.md` + `.acli/skills/` |

#### M6 — codegen

| # | 任务 | 交付 |
| --- | --- | --- |
| 6.1 | `specs/*.yaml` 元数据格式定义（含任务型/流式标记、协议映射注解） | `specs/schema` |
| 6.2 | 存量接口逆向成 spec（先 chat/files/models 三个试点） | `specs/*.yaml` |
| 6.3 | 生成器：SDK 资源方法 + Pydantic 模型 | `scripts/codegen.py` |
| 6.4 | 生成器：CLI 命令骨架 + help | 同上 |
| 6.5 | 生成器：文档接口表 + 冒烟测试；`# manual` 豁免机制 | 同上 |
| 6.6 | CI 校验"生成物与 spec 一致" | `.github/`/CI |

#### M7 — 2.0 GA 与共存运营

| # | 任务 | 交付 |
| --- | --- | --- |
| 7.1 | 迁移指南（1.x → 2.0 对照表）+ 1.x/2.0 双轨文档体系 | docs ✅ [`dashscope-2.0-migration-guide.md`](dashscope-2.0-migration-guide.md)（对照表另见 `dashscope-2.0-interfaces-mapping.md`） |
| 7.2 | 现有 1.x 测试套件原样接入 CI 作冻结回归（不改不扩）；2.0 全量测试新建于 `tests/v2/` | `tests/v2/` |
| 7.3 | 遥测埋点：旧接口调用占比（脱敏）+ 季度评估机制 | 核心 + dashboard |
| 7.4 | README/Quickstart 全面改写（以 `dashscope.chat` 为首例） | docs |
| 7.5 | 版本号切 2.0.0 发布 + changelog（明示共存策略与兼容承诺） | release |

#### 后续（遥测门槛触发，不设死线）

| # | 任务 | 触发条件 |
| --- | --- | --- |
| 8.1 | 3.0 包结构重整：整目录删除 1.x 旧代码（旧模块 + `cli/` 旧命令树）、`v2/` 内容上移为包主体、acli 迁移适配（import 从 `dashscope.v2.*` 切至包根；与 agenticCLI 仓库同步，单独 commit） | 旧接口调用占比 <5% 且无大客户依赖 |
| 8.2 | `dashscope serve`（入站兼容端点） | M2 后评估，视用户需求立项 |
| 8.3 | 多语言 codegen（Java 优先，其次 Node.js/Go）：基于同一 `specs/*.yaml` 生成；过渡期 Java 仓库先接"CI 对照 spec 一致性校验"止血漂移 | M6 稳定后立项 |

### 风险与决策点

- **双依赖并存**：零接触方案的代价是 requests/aiohttp（1.x）与 httpx（2.0）同时安装，包体积与依赖面增大；不做 extras 拆分（避免用户选型困惑），3.0 删旧代码时自然消除。
- **重复实现的双修风险**：同一服务端 bug 若同时命中新旧两套代码需修两处；因 1.x 冻结、仅修安全/严重 bug，概率低且可控。
- **"不优雅"的静态成本**：包内短期并存两套实现是刻意为之的权衡——换来的是 1.x 回归风险结构性归零与零兼容层成本；3.0 一次性删除收尾。
- **多模态消息格式**：native 与 OpenAI content parts 差异最大的地方，M1 前需先定内部规范消息模型。
- **ws 实时接口与协议层的关系**：Recognition/OmniRealtime 无 OpenAI/Anthropic 对应物，协议层仅覆盖 HTTP 文本类接口，需在文档明示。
- **`dashscope serve`（入站兼容）**是否做、何时做，待 M2 后评估。
- **expert 的包归属**：保持 `dashscope.acli` 模块名不变（存量用户脚本兼容），仅在命令层加 `expert` 入口。
- **多语言 SDK 迭代不同步**（现状：Java 独立 repo、手写维护，接口迭代漂移）：2.0 不新建语言生成器，靠 spec 单一事实源 + CI 一致性校验先收敛；多语言生成列入后续（8.3），避免本次迭代范围失控。
