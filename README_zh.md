# DashScope Python SDK

> [English](README.md) | **中文**

DashScope Python SDK 是[阿里云百炼（Model Studio）](https://help.aliyun.com/zh/model-studio/)的官方客户端，覆盖文本生成、多模态理解、向量（Embedding）、重排（Rerank）、图像/视频生成、语音合成与识别、模型调优、Agentic RL、应用调用等能力。

**2.0 版本**引入了资源风格的客户端 —— `dashscope.Client()`，其易用性对标 OpenAI SDK：一个客户端持有共享连接池，每项能力都以类型化资源的形式挂载其上（`client.chat`、`client.images`、`client.audio`……），响应是类型化对象而非裸字典，失败默认抛出异常。1.x 的模块级 API（`dashscope.Generation` 等）保持不变且仍被完整支持，因此 2.0 是纯增量的。

## 2.0 新特性

| | 1.x | 2.0 |
|---|---|---|
| 入口 | 模块级静态类：`Generation.call(...)` | 资源风格客户端：`client.chat.completions.create(...)` |
| 一行调用 | —— | `dashscope.chat("qwen-plus", "讲个笑话")` |
| 响应 | 裸字典；需判断 `status_code == HTTPStatus.OK` | 类型化对象；直接读 `resp.output_text`、`resp.usage.total_tokens` |
| 错误 | 手动检查 `resp.code` / `resp.message` | 默认抛出类型化的 `DashScopeError` 异常 *（完善中）* |
| 流式 | 裸帧生成器 | `ChatStream`，提供 `.text_stream`、`.reasoning_stream`、`.get_final_message()` |
| 协议 | DashScope 原生 + 单独的 compatible-mode 入口 | 一个客户端通过 `protocol=` 讲 `native`、`openai` 或 `anthropic` |
| 异步 | 独立的 `Aio*` 类 | `dashscope.AsyncClient()` 完整镜像同步接口 |
| 连接 | 每次调用新建 session | 每个客户端一个 keep-alive 连接池，跨资源共享 |
| 配置 | 模块级全局变量（`dashscope.api_key`、`set_region`） | 分层：参数 → `dashscope.settings` → 环境变量 → `~/.dashscope/config.toml` |

## 安装

```shell
pip install -U dashscope
```

基础安装包含 SDK API 调用、`dashscope` CLI 命令、Agentic RL 微调和本地 tokenizer，要求 Python 3.9+。仅剩一个可选功能组：

| Extra | 提供的能力 | 安装命令 |
|-------|----------|---------|
| `acli` | 交互式 AI 助手（DashScope SDK Expert） | `pip install "dashscope[acli]"` |

从源码安装：

```shell
pip install -e .
```

## 快速开始

先配置 API Key（[如何获取](https://help.aliyun.com/zh/model-studio/get-api-key)）：

```shell
export DASHSCOPE_API_KEY="sk-xxx"
```

**一行调用** —— 最快拿到一次补全：

```python
import dashscope

completion = dashscope.chat("qwen-plus", "讲个笑话")
print(completion.output_text)
```

**客户端** —— 正式项目的推荐写法。一个客户端，跨调用复用：

```python
import dashscope

client = dashscope.Client()  # 读取 DASHSCOPE_API_KEY

resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
)
print(resp.output_text)
print(resp.usage.total_tokens, "tokens |", resp.request_id)
```

**流式** —— `stream=True` 返回 `ChatStream`：

```python
stream = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "写一首关于大海的俳句"}],
    stream=True,
)
for text in stream.text_stream:
    print(text, end="", flush=True)

# 或者把整个流合并成一个 ChatCompletion（含 usage）：
# final = stream.get_final_message()
```

**异步** —— `AsyncClient` 镜像同步接口：

```python
import asyncio
import dashscope

async def main():
    client = dashscope.AsyncClient()
    resp = await client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "你好"}],
    )
    print(resp.output_text)
    await client.aclose()

asyncio.run(main())
```

## 核心概念

### 客户端

`dashscope.Client()`（以及 `dashscope.AsyncClient()`）是唯一入口。它持有一个被所有资源调用共享的 keep-alive 连接池，并以属性形式暴露各项能力：

```python
client = dashscope.Client(
    api_key="sk-xxx",       # 默认取 DASHSCOPE_API_KEY
    workspace="ws-xxx123",  # 默认取 DASHSCOPE_WORKSPACE
    region="cn-beijing",    # 默认取 DASHSCOPE_REGION
    protocol="native",      # native | openai | anthropic
    timeout=300.0,
    max_retries=2,
)
```

用上下文管理器可自动关闭连接池；也可用 `with_options()` 派生一个变体（未指定的选项会被继承；当连接级选项不变时共享同一连接池）：

```python
with dashscope.Client() as client:
    client.chat.completions.create(model="qwen-plus", messages=["hi"])

low_latency = client.with_options(timeout=10.0, max_retries=0)
```

每次调用还接受请求级的 `api_key=`、`workspace=`、`timeout=` 覆盖。

### 协议

一个客户端可以讲三种线路协议，通过 `protocol=`（或 `DASHSCOPE_PROTOCOL` / `settings.protocol`）选择。无论哪种协议，请求/响应对象都保持同一套规范化的类型形态 —— 变化的只是线路格式：

| `protocol` | 请求的目标 | 适用场景 |
|------------|----------|----------|
| `native`（默认） | DashScope 原生 API（`/api/v1`） | 需要 DashScope 的完整能力 |
| `openai` | OpenAI 兼容模式（`/compatible-mode/v1`） | 迁移 OpenAI 代码 —— 无需 `openai` 库 |
| `anthropic` | Anthropic 兼容模式（`/apps/anthropic`） | 迁移 Anthropic Messages 代码 |

```python
client = dashscope.Client(protocol="openai")
resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "你好"}],
)
```

### 类型化响应

每个响应都是类型化对象。对于对话，用 `output_text` 读取第一个 choice，或深入 `choices`、`usage`、`search_info`、`request_id`、`model`：

```python
resp = client.chat.completions.create(model="qwen-plus", messages=["hi"])

resp.output_text                       # choices[0].message.content
resp.choices[0].message.content        # 等价，显式写法
resp.choices[0].message.reasoning_content  # 思考模型的推理内容（若有）
resp.choices[0].message.tool_calls     # 工具调用（若有）
resp.usage.input_tokens                # 还有 .output_tokens / .total_tokens
resp.request_id
```

所有响应共享统一信封：`request_id`、`status_code`、`code`、`message`、`is_error`，以及原始的 `headers`。

### 流式

`stream=True` 返回 `ChatStream`（`AsyncClient` 对应 `AsyncChatStream`），并提供类型化辅助属性：

```python
stream = client.chat.completions.create(model="qwen-plus", messages=["hi"], stream=True)

for text in stream.text_stream:            # 内容增量
    ...
# stream.reasoning_stream                  # 推理增量（思考模型）
# stream.tool_call_stream                  # 工具调用增量
# stream.get_final_message()               # 合并后的 ChatCompletion，含 usage
```

默认每帧是增量的。传入 `incremental_output=False` 可改为接收累积快照（SDK 会替你合并增量）。

> **注意：** SDK **不会**自动重试流式请求。生产环境请在你自己的重试/退避逻辑中包裹流消费 —— 参见 [`examples/04_real_world_scenarios/production_ready_rag_harness.py`](examples/04_real_world_scenarios/production_ready_rag_harness.py)。

### 错误

> **仍在完善中。** 2.0 的类型化错误体系正在持续开发 —— 可能新增异常类，API 错误码到异常类的映射也可能调整。若需要向前兼容，请捕获基类 `errors.DashScopeError`。

失败默认抛出类型化的 `DashScopeError`（非 2xx 的 HTTP 状态码，以及 HTTP 200 但业务报错，都会归入这里）：

```python
from dashscope import errors

try:
    client.chat.completions.create(model="qwen-plus", messages=["hi"])
except errors.RateLimitError as exc:
    print("被限流：", exc.code, exc.status_code, exc.request_id)
except errors.ContentFilterError as exc:
    print("被内容审核拦截：", exc.message)
except errors.DashScopeError as exc:       # 所有 SDK 错误的基类
    print("请求失败：", exc.code, exc.message)
```

异常层级包含 `APIConnectionError` / `APITimeoutError`（网络层），以及在 `APIStatusError` 之下的：`BadRequestError`、`AuthenticationError`、`PermissionDeniedError`、`NotFoundError`、`RequestTooLargeError`、`RateLimitError`、`InternalServerError`、`ServiceUnavailableError`、`GatewayTimeoutError`、`ContentFilterError`。

想内联检查错误而不捕获异常？用 `raise_on_error=False` 构建客户端 —— 失败会以正常响应对象返回，带 `is_error=True`、`code`、`message`、`status_code`：

```python
client = dashscope.Client(raise_on_error=False)
resp = client.chat.completions.create(model="qwen-plus", messages=["hi"])
if resp.is_error:
    print(resp.code, resp.message)
```

## 鉴权

SDK 使用 API Key 鉴权，可在[百炼控制台](https://help.aliyun.com/zh/model-studio/get-api-key)获取。可通过以下任一方式提供（优先级由高到低）：

1. **按客户端** —— `dashscope.Client(api_key="sk-xxx")`
2. **全局设置** —— `dashscope.settings.api_key = "sk-xxx"`（对之后创建的客户端生效）
3. **环境变量** —— `export DASHSCOPE_API_KEY="sk-xxx"`
4. **配置文件** —— `~/.dashscope/config.toml` 中的 `api_key`
5. **1.x key 文件** —— 由 `dashscope.save_api_key(...)` 写入的文件（`~/.dashscope/api_key`，或 `DASHSCOPE_API_KEY_FILE_PATH`），作为最低优先级，让既有 1.x 配置继续可用

每次调用还接受请求级的 `api_key=` 覆盖。

> **各地域 API Key 相互独立。** 每个地域签发各自的 Key；使用其他地域的 Key 会返回 `401`。切换 `region` 时请同步更换 `api_key`。

## 配置

配置按层解析 —— **显式参数 → `dashscope.settings` → 环境变量 → `~/.dashscope/config.toml`**。可以在启动时设置一次、按客户端设置，或完全不设置。

**全局设置门面：**

```python
import dashscope

dashscope.settings.api_key = "sk-xxx"
dashscope.settings.workspace = "ws-xxx123"
dashscope.settings.region = "ap-southeast-1"
dashscope.settings.protocol = "openai"
dashscope.settings.timeout = 60.0
dashscope.settings.max_retries = 3
dashscope.settings.log_level = "INFO"
```

**环境变量：**

| 变量 | 设置的对象 | 默认值 |
|----------|------|---------|
| `DASHSCOPE_API_KEY` | API Key | —— |
| `DASHSCOPE_WORKSPACE` | 业务空间 ID（端点子域名） | —— |
| `DASHSCOPE_REGION` | 地域 | `cn-beijing` |
| `DASHSCOPE_PROTOCOL` | `native` / `openai` / `anthropic` | `native` |
| `DASHSCOPE_BASE_URL` | 完全覆盖端点（逃生舱） | —— |
| `DASHSCOPE_TIMEOUT` | 请求超时（秒） | `300` |
| `DASHSCOPE_MAX_RETRIES` | 自动重试次数 | `2` |
| `DASHSCOPE_LOG` | 日志级别 | `WARNING` |
| `DASHSCOPE_CONFIG` | `config.toml` 路径 | `~/.dashscope/config.toml` |

**配置文件**（`~/.dashscope/config.toml`）：

```toml
api_key = "sk-xxx"
workspace = "ws-xxx123"
region = "ap-southeast-1"
protocol = "native"
timeout = 300.0
max_retries = 2
```

### 区域与端点

默认请求发往华北2（北京）网关。当设置了 `workspace` 时，端点会基于业务空间专属的 MaaS 网关 `https://{workspace}.{region}.maas.aliyuncs.com` 构造，每种协议走各自的路径。用 `region=`（或 `DASHSCOPE_REGION` / `settings.region`）指向其他地域：

```python
client = dashscope.Client(region="ap-southeast-1", workspace="ws-xxx123")
# -> https://ws-xxx123.ap-southeast-1.maas.aliyuncs.com/api/v1
```

| 区域 | 地理位置 |
|--------|----------|
| `cn-beijing` | 华北2（北京）—— 默认 |
| `cn-hongkong` | 中国（香港） |
| `ap-southeast-1` | 新加坡 |
| `ap-northeast-1` | 日本（东京） |
| `eu-central-1` | 德国（法兰克福） |
| `us-east-1` | 美国（弗吉尼亚） |

说明：

- **未配置 workspace？** SDK 会回退到传统的共享网关（北京为 `https://dashscope.aliyuncs.com`）并发出 `UserWarning`。这可以正常工作，但为了性能与稳定性，推荐使用业务空间专属网关。
- 基于 WebSocket 的实时 API（实时语音、多模态对话）在 `cn-beijing` 与 `ap-southeast-1` 提供。
- 批量推理、模型调优、应用开发目前仅在 `cn-beijing` 与 `ap-southeast-1` 提供。
- `base_url=`（或 `DASHSCOPE_BASE_URL`）会覆盖一切 —— 这是自定义/私有网关的逃生舱，且不要求配置 workspace。
- 1.x 的 `dashscope.set_region(region, workspace_id)` 全局函数对 1.x 模块级 API 仍然有效。

## 能力一览

每项能力都是客户端上的一个资源。同步与异步客户端暴露相同的接口（异步方法用 `await` 调用）。

| 能力 | 2.0 资源 | 示例 |
|------------|--------------|---------|
| 文本对话 / 生成 | `client.chat.completions.create(...)` | [`text_generation/chat.py`](examples/01_atomic_capabilities/text_generation/chat.py) |
| 多模态对话（图像、视频） | `client.multimodal.chat.create(...)` | [`multimodal_vision/multimodal_chat.py`](examples/01_atomic_capabilities/multimodal_vision/multimodal_chat.py) |
| 文本向量 | `client.embeddings.create(...)`（批量用 `.jobs`） | [`embeddings/embeddings.py`](examples/01_atomic_capabilities/embeddings/embeddings.py) |
| 多模态向量 | `client.multimodal.embeddings.create(...)` | [`embeddings/`](examples/01_atomic_capabilities/embeddings/) |
| 重排 | `client.rerank.create(...)` | [`rerank/rerank.py`](examples/01_atomic_capabilities/rerank/rerank.py) |
| 决策（System One） | `client.decision.create(...)` | [`decision/decision.py`](examples/01_atomic_capabilities/decision/decision.py) |
| Token 计数 | `client.tokenization.count(...)` | [`text_generation/tokenization.py`](examples/01_atomic_capabilities/text_generation/tokenization.py) |
| 图像生成 | `client.images.generate(...)`（+ `.jobs`） | [`image_video_synthesis/images.py`](examples/01_atomic_capabilities/image_video_synthesis/images.py) |
| 视频生成 | `client.videos.generate(...)`（+ `.jobs`） | [`image_video_synthesis/videos.py`](examples/01_atomic_capabilities/image_video_synthesis/videos.py) |
| 语音：ASR / TTS / 实时 *（预览）* | `client.audio.asr` · `client.audio.tts` · `client.audio.realtime` | [`audio/`](examples/01_atomic_capabilities/audio/) |
| 会议与音频理解（通义听悟 TingWu） | `client.tingwu.call(...)` / `.realtime(...)` | [`audio/tingwu.py`](examples/01_atomic_capabilities/audio/tingwu.py) |
| 文件 | `client.files.upload/list/get/delete(...)` | [`files/files.py`](examples/01_atomic_capabilities/files/files.py) |
| 模型 | `client.models.list/get(...)` | [`models/models.py`](examples/01_atomic_capabilities/models/models.py) |
| 模型调优 | `client.fine_tunes.create/get/list/wait/cancel/delete(...)` | [`fine_tuning/fine_tunes.py`](examples/01_atomic_capabilities/fine_tuning/fine_tunes.py) |
| 模型部署 | `client.deployments.create/scale/list/delete(...)` | [`fine_tuning/`](examples/01_atomic_capabilities/fine_tuning/) |
| Agentic RL | `client.rl`（`.jobs`、`register_functions`、`upload_data`） | [`fine_tuning/reinforcement_learning.py`](examples/01_atomic_capabilities/fine_tuning/reinforcement_learning.py) |
| 百炼应用 | `client.apps.completions.create(...)` | [`applications/apps.py`](examples/01_atomic_capabilities/applications/apps.py) |
| AgentStudio（智能体、会话、记忆） | `client.agentstudio` | [`03_agent_recipes/agentstudio.py`](examples/03_agent_recipes/agentstudio.py) |

几个常见写法：

```python
# 向量
resp = client.embeddings.create(model="text-embedding-v4", input=["你好", "世界"], dimension=1024)
resp.embeddings[0].embedding

# 多模态对话（图像理解）
resp = client.multimodal.chat.create(
    model="qwen3-vl-plus",
    messages=[{"role": "user", "content": [{"image": "https://.../cat.png"}, {"text": "这是什么？"}]}],
)
resp.output_text

# 图像生成（提交 + 等待）；手动控制用 client.images.jobs
job = client.images.generate(model="wanx2.1-t2i-turbo", prompt="一间温馨的花店", size="1024*1024")
if job.succeeded:
    print(job.image_urls)

# 百炼应用
resp = client.apps.completions.create(app_id="YOUR_APP_ID", prompt="土豆炖猪脚怎么做？")
resp.text
```

> **语音为预览版。** 2.0 的 `client.audio` 接口尚未定型，可能变化；访问它会发出 `FutureWarning`。

### 单条 WebSocket 上的实时能力

对于时延敏感的实时语音，可以打开一条共享双工 socket，在其上运行多个会话，而不必每个任务都重新握手：

```python
connection = client.ws_connection()
session = client.audio.asr.recognize(model="fun-asr-realtime", connection=connection)
# ……驱动该会话；连接由调用方持有……
connection.close()
```

## 支持的模型

| 类别 | 示例模型 | 2.0 资源 |
|----------|----------------|--------------|
| 文本生成 | qwen-max、qwen-plus、qwen-flash | `client.chat` |
| 多模态理解 | qwen3-vl-plus、qwen-omni | `client.multimodal.chat` |
| 文本向量 | text-embedding-v4、text-embedding-v3 | `client.embeddings` |
| 多模态向量 | tongyi-embedding-vision-plus | `client.multimodal.embeddings` |
| 重排 | gte-rerank-v2、qwen3-rerank | `client.rerank` |
| 图像生成 | wanx2.1-t2i-turbo、wan-image | `client.images` |
| 视频生成 | wanx2.1-t2v、wan-i2v | `client.videos` |
| 语音合成（TTS） | cosyvoice-v1、qwen-tts | `client.audio.tts` |
| 语音识别（ASR） | fun-asr-realtime、paraformer-v1 | `client.audio.asr` |
| 全模态（实时） | qwen-omni-realtime | `client.audio.realtime` |

最新模型列表请访问[百炼模型广场](https://bailian.console.aliyun.com/)。

## AI 助手：DashScope SDK Expert

SDK 内置了交互式 AI 助手 **DashScope SDK Expert**，基于随包提供的 Agentic CLI（`dashscope/acli`）框架构建。它能直接在终端中解答 SDK/API 问题、生成可运行示例、展示 CLI 用法、诊断错误。

- 直接运行 `dashscope`（不带参数）即可启动助手。首次运行时会提示安装知识包（按领域划分的速查技能：文本、多模态、语音、检索、微调、Agent、CLI），使助手的指导来自 SDK 的公开接口 —— 参数、输出、错误码 —— 而无需阅读源码。
- 直接提问代替翻文档 —— 如 `dashscope expert "如何流式输出对话结果"`（一次性提问后退出）。在助手内输入 `/help` 查看可用命令；`dashscope expert --help` 查看全部形式。
- 通过 `acli` extra 安装：`pip install "dashscope[acli]"`。
- 完整使用指南：[DashScope SDK Expert 文档](https://help.aliyun.com/zh/model-studio/dashscope-sdk-expert)。

## Shell 命令补全

针对你的 Shell 运行一次对应命令，然后重启 Shell（或重新 source 配置文件）：

| Shell | 安装命令 |
|-------|-----------------|
| **bash** | `dashscope --install-completion bash` |
| **zsh** | `dashscope --install-completion zsh` |
| **fish** | `dashscope --install-completion fish` |

预览补全脚本而不安装：

```shell
dashscope --show-completion bash
```

## 日志

通过设置或环境变量配置日志级别：

```python
dashscope.settings.log_level = "INFO"
```

```shell
export DASHSCOPE_LOG="INFO"
```

## 示例

[`examples/`](examples/) 目录是基于 2.0 客户端的生产最佳实践合集，按进阶路径组织：

```
examples/
├── 01_atomic_capabilities/   每项能力的极简可运行速查
├── 02_advanced_engineering/  本地 mock 调试、语义缓存、熔断 + 多地域容灾
├── 03_agent_recipes/         工具调用循环、RAG、MCP server、AgentStudio
├── 04_real_world_scenarios/  端到端生产级 harness（RAG、实时会议、智能客服）
└── legacy_1x/                1.x 模块级 API 示例，仅供参考
```

详见 [`examples/README_zh.md`](examples/README_zh.md)。其中若干旗舰案例可完全离线运行（mock 传输），无需 API Key。

## 从 1.x 迁移

1.x 的模块级 API 保持不变且仍被支持 —— 你可以渐进式地采用 2.0。准备好之后，映射关系是一一对应的：

| 1.x | 2.0 |
|-----|-----|
| `Generation.call(model=..., messages=..., result_format="message")` | `client.chat.completions.create(model=..., messages=...)` |
| 先判断 `resp.status_code == HTTPStatus.OK`，再取 `resp.output.choices[0].message.content` | `resp.output_text`（错误默认抛异常） |
| `MultiModalConversation.call(...)` | `client.multimodal.chat.create(...)` |
| `TextEmbedding.call(...)` | `client.embeddings.create(...)` |
| `TextReRank.call(...)` | `client.rerank.create(...)` |
| `ImageSynthesis.call(...)` / `.async_call(...)` | `client.images.generate(...)` / `client.images.jobs` |
| `dashscope.api_key = "sk-xxx"` | `dashscope.Client(api_key=...)` 或 `dashscope.settings.api_key = ...` |
| `dashscope.set_region(region, workspace_id)` | `dashscope.Client(region=..., workspace=...)` |
| OpenAI 兼容通过 `dashscope.aigc.chat_completion.Completions` | `dashscope.Client(protocol="openai")` |

### 包布局与兼容策略

2.0 实现直接位于 `dashscope` 根包，原 `dashscope.v2` 命名空间已移除。
包根已有的 1.x 导出继续可用；`dashscope.aigc`、`dashscope.client` 等旧
deep import 已移至 `dashscope.legacy_1x` 下。2.0 Agentic RL 工作流位于
`dashscope.finetune`，1.x 任务管理接口位于 `dashscope.legacy_1x.finetune`。

> **破坏性变更（Agentic RL）：** 公开路径 `dashscope.finetune.agentic_rl`
> 现在解析到 **2.0** 的 `AgenticRL` —— `submit_job` 返回 `SubmitResult`
> （不再是 1.x 的 `FineTune`），且该类不再带任务管理动词。2.0 的任务生命
> 周期在客户端上（`client.rl.jobs.create()`、`.wait()`、`.list()`、
> `.cancel()`、`.delete()`、`.logs()`）。依赖 1.x 类接口
> （`AgenticRL.cancel(...)`、`.get(...)`、`submit_job` → `FineTune`）的代码
> 应改为导入 `dashscope.legacy_1x.finetune.agentic_rl`，它保留五个动词
> （`cancel`/`list`/`get`/`delete`/`logs`）与 `FineTune` 返回值。

AgentStudio 与 Agentic RL 的异常已统一到 2.0 错误体系：`AgentStudioError`
现在是 `dashscope.errors.DashScopeError` 的别名，Agentic RL 的 `*WithCode`
错误继承自 `AgenticRLError`。原 AgentStudio 的 `raw` 响应属性改以 `body`
暴露（`err.raw` → `err.body`）；`RuntimeErrorWithCode` / `ValueErrorWithCode`
仍可作为其包装的内置 `RuntimeError` / `ValueError` 被捕获。由于两套栈现在
共享 `dashscope._shared._errors.DashScopeException` 基类，一个
`except DashScopeException` 即可同时捕获 1.x、2.0 与 AgentStudio 错误。
Agentic RL 的 `*WithCode` 错误 `str()` 渲染也变了 —— 1.x 的
`[1800] I/O error: …` 前缀被替换为 2.0 的 `<IOErrorWithCode> … | code=sdk.…`
形式（数值码仍可通过 `err.error_code` 获取、时间戳通过 `err.timestamp`），
依赖旧格式的日志解析需要更新。

> **破坏性变更（AgentStudio 错误）：** `ConflictError` 现在是
> `APIStatusError` 的别名，而不再是仅对应 409 的子类，因此
> `except ConflictError` 会捕获**所有**状态错误（400/401/404/429/5xx），
> 而不仅是冲突。若需要旧行为，请改用 `err.status_code == 409` 判断。
> Agentic RL 的 `error_code` 现在是只读 property —— 赋值
> `err.error_code = …` 会抛 `AttributeError`。

> **破坏性变更（`submit_job` / realtime WebSocket）：** `submit_job` 会把
> 额外关键字参数直接转发给传输层，因此 1.x 的 `request_timeout=` 改名为
> `timeout=`，且不再接受 `workspace=` / `headers=` / `stream=`（传入任一
> 都会抛 `TypeError`）。realtime WebSocket 状态错误（`WsStateError`、
> `WsBinaryOwnerConflictError`）现在继承自 `DashScopeError` 而非
> `RuntimeError`，因此裸 `except RuntimeError` 不再能捕获它们 —— 请改捕获
> `errors.DashScopeError`（或具体的错误类）。

`dashscope.save_api_key` 现在来自 `dashscope._config`，仍接受可选的 key
文件路径，并会展开 `~`、创建缺失的父目录、按 UTF-8 写入。当前客户端使用
独立的 `dashscope._logging` logger 层级，不会重新配置共享的 `dashscope`
根 logger。

### 开发测试

```bash
python -m pytest tests --ignore=tests/legacy_1x -q  # 当前实现
python -m pytest tests/legacy_1x -q                 # 归档兼容套件
python -m pytest tests -q                           # 全量
```

当前实现的 coverage 使用 `coverage run -m pytest tests
--ignore=tests/legacy_1x`，随后执行 `coverage report -m`。

## 许可证

本项目采用 Apache License (Version 2.0) 许可证。
