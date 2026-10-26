# dashscope 2.0 接口调用示例集

本文按资源分组给出 dashscope 2.0 可直接运行的调用示例，全部基于 `dashscope/v2/`
当前实现逐条核对（同步为主，有异步对应物的接口附异步版本）。1.x → 2.0 的逐接口
对照见 [`md/dashscope-2.0-interfaces-mapping.md`](dashscope-2.0-interfaces-mapping.md)；
设计依据见 [`md/dashscope-2.0-design.md`](dashscope-2.0-design.md)。

## 为什么是 2.0：相对 1.x 的优化、对齐业界与收益

2.0 做两件事：把 OpenAI / Anthropic 已验证的现代 SDK 基线**追平**，再叠加
DashScope 独有的**差异化能力**。

**追平业界基线**（1.x 短板 → 2.0 落点）：

| 维度 | 1.x | 2.0 |
| --- | --- | --- |
| 入口形态 | 全局变量 `dashscope.api_key` + 类方法 `.call`，多租户 / 多环境易错 | `dashscope.Client(...)` 资源树，同步 / 异步孪生（§0） |
| 错误契约 | 返回值自查 `status_code`，漏写 `if` 即静默失败 | 异常驱动：9 个扁平状态类 + 错误目录派生，`request_id` 全链路贯通（§9） |
| 自动重试 | 核心路径无重试 | 默认 `max_retries=2`，`Retry-After` + 指数退避 + 抖动（§9.4） |
| 类型化 | `**kw` 透传、无 IDE 提示、字段拼写错误运行时才炸 | Pydantic 响应 + `@overload` 精确签名（流式 / 非流式静态可区分） |
| 流式体验 | 默认重放全量、需手设 `incremental_output`、深层取值 | 一律增量 chunk，`text_stream` / `get_final_message()` 助手（§0.5） |
| HTTP 栈 | requests + aiohttp 双栈，每次调用新建 Session | httpx 统一同步 / 异步 + 每 Client 连接池 keep-alive（§8.5） |
| 日志 | 缺 request_id、无敏感信息过滤 | request_id / 状态码 / 重试原因入日志，Authorization 过滤，`DASHSCOPE_LOG` 一键开启（§8.2） |

**差异化优势**（两家官方 SDK 没有的）：

- **一个 SDK 管全部**：一个 Client 覆盖「模型模态（文本 / 多模态 / 图像 / 视频 /
  语音 / 向量 / 排序 / 分词）× 协议（native / openai / anthropic，§0.1）×
  provider（百炼上第三方模型靠协议层零成本接入）」。
- **全链路长任务**：`files → fine_tunes → deployments → rl → agentstudio` 的
  生产链路全覆盖；耗时环节由统一的 `jobs.create/fetch/wait` 长任务模式支撑
  （统一 `Job` 对象 + 类型化视图 + 内置退避轮询，§2.2），OpenAI / Anthropic
  SDK 无此概念。
- **协议翻译收敛在 adapter**：一套调用代码、一套异常族、一套流式 API，
  换 provider 只改 `protocol=` / `base_url=`，不再维护多套 SDK（§0.2 / §0.3）。

**收益**：上手从 6 行降到 2 行（§0）；「请求失败被静默吞掉」的已知路径系统性
堵死；连接池复用消除每请求 TCP/TLS 握手；同一语义全 SDK 只有一个名字
（动词收敛 8 类、字段命名统一，见设计文档 §2.6），学习与排障成本同降。

约定：

- 节标题以 2.0 接口为主体，括号内的"1.x：…"是与之对应的旧入口，仅供迁移对照。
- 示例中的 `sk-xxx` 替换为真实 API Key；不显式传 `api_key` 时读取环境变量
  `DASHSCOPE_API_KEY`。
- 示例中用 `# 可选` 标注可省略的参数（多数带默认值，写成 `# 可选，默认 …`）；
  未标注的为必填。
- `timeout=`（秒）几乎处处可用，示例中不再重复演示。**例外**：`client.rl` 的
  1.x 直通方法（`jobs.create` / `register_functions` / `test_functions` /
  `upload_data` / `query_function_instance_logs` /
  `query_all_function_instance_logs` / `delete_function_instance`）复用 1.x
  业务逻辑，不接受 `timeout=`；`client.agentstudio` 是 1.x 子 SDK 的懒挂载桥，
  参数约定以 1.x 为准。
- `timeout=` 的语义按方法类型区分：普通请求方法上是**单次 HTTP 请求**的超时；
  阻塞式领域动词（`images.generate` / `videos.generate` /
  `audio.asr.transcribe`）上是**等待任务终态**的总预算（默认 24h），提交这一次
  请求的超时另用 `submit_timeout=`；`*.jobs.wait` / `fine_tunes.wait` /
  `rl.jobs.wait` 上同样是等待预算。
- 除 `client.rl` 直通方法外，每个公开方法还接受调用级 `api_key=` /
  `workspace=` 覆盖（跟随 Client 配置优先级），示例中不再重复演示。
- 未捕获的失败默认抛出 `dashscope.v2.errors` 异常族
  （`AuthenticationError` / `RateLimitError` / `NotFoundError` / ...，
  构造 Client 时传 `raise_on_error=False` 可关闭）。完整异常族清单、
  每个类对应的状态码与 `except` 写法见 **§9**。

## 0. 客户端构造与一行调用

```python
import dashscope

# 标准构造（推荐）：显式传 key + workspace，或走环境变量
# DASHSCOPE_API_KEY / DASHSCOPE_WORKSPACE
client = dashscope.Client(api_key="sk-xxx", workspace="ws-xxx")

# 异步客户端：资源树与同步完全一致
aclient = dashscope.AsyncClient(api_key="sk-xxx", workspace="ws-xxx")

# 一行式快捷入口（懒加载全局单例 Client，等价于
# Client().chat.completions.create(model=..., messages=[prompt])）
resp = dashscope.chat("qwen3.7-plus", "给我讲个笑话")
print(resp.choices[0].message.content)

# 用完显式释放连接
client.close()
# async 侧： await aclient.aclose()
```

**`workspace` 建议配置、非必需**：默认端点走 workspace 专属网关
`https://{workspace}.{region}.maas.aliyuncs.com`（官方推荐，性能与稳定性更优）。
参数、`DASHSCOPE_WORKSPACE`、`config.toml` 三处都没给 `workspace`、又没有显式
`base_url` 时，回退到 1.x 共享网关（`cn-beijing` 为
`https://dashscope.aliyuncs.com`）并发出一次 `UserWarning`——api_key-only 的
quickstart 仍能跑通，但建议尽早配置 workspace 切到专属网关。websocket 默认端点
同理（`wss://{workspace}.{region}.maas.aliyuncs.com`），可用调用级 `ws_url=` 或
`DASHSCOPE_WEBSOCKET_BASE_URL` 覆盖（`audio.realtime` 例外：走独立路径，只认
`ws_url=`，见 §4.6）。显式给 `base_url` 是自建 / 私有网关的逃生口，完全接管端点。

**`region`（默认 `cn-beijing`）**选择六个地域网关之一：`cn-beijing` /
`ap-southeast-1` / `cn-hongkong` / `us-east-1` / `eu-central-1` /
`ap-northeast-1`（非法值直接 `ValueError`）。国际用户传 `region=`（或
`DASHSCOPE_REGION` / `settings.region` / config.toml 的 `region`）即可切换
默认端点，不必手拼 `base_url`。

自定义端点 / 协议（对应 1.x `base_http_api_url` 等模块变量）：

```python
# 全部可选（不传则回落到 settings / 环境变量 / config.toml / 默认值）
client = dashscope.Client(
    api_key="sk-xxx",          # 缺省读 DASHSCOPE_API_KEY
    workspace="ws-xxx",        # 缺省回退共享网关 + UserWarning
    region="cn-beijing",       # 默认 cn-beijing，六地域之一
    base_url="https://ws-xxx.cn-beijing.maas.aliyuncs.com/api/v1",  # 逃生口
    protocol="native",         # 默认 "native"；或 "openai" / "anthropic"
    timeout=60.0,
    max_retries=2,
    raise_on_error=True,       # 默认 True
    config_path="~/.dashscope/config.toml",
    protocol_options={"strict": False},
    http_client=None,          # 注入自带 httpx.Client（自定义连接池 / 代理）
)
```

`protocol` 决定的默认端点只作用于 `protocol_base_url`，即协议适配的 chat 家族
（`chat.completions` / `multimodal.chat`）实际发往的地址；其余资源一律走
`base_url`。

### 0.1 协议选择：`native` / `openai` / `anthropic`

`protocol` 只改变 chat 类调用（`chat.completions` / `multimodal.chat`）的线缆格式，
**调用方式与返回类型（`ChatCompletion`）不变**，其余资源（embeddings/images/audio
等）不受影响。三个取值：

- `"native"`（默认）：DashScope 自有协议（`/services/...`），全能力。
- `"openai"`：OpenAI 兼容（`/v1/chat/completions`），可对接百炼 compatible-mode
  或任意 OpenAI 兼容端点 / 第三方网关。
- `"anthropic"`：Anthropic 兼容（`/v1/messages`，system 提取、tool_use 映射）。

未显式配置 `base_url` 时，默认端点按资源家族区分：`chat.completions` /
`multimodal.chat` 跟随 `protocol`（`openai` → `.../compatible-mode/v1`，
`anthropic` → `.../apps/anthropic`，见 `config.protocol_base_url`），其余资源
始终走 native 网关 `.../api/v1`；一旦显式配置 `base_url`，则对全部资源生效。

```python
# 用 OpenAI 兼容协议对接百炼 compatible-mode；调用代码与 native 完全一致
# （base_url 缺省即可，chat 家族自动落到 compatible-mode 端点）
openai_client = dashscope.Client(api_key="sk-xxx", protocol="openai")
resp = openai_client.chat.completions.create(model="qwen3.7-plus", messages="你好")

# protocol_options：strict=True 时未知参数抛错；默认 False 透传
# （厂商新参数不必等 SDK 发版即可使用）
strict_client = dashscope.Client(
    api_key="sk-xxx", protocol="openai", protocol_options={"strict": True},
)
```

字段级的协议翻译规则见设计文档 [`dashscope-2.0-design.md` §3](dashscope-2.0-design.md)。

### 0.2 同一次调用在三种协议下的线缆报文

SDK 调用代码三个协议完全一致：

```python
resp = client.chat.completions.create(
    model="qwen3.7-plus",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好"},
    ],
    temperature=0.7,           # 可选
    max_tokens=512,            # 可选
)
```

差异只在实际收发的 HTTP 报文：

**`native`（默认）** —— `POST {base_url}/services/aigc/text-generation/generation`

```json
{
  "model": "qwen3.7-plus",
  "input": {
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "你好"}
    ]
  },
  "parameters": {"result_format": "message", "temperature": 0.7, "max_tokens": 512}
}
```

```json
{
  "request_id": "...",
  "output": {
    "choices": [
      {"finish_reason": "stop",
       "message": {"role": "assistant", "content": "你好！有什么可以帮你？"}}
    ]
  },
  "usage": {"input_tokens": 21, "output_tokens": 9, "total_tokens": 30}
}
```

**`openai`** —— `POST {base_url}/chat/completions`
（base_url 为 `.../compatible-mode/v1` 时即百炼兼容端点）

```json
{
  "model": "qwen3.7-plus",
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "你好"}
  ],
  "temperature": 0.7,
  "max_tokens": 512
}
```

```json
{
  "id": "chatcmpl-...",
  "model": "qwen3.7-plus",
  "choices": [
    {"index": 0, "finish_reason": "stop",
     "message": {"role": "assistant", "content": "你好！有什么可以帮你？"}}
  ],
  "usage": {"prompt_tokens": 21, "completion_tokens": 9, "total_tokens": 30}
}
```

**`anthropic`** —— `POST {base_url}/v1/messages`（请求头带 `anthropic-version: 2023-06-01`）
（base_url 为 `.../apps/anthropic` 时即百炼 Anthropic 兼容端点，不含 `/v1`）

```json
{
  "model": "qwen3.7-plus",
  "system": "你是一个有帮助的助手。",
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "max_tokens": 512,
  "temperature": 0.7
}
```

注意 system 被提取为独立字段；`max_tokens` 在该协议下必填，缺省时 SDK 自动补
`4096`；`stop` 参数会改名为 `stop_sequences`。

```json
{
  "id": "msg-...",
  "role": "assistant",
  "content": [{"type": "text", "text": "你好！有什么可以帮你？"}],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 21, "output_tokens": 9}
}
```

三种响应都被 adapter 翻译回同一个 `ChatCompletion`，所以
`resp.choices[0].message.content`、`resp.usage.total_tokens` 等取法完全一致。

### 0.3 参数在三种协议下的支持矩阵

同一段调用代码，参数在三个协议下的落点并不相同。图例：
✅ 原生支持（映射到该协议的正确字段）｜ 🔀 透传（进 body/parameters，不报错，
除非 `strict=True`）｜ ❌ 400 拒绝（抛 `BadRequestError`，`status_code=400`、
`code=InvalidParameter`）。

| 规范参数 | native | openai | anthropic |
| --- | --- | --- | --- |
| `model` / `messages` / `stream` | ✅ | ✅ | ✅（system 拆为独立字段） |
| `temperature` / `top_p` | ✅ | ✅ | ✅ |
| `max_tokens` | ✅ | ✅ | ✅（缺省自动补 `4096`） |
| `stop` | ✅ | ✅ | ✅（改名 `stop_sequences`） |
| `tools` | ✅ | ✅ | ✅（转 `input_schema`） |
| `seed` | ✅ | ✅ | ❌ |
| `presence_penalty` | ✅ | ✅ | ❌ |
| `response_format` | ✅ | ✅ | ❌ |
| `repetition_penalty` | ✅ | 🔀 | ❌ |
| `enable_thinking` | ✅ | 🔀 | ❌ |
| `parameters` | 🔀 | 🔀 | 🔀 |

> 表中是**调用侧参数名**。`parameters=` 转发到协议层时叫 `extra_parameters`
> （`ChatProtocol.build_chat_request` 的形参名），读协议代码时对得上即可。

三个协议的差异要点：

- **native**：所有规范参数都落进线缆 `parameters` 段（恒定带 `result_format="message"`），
  不存在“未知参数”，故 `strict` 在 native 上是惰性的（无效果）。
- **openai**：OpenAI 原生参数（`temperature/top_p/max_tokens/stop/seed/`
  `presence_penalty/response_format/tools`）直接进 body 顶层；OpenAI 没有的
  `repetition_penalty` / `enable_thinking` 与 `parameters` 归入“未知”，
  默认透传，`strict=True` 时抛 `DashScopeError`。
- **anthropic**：只认 Messages API 有的字段；`seed` / `repetition_penalty` /
  `presence_penalty` / `enable_thinking` / `response_format` / `n` / `logprobs` /
  `top_logprobs` / `thinking_budget` / `enable_search` / `search_options` /
  `parallel_tool_calls` **无条件 400 拒绝**（与 `strict` 无关）——因为静默丢弃对
  用户不可见，请求会”假成功”但参数并未生效。这些名字即使通过 `parameters` 夹带
  进来也一样 400 拒绝，堵死绕过路径；其余 `parameters` 则 `strict=False` 透传、
  `strict=True` 抛 `DashScopeError`。

> 一句话记忆：native 啥都收且全生效；openai 对没有的参数默认透传（可切 `strict`
> 改成报错）；anthropic 对无对应项的参数直接 400，逼你显式删除或改用其它协议。

### 0.4 响应信封已扁平化

服务端普遍把业务字段包在 `output` / `data` 里，2.0 在解析时统一摊平，**响应对象
上直接读业务字段**，不需要 `.output.` / `.data.` 这一跳：

```python
ft = client.fine_tunes.get("ft-xxx")
print(ft.status, ft.finetuned_model)     # 不是 ft.output.status

info = client.files.get("file-xxx")
print(info.file_id, info.name)           # 不是 info.data["file_id"]
```

每个响应对象都继承 `ResponseBase`，共用一组公共字段：`request_id` / `code` /
`message` / `is_error` / `status_code` / `headers`（网关原始 HTTP 响应头，
小写键 dict；ws 资源无 HTTP 信封，保持空 dict）。**`is_error` 由非空 `code`
自动推导**（与 SDK 判断是否抛异常用的是同一条规则），所以在
`raise_on_error=False` 下任何资源都不会漏标业务错误：

```python
job = client.fine_tunes.get("ft-xxx")   # 服务端 HTTP 200 + code="Job.Failed"
if job.is_error:
    print(job.code, job.message, job.request_id)
```

三个类型**不带**信封，因为它们的载荷里根本没有 HTTP 信封：`FineTuneEvent`
（SSE 事件帧，自带 `is_error` 标记错误帧）、`RegisteredFunctions` /
`UploadedData`（rl 侧透传 1.x 返回值拼出来的记录）。此外列表里嵌套的条目
（`FileList.files[i]` / `ModelList.models[i]` 等）信封字段为空——服务端一次响应
只报一次 `request_id`，读信封请用列表对象本身。

`output` 仅在**服务端字段因服务而异**、无法类型化的地方保留，目前只有
`Job.output`（异步任务原始载荷）与 `FineTuneEvent.output`（事件流原始帧）；
这两处都配了类型化的便捷属性（见 §2.2）。

### 0.5 四种流式形态

| 场景 | 写法 | 产出 |
| --- | --- | --- |
| chat 增量文本 | `client.chat.completions.create(..., stream=True)` | `ChatStream`，`for chunk in stream` → `ChatCompletionChunk`（`choices[].delta`） |
| 百炼应用增量输出 | `client.apps.completions.create(..., stream=True)` | `AppStream`，`for partial in stream` → `AppCompletion`（每帧是完整快照或增量，取决于 `incremental_output`） |
| 微调训练事件 | `for event in client.fine_tunes.stream_events(job_id)` | `FineTuneEvent`（`output` 为原始帧，`is_error` 标记错误帧） |
| 语音合成音频 | `for chunk in client.audio.tts.stream(text)` | `bytes` |
| 语音合成音频（HTTP） | `for chunk in client.audio.tts.http.stream(text, ...)` | `HttpSynthesisChunk`（`audio` 为解码后字节，末片带整段音频 URL） |

异步侧对应 `async for`；websocket 双工（`audio.asr.recognize` /
`audio.realtime.connect`）不是"流式响应"，而是长连接会话，用 `.events()` 迭代
服务端推送的事件 dict（见 §4.1 / §4.4）。

流中出错不会静默：默认（`raise_on_error=True`）抛出异常族；置 `False` 时在流里
产出一个 `is_error=True` 的对象后结束迭代。

## 1. 文本 / 对话

### 1.1 `client.chat.completions.create`（1.x：`Generation.call`）

```python
resp = client.chat.completions.create(
    model="qwen3.7-plus",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "用一句话介绍杭州"},
    ],
    temperature=0.7,           # 可选
    max_tokens=512,            # 可选
)
print(resp.choices[0].message.content)
print(resp.usage.total_tokens, resp.request_id)
```

返回 `ChatCompletion`（三个协议统一解析成它，线缆差异见 §0.2）：

```json
{
  "request_id": "...",
  "model": "qwen3.7-plus",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "……",
        "reasoning_content": null,
        "tool_calls": null
      }
    }
  ],
  "usage": {"input_tokens": 21, "output_tokens": 9, "total_tokens": 30}
}
```

便捷属性 `resp.output_text` 等价于 `choices[0].message.content`。流式时逐 chunk
返回 `ChatCompletionChunk`（`choices[].delta` 取代 `message`，末尾 chunk 带
`usage`）。

messages 也接受纯字符串（自动包装成 user 消息）：

```python
resp = client.chat.completions.create(model="qwen3.7-plus", messages="你好")
```

具名参数全集：`model` / `messages` / `stream` / `temperature` / `top_p` /
`top_k` / `max_tokens` / `stop` / `seed` / `tools` / `tool_choice` /
`response_format` / `n` / `logprobs` / `top_logprobs` / `presence_penalty` /
`repetition_penalty` / `enable_thinking` / `thinking_budget` / `enable_search` /
`search_options` / `parallel_tool_calls` / `incremental_output`。签名之外的模型
参数统一走 `parameters=`（原样并入线缆 `parameters`，兼容协议下并入请求体顶层）：

```python
resp = client.chat.completions.create(
    model="qwen3-vl-plus",
    messages=[{"role": "user", "content": [
        {"image": "https://example.com/cat.jpg"}, {"text": "这张图里有什么？"}]}],
    parameters={"vl_high_resolution_images": True},   # 可选：签名外的模型参数
)
```

流式（对应 1.x `Generation.call(..., stream=True)`）：

```python
stream = client.chat.completions.create(
    model="qwen3.7-plus",
    messages="写一首短诗",
    stream=True,               # 可选，默认 False
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
    # 思考模型还会带 delta.reasoning_content
```

`ChatStream` 另提供便捷助手：`stream.text_stream` 只迭代增量文本，
`stream.get_final_message()` 把整流聚合为一条完整 message（含 tool_calls），
`stream.headers` 取响应头。

工具调用与结构化输出：

```python
resp = client.chat.completions.create(
    model="qwen3.7-plus",
    messages="北京今天天气怎么样？",
    tools=[{                     # 可选
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
)
tool_calls = resp.choices[0].message.tool_calls

resp2 = client.chat.completions.create(
    model="qwen3.7-plus",
    messages="给我一个用户 JSON",
    response_format={"type": "json_object"},   # 可选
)
```

思考模型（对应 1.x `enable_thinking` 增量输出参数）：

```python
resp = client.chat.completions.create(
    model="qwen3-235b-a22b-thinking-2507",
    messages="9.11 和 9.9 哪个大？",
    enable_thinking=True,        # 可选
)
print(resp.choices[0].message.reasoning_content)
print(resp.choices[0].message.content)
```

异步（对应 1.x `AioGeneration.call`）：

```python
resp = await aclient.chat.completions.create(
    model="qwen3.7-plus",
    messages="你好",
)

stream = await aclient.chat.completions.create(
    model="qwen3.7-plus", messages="写一首短诗", stream=True,
)
async for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

### 1.2 多轮对话：messages 列表自行维护

2.0 的 chat 无状态，多轮对话由调用方自己维护 messages 列表：

```python
messages = [{"role": "system", "content": "你是一个有帮助的助手。"}]

for user_input in ["我叫小明", "我叫什么？"]:
    messages.append({"role": "user", "content": user_input})
    resp = client.chat.completions.create(model="qwen3.7-plus", messages=messages)
    answer = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})
    print(answer)
```

### 1.3 代码生成 / 文本理解：同一端点换 model

代码生成、文本理解等场景不需要独立接口，同一端点换模型即可：

```python
# 代码生成
resp = client.chat.completions.create(
    model="qwen3-coder-plus",
    messages="用 Python 写一个二分查找",
)

# 理解 / 总结类任务同理，只是 model 不同
resp = client.chat.completions.create(
    model="qwen3.7-plus",
    messages="总结这段话：……",
)
```

### 1.4 `client.apps.completions.create`（1.x：`Application.call`）

```python
# 非流式
resp = client.apps.completions.create(
    app_id="your-app-id",
    prompt="帮我写一份周报大纲",   # prompt 与 messages 二选一（至少传一个）
)
print(resp.text)                 # AppCompletion，信封已扁平化

# 带会话与 RAG 参数
resp = client.apps.completions.create(
    app_id="your-app-id",
    prompt="继续",
    session_id=resp.session_id,                 # 可选
    rag_options={"pipeline_ids": ["pipe-xxx"]},  # 可选
)
```

`prompt` 传单轮提问，`messages` 传调用方自己维护的完整对话；两者都给时
`prompt` 作为最新一条 user 消息追加到 `messages` 之后。要让服务端替你保存对话
状态，用 `session_id=`（会话）或 `memory_id=`（长期记忆）。
`biz_params=` 承载应用自身的流程 / 插件入参（服务端字段名，位于线缆 `input`
内），模型侧旋钮与签名未列出的参数放 `parameters=`。其余具名参数：
`has_thoughts`（返回思考过程）/ `doc_tag_codes` / `doc_reference_type` /
`image_list` / `file_list` / `temperature` / `top_p` / `top_k` / `seed`。

返回 `AppCompletion`：

```json
{
  "request_id": "...",
  "text": "……",
  "finish_reason": "stop",
  "session_id": "session-xxx",
  "thoughts": null,
  "doc_references": null,
  "workflow_message": null,
  "usage": {"models": [{"model_id": "qwen3.7-plus", "input_tokens": 30, "output_tokens": 50}]}
}
```

业务级错误体现在继承来的 `code` / `message` / `is_error` 字段上。

```python
# 流式
for partial in client.apps.completions.create(
    app_id="your-app-id",
    prompt="讲个长一点的故事",
    stream=True,               # 可选，默认 False
    incremental_output=True,   # 可选
):
    print(partial.text, end="")
```

## 2. 多模态

### 2.1 `client.multimodal.chat.create`（1.x：`MultiModalConversation.call`）

content 为 parts 列表（沿用 1.x 的 `{"image": ...}` / `{"text": ...}` / `{"audio": ...}`
part 形式）：

```python
resp = client.multimodal.chat.create(
    model="qwen3-vl-plus",
    messages=[{
        "role": "user",
        "content": [
            {"image": "https://example.com/cat.jpg"},
            {"text": "这张图里有什么？"},
        ],
    }],
)
print(resp.choices[0].message.content)

# 流式（对应 AioMultiModalConversation 的异步流式同理加 await/async for）
stream = client.multimodal.chat.create(
    model="qwen3-vl-plus",
    messages=[{
        "role": "user",
        "content": [
            {"image": "https://example.com/cat.jpg"},
            {"text": "详细描述这张图"},
        ],
    }],
    stream=True,               # 可选，默认 False
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

返回与文本对话相同的 `ChatCompletion`（结构见 §1.1）；多模态端点返回的 list 形态
content 会被扁平化为纯文本字符串。

**多数情况下不需要这个资源**：`chat.completions.create` 会检查 messages，只要
有 image / audio / video part 就自动改走多模态端点，签名与返回类型完全一致。

```python
# 等价于上面的 multimodal.chat.create
resp = client.chat.completions.create(
    model="qwen3-vl-plus",
    messages=[{"role": "user", "content": [
        {"image": "https://example.com/cat.jpg"},
        {"text": "这张图里有什么？"},
    ]}],
)
```

`multimodal.chat.create` 保留为**显式逃生口**：无论有没有媒体 part，每次调用都
打到多模态端点。典型用途是给 qwen-vl 系列模型发一条纯文本消息（按内容路由会
把它留在文本端点上）。兼容协议（`openai` / `anthropic`）没有独立的多模态端点，
part 翻译在它们自己的 chat 路径上完成，所以那里两个资源是同一次调用。

### 2.2 `client.images.generate`：提交 + 等待一步到位（1.x：`ImageSynthesis.call`）

```python
job = client.images.generate(
    model="wanx2.1-t2i-turbo",
    prompt="一只在月球上弹吉他的猫，水彩风格",
    negative_prompt="模糊, 低质量",   # 可选
    size="1024*1024",                  # 可选，写 "1024x1024" 也可以
    n=1,                               # 可选
)
# 返回已完成 Job
for url in job.image_urls:
    print(url)
```

其余具名参数：`seed` / `style` / `prompt_extend`（智能改写）/ `watermark` /
`function` / `image_urls`（wan2.6-image 参考图列表）/ `extra_input=`（并入线缆
`input` 段）；`generate` 还接受 `poll_interval=` / `max_interval=` 调轮询节奏。

**按模型名自动路由协议**：wan2.5 及以下走旧 text2image 端点（仅异步任务）；
wan2.6-t2i / wan2.6-image 只应答 multimodal-generation 新协议（`generate` 同步打
multimodal-generation 端点，`jobs` 异步任务打 image-generation 端点），出参统一
归一化为 `Job.results`。调用代码不变，换模型即自动切换。

对应 1.x `async_call/fetch/wait` 的分步写法：

```python
job = client.images.jobs.create(
    model="wanx2.1-t2i-turbo",
    prompt="一只在月球上弹吉他的猫",
    size="1024*1024",          # 可选
)
print(job.task_id, job.status)          # PENDING / RUNNING

snapshot = client.images.jobs.fetch(job.task_id)
print(snapshot.status)

done = client.images.jobs.wait(job)     # 退避轮询到终态，超时抛 DashScopeError
# wait 的 poll_interval / max_interval / timeout 均可选（默认 1s → 8s 退避，总预算 24h）
done = client.images.jobs.wait(job, poll_interval=2.0, timeout=600.0)

# 任务管理：取消与列举
client.images.jobs.cancel(job.task_id)
page = client.images.jobs.list(status="SUCCEEDED", page_no=1, page_size=10)
# list 可按 start_time / end_time / model_name / api_key_id / region / status 过滤
```

`generate` 与 `jobs.*` 均返回 `Job`（images / videos / transcription /
embeddings.jobs 共用此结构）：

```json
{
  "task_id": "task-xxx",
  "status": "SUCCEEDED",
  "request_id": "...",
  "submitted_at": "...",
  "scheduled_at": "...",
  "finished_at": "...",
  "output": {"results": [{"url": "https://..."}]},
  "usage": {"image_count": 1},
  "code": null,
  "message": null
}
```

`status` 取值 `PENDING` / `RUNNING` / `SUCCEEDED` / `FAILED` / `CANCELED`
（另有终态 `UNKNOWN`：任务状态无法确认，`wait` 遇到它直接返回、不抛异常）。
`FAILED` 时默认（`raise_on_error=True`）抛携带 `code`/`message` 的异常；`wait`
期间任务被 cancel 同样抛错。

`Job` 是**唯一保留 `output` 原始字典**的响应类型（各服务的结果字段不同），但配了
类型化视图，正常取值不必碰 `output`：

| 取法 | 含义 |
| --- | --- |
| `job.results` | `List[JobResult]`，每项有 `.url` / `.transcription_url`，服务端多给的字段仍可按属性读 |
| `job.image_urls` | `List[str]`，所有结果的 `url` |
| `job.video_url` | `Optional[str]`，视频任务的结果 |
| `job.result_url` | `Optional[str]`，单结果任务的唯一产物（图片 / 视频 / 文件 URL 都收在这一个属性下） |
| `job.job_id` | `task_id` 的别名（`FineTune` 把同一个标识拼作 `job_id`） |
| `job.succeeded` / `job.finished` | 状态判断 |

时间戳统一为 `submitted_at` / `scheduled_at` / `finished_at`；服务端原始的
`submit_time` / `scheduled_time` / `end_time` 也能读到同一份值（校验别名），
`status` 同理接受 `task_status`。

异步版本（`AioImageSynthesis`）只需 `aclient.images.jobs.create` / `await ...wait(job)`。

### 2.3 `client.videos.generate`（1.x：`VideoSynthesis.call`）

```python
# 文生视频
job = client.videos.generate(
    model="wan2.7-t2v",
    prompt="一只小猫在草地上奔跑",
    size="1280*720",           # 可选；另有 duration / resolution / ratio /
                               # seed / prompt_extend / watermark 具名参数
)
print(job.video_url)

# 图生视频（prompt 与 image_url 可同传，至少传一个）
job = client.videos.generate(
    model="wanx2.1-i2v-turbo",
    image_url="https://example.com/cat.jpg",
)

# wan2.7-i2v 起走 media 新协议：类型化素材列表，
# type 为 first_frame / last_frame / driving_audio / first_clip，每种至多一个；
# image_url= 是单个 first_frame 素材的简写（与显式 first_frame 同传会报错）
job = client.videos.generate(
    model="wan2.7-i2v",
    prompt="镜头缓缓推近",
    media=[
        {"type": "first_frame", "url": "https://example.com/start.png"},
        {"type": "last_frame", "url": "https://example.com/end.png"},
    ],
)

# 分步：jobs.create / fetch / wait / cancel / list，与 images 相同
job = client.videos.jobs.create(model="wan2.7-t2v", prompt="海浪拍打礁石")
done = client.videos.jobs.wait(job)
```

返回值同为 `Job`（结构见 §2.2），视频结果读 `job.video_url`。模型专属的
`input` 字段经 `extra_input=` 透传，其余服务端参数放 `parameters=`。

### 2.4 `client.multimodal.embeddings.create`（1.x：`MultiModalEmbedding.call`）

```python
resp = client.multimodal.embeddings.create(
    model="multimodal-embedding-v1",
    input=[
        {"text": "一只猫"},
        {"image": "https://example.com/cat.jpg"},
        # 还支持 {"video": url}、{"audio": url}
    ],
    # fps / res_level（分辨率档位 0-3）/ max_video_frames 是顶层参数，
    # 仅对快照模型生效
)
for item in resp.embeddings:
    print(item.index, item.type, len(item.embedding))

# 异步：await aclient.multimodal.embeddings.create(...)
```

返回与文本向量相同的 `EmbeddingResponse`（结构见 §3.1），文本 / 图片 / 音频 /
视频的结果落在同一个向量空间里，靠 `item.type` 区分。

input 元素也可用类型化的 `EmbeddingItem`（1.x 的
`MultiModalEmbeddingItemText/Image/Audio` 收敛而来）：

```python
from dashscope.v2.resources.multimodal.embeddings import EmbeddingItem

resp = client.multimodal.embeddings.create(
    model="multimodal-embedding-v1",
    input=[
        EmbeddingItem(text="一只猫", factor=1.0),  # factor 可选
        EmbeddingItem(image="https://example.com/cat.jpg"),
    ],
)
```

每个 input 元素至少要设置 `text` / `image` / `audio` / `video` 之一（全空报错）；
`EmbeddingItem` 的具名字段就是这四个模态加 `factor`。签名未列出的参数放
`parameters=`。

`oss://` 输入由服务端解析，SDK 自动带上 `X-DashScope-OssResourceResolve: enable`
请求头，**不需要**调用方额外开关（1.x 需要手工传 headers，2.0 已移除该参数）。

## 3. 向量 / 排序 / 分词

### 3.1 `client.embeddings.create`（1.x：`TextEmbedding.call`）

```python
resp = client.embeddings.create(
    model="text-embedding-v3",
    input=["今天天气不错", "明天会下雨吗"],
    dimension=1024,              # 可选
    text_type="document",        # 可选；检索场景 query 侧传 "query"
    # 另有 output_type / instruct 具名参数
)
for emb in resp.embeddings:
    print(emb.index, len(emb.embedding))
print(resp.usage.total_tokens)

# 异步：await aclient.embeddings.create(...)
```

`input` 也接受单个字符串。返回 `EmbeddingResponse`：

```json
{
  "request_id": "...",
  "embeddings": [
    {"index": 0, "type": "text", "embedding": [0.013, ...]},
    {"index": 1, "type": "text", "embedding": [0.021, ...]}
  ],
  "usage": {"total_tokens": 12}
}
```

序号字段统一叫 `index`（1.x 文本向量侧的 `text_index` 是它的校验别名，读到同一
份值）；多模态向量复用同一个 `Embedding` / `EmbeddingResponse`，靠 `type` 区分
模态。

### 3.2 `client.embeddings.jobs.create/fetch/wait`（1.x：`BatchTextEmbedding`）

```python
job = client.embeddings.jobs.create(
    model="text-embedding-async-v2",
    file_url="https://example.com/batch_input.txt",   # 待向量化文本文件 URL
    text_type="document",                             # 可选
)
done = client.embeddings.jobs.wait(job)
print(done.result_url)        # 结果文件 URL

# 任务管理：fetch / cancel / list 与 images.jobs 同形
```

返回 `Job`（结构见 §2.2）；结果文件 URL 读 `done.result_url`，需要看服务端原始
字段时再落到 `done.output`。

### 3.3 `client.rerank.create`（1.x：`TextReRank.call`）

```python
resp = client.rerank.create(
    model="gte-rerank-v2",
    query="什么是量子计算？",
    documents=[
        "量子计算是一种利用量子力学原理进行计算的技术。",
        "今天超市鸡蛋打折。",
        "量子比特可以同时处于 0 和 1 的叠加态。",
    ],
    top_n=2,                     # 可选
    return_documents=True,       # 可选
    instruct="...",              # 可选
)
for r in resp.results:
    print(r.index, r.relevance_score)

# 异步：await aclient.rerank.create(...)
```

返回 `RerankResponse`：

```json
{
  "request_id": "...",
  "results": [
    {"index": 0, "relevance_score": 0.92, "document": {"text": "……"}},
    {"index": 2, "relevance_score": 0.71, "document": {"text": "……"}}
  ],
  "usage": {"total_tokens": 40}
}
```

`document` 仅在 `return_documents=True` 时携带。

### 3.4 `client.tokenization.count`（1.x：`Tokenization.call`）

```python
# model 必填；messages 与 prompt 至少传一个（同给时 prompt 追加为最新 user 消息）
count = client.tokenization.count(
    model="qwen3.7-plus",
    messages=[
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"},
    ],
)
print(count.usage.input_tokens)

# 或按纯 prompt 计
count = client.tokenization.count(model="qwen3.7-plus", prompt="你好，世界")
```

返回 `TokenCount`：`{"request_id": "...", "usage": {"input_tokens": 12}}`，另带
`token_ids` / `tokens` 明细；`count.input_tokens` 是 `usage.input_tokens` 的便捷
属性。`messages` 与 chat 一样接受纯字符串与 parts 列表。

只需要本地分词（不发请求）时用 `dashscope.v2.get_tokenizer(model)`；
`dashscope.v2.list_tokenizers()` 列出可用分词器。

## 4. 语音

完整参数、双向流示例、元数据和生命周期说明见：

- [ASR Python SDK：实时识别与文件转写](dashscope-2.0-asr-sdk.md)
- [TTS Python SDK：一次性、输出流式与双向流合成](dashscope-2.0-tts-sdk.md)
- [Realtime Python SDK：Omni、Qwen-Audio 与 Livetranslate](dashscope-2.0-realtime-sdk.md)
- [实时语音翻译（Gummy）Python SDK](dashscope-2.0-translation-sdk.md)
- [语音管理面：词表 / 热词 / 声音复刻 / 声音设计](dashscope-2.0-audio-management-sdk.md)

### 4.1 `client.audio.asr.recognize(...)` 上下文管理器（1.x：`Recognition.start/send_audio_frame/stop`）

```python
from concurrent.futures import ThreadPoolExecutor


def receive_recognition(rec):
    for event in rec.events():
        print(event)


with client.audio.asr.recognize(
    model="paraformer-realtime-v2",
    format="pcm",
    sample_rate=16000,
) as rec:
    with ThreadPoolExecutor(max_workers=1) as pool:
        receiver = pool.submit(receive_recognition, rec)
        try:
            for chunk in iter_pcm_chunks():  # 由应用提供实时 PCM 帧。
                rec.send_audio_frame(chunk)
            rec.stop()
            receiver.result()
        finally:
            rec.close()
```

识别走 websocket 双工，无 HTTP JSON 响应；`rec.events()` 逐个产出识别事件
dict（结构同 1.x Recognition 回调里的服务端事件），`rec.results()` 只产出
识别结果帧。其余可选参数：`vocabulary_id`（定制词表）/ `phrase_id`（热词）/
`diarization_enabled`（说话人分离）/ `disfluency_removal_enabled`（去语气词）/
`input=`（透传线缆 input 段）/ `parameters=`。
会话方法全集：`start()`（进上下文自动调用）/ `send_audio_frame()` /
`update_context()` / `stop()` / `events()` / `results()` / `close()`；
`rec.request_id` / `first_package_delay` / `last_package_delay` 可取。
`recognize_file(path, ...)` 支持本地文件一次性识别。

### 4.2 `client.audio.asr.transcribe`（1.x：`Transcription.async_call/fetch/wait`）

```python
# 一步到位：提交录音文件转写并等待完成
# （QwenTranscription 也走这里，换 model 即可，如 model="qwen3-asr-flash-filetrans"）
job = client.audio.asr.transcribe(
    ["https://example.com/meeting1.wav", "https://example.com/meeting2.wav"],
    model="paraformer-v2",     # 可选，默认 "paraformer-v2"
    # submit_timeout= / poll_interval= / max_interval= 可选，语义同 images.generate
)
for result in job.results:
    print(result.transcription_url)

# 分步控制
jobs = client.audio.asr.jobs
job = jobs.create(model="paraformer-v2", file_urls=["https://example.com/a.wav"])
print(job.task_id)
snapshot = jobs.fetch(job.task_id)
done = jobs.wait(job)

# 异步：await aclient.audio.asr.transcribe(...)
```

`transcribe` 返回已完成 `Job`（结构见 §2.2），转写结果读
`[r.transcription_url for r in job.results]`。

### 4.3 `client.audio.tts.synthesize/stream`（1.x：`SpeechSynthesizer.call`）

```python
# 一次性合成：拿到完整音频字节后写文件
# 当前走 CosyVoice SpeechSynthesizer WebSocket 协议，不通用适配所有 TTS 协议。
audio = client.audio.tts.synthesize(
    "你好，欢迎收听今天的节目。",   # text 必填（位置参数）
    model="cosyvoice-v1",         # 可选，默认 "cosyvoice-v1"
    voice="longxiaochun",         # 可选，默认 "longxiaochun"
    format="mp3",                 # 可选，默认 "mp3"
    sample_rate=22050,            # 可选，默认 22050
    # 其余可选具名参数：volume（音量）/ rate（语速）/ pitch（音调）/
    # language_hints（语种提示）/ enable_ssml
)
with open("out.mp3", "wb") as f:
    f.write(audio)

# 流式合成（对应 1.x tts_v2 SpeechSynthesizer 的流式回调）：边收边播
with open("out.mp3", "wb") as f:
    for chunk in client.audio.tts.stream(
        "这是一段比较长的文本，会被逐步合成为音频流。",
        model="cosyvoice-v1",       # 可选，默认 "cosyvoice-v1"
        voice="longxiaochun",       # 可选，默认 "longxiaochun"
    ):
        f.write(chunk)                    # 或直接喂给播放器
```

`stream` 返回 `Iterator[bytes]`（逐块产出的音频字节）；`synthesize` 等价于把流
拼成完整 `bytes` 返回。这两个便捷入口只输出音频，不暴露逐条 JSON 元数据。

逐段输入文本时使用 `client.audio.tts.connect(...)` 会话：
`start()` → 多次 `send_text()` / `flush()` → `stop()` → 读完结果 → `close()`。
进入上下文会自动 `start()`；发送文本与接收结果应并发。
会话 `events()` 同时返回音频 `bytes` 和原始 JSON，能够读取时间戳、用量等字段；
`audio_chunks()` 只返回音频，但仍更新会话的 `last_response` 和 `usage`。
完整示例见 [TTS Python SDK](dashscope-2.0-tts-sdk.md)。

### 4.3.1 `client.audio.tts.http.synthesize/stream`（1.x：`HttpSpeechSynthesizer.call` / qwen-tts HTTP）

真 HTTP POST（不经 WebSocket），sambert/cosyvoice 与 qwen-tts 两族共用，按模型名
前缀路由端点（`qwen*` → multimodal-generation，其余 → audio/tts/SpeechSynthesizer）：

```python
# 一次性（对应 1.x HttpSpeechSynthesizer.call(stream=False)）：音频以临时 URL 返回
result = client.audio.tts.http.synthesize(
    "今天天气怎么样？",
    model="sambert-zhichu-v1",
    voice="zhichu",
    format="wav",
    sample_rate=16000,
)
print(result.audio_url, result.audio_id, result.expires_at, result.usage)

# 流式（对应 1.x qwen SpeechSynthesizer.call(stream=True)）：SSE 逐片产出
with open("out.mp3", "wb") as f:
    for chunk in client.audio.tts.http.stream(
        "那我来给大家推荐一款T恤，这款T恤是纯棉的。",
        model="qwen3-tts-flash",
        voice="Cherry",
        language_type="Chinese",
    ):
        if chunk.audio:            # 本片音频帧（已 base64 解码）
            f.write(chunk.audio)
        if chunk.is_final:         # finish_reason == "stop"，带整段音频 URL
            print(chunk.audio_url, chunk.usage)
```

模型专属的 `input` 字段（如 cosyvoice 的 `volume/rate/pitch`）经 `extra_input=` 透传，
顶层 `parameters=` 原样放入请求体；具名参数与 `extra_input` 同名时具名优先。
完整参数与返回类型表见 [TTS Python SDK](dashscope-2.0-tts-sdk.md) 的 HTTP 合成一节。

### 4.4 `client.audio.realtime.connect(...)`（1.x：`audio.qwen_omni.OmniRealtimeConversation`）

```python
# connect() 仅 model 必填；ws_url / connector / timeout / headers 均可选
with client.audio.realtime.connect(model="qwen-omni-turbo-realtime") as rt:
    rt.send_event({
        "type": "session.update",
        "session": {"modalities": ["text", "audio"], "voice": "Chelsie"},
    })
    rt.send_event({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "你好"}],
        },
    })
    rt.send_event({"type": "response.create"})
    for event in rt.events():             # 连接关闭时迭代结束
        if event["type"] == "response.text.delta":
            print(event["delta"], end="")
        elif event["type"] == "response.done":
            break
```

`RealtimeSession` 还提供 `update_session()`、媒体输入、对话条目和响应控制等便捷方法。
`recv_event()` 取一条事件，正常关闭后返回 `None`；活动响应或结束确认缺失时会抛异常。
`rt.model` / `rt.url` 可读模型与 websocket 地址。
`rt.events()` 产出服务端推送的原始事件 dict；`response.done` 只表示本轮响应结束，
不会自动结束会话。Omni/Live 可调用 `end_session()` 后读取至 `session.finished`；
Qwen-Audio 则读完当前响应后 `close()`，不调用 `end_session()`。
三类模型的字段映射和流程差异见 [Realtime Python SDK](dashscope-2.0-realtime-sdk.md)。

### 4.5 `client.audio.translation.recognize(...)`（1.x：`TranslationRecognizerRealtime.start/send_audio_frame/stop`）

Gummy 实时翻译（`gummy-realtime-v1` / `gummy-chat-v1`）与 `asr.recognize` 同协议、同会话生命周期，
翻译参数具名化，结果走双语类型化视图（对应 1.x 回调里的
`(transcription_result, translation_result, usage)` 组合）：

```python
with client.audio.translation.recognize(
    model="gummy-realtime-v1",
    format="pcm",
    sample_rate=16000,
    source_language="zh",
    translation_enabled=True,
    translation_target_languages=["en"],
) as session:
    session.send_audio_frame(chunk)     # 发送线程逐帧喂音频
    session.stop()
    for result in session.results():    # 接收线程迭代双语结果
        transcription = result.get_transcription()
        for translation in result.get_translations():
            print(translation["lang"], translation["text"])
```

本地文件一次性翻译（1.x `TranslationRecognizerRealtime.call(file)`）：
`client.audio.translation.recognize_file(path, model="gummy-realtime-v1", ...)`，
逐项产出 `TranslationResult`。完整示例见 [实时语音翻译（Gummy）Python SDK](dashscope-2.0-translation-sdk.md)。
qwen3-livetranslate 实时翻译模型走 `realtime.connect`（4.4），不在本资源面。

### 4.5.1 `client.tingwu.call` / `client.tingwu.realtime`（1.x：`TingWu.call` / 实时理解）

听悟（会议 / 音视频理解）两个入口（对应 1.x `dashscope.multimodal.tingwu` 一对）：

```python
# 离线纪要：一次 POST 返回解析结果
result = client.tingwu.call(
    model="tingwu-turbo",
    input={"file_url": "https://example.com/meeting.mp4", ...},
    parameters={...},          # 可选
)
print(result.output, result.usage)

# 实时理解：websocket 双工上下文管理器，协议同 asr.recognize
with client.tingwu.realtime(
    model="tingwu-turbo",
    app_id="your-app-id",      # 必填
    format="pcm",              # 可选
    sample_rate=16000,         # 可选
    terminology="...",         # 可选，术语库
    max_end_silence=800,       # 可选
) as session:
    session.send_audio_frame(chunk)
    session.stop()
    for event in session.events():   # 产出服务端事件 dict，speech-end 后结束
        ...
```

`client.tingwu.realtime` 支持 `connection=` 复用（§4.6.1）；异步侧只有
`aclient.tingwu.call`（ws 实时类保持同步面，与 audio 其余资源一致）。

### 4.6 自定义 websocket 连接：`connector=`

`audio.asr.recognize` / `audio.tts.connect` / `audio.tts.synthesize` /
`audio.tts.stream` / `audio.realtime.connect` / `audio.translation.recognize`
都接受 `connector=`，用于注入自建连接
（代理、自定义握手或测试）。SDK 会关闭返回的连接对象，不提供连接池归还协议。
它是一个**可调用对象**：

```python
# Connector = Callable[[str, Dict[str, str]], Any]
# 入参是 (url, headers)，返回一个 websocket 会话对象；
# 缺省时用内置的 websocket-client 实现。
def my_connector(url, headers):
    return my_ws_library.connect(url, header=[f"{k}: {v}" for k, v in headers.items()])

rt = client.audio.realtime.connect(
    model="qwen-omni-turbo-realtime",
    connector=my_connector,     # 可选
)
```

只需要换地址时用 `ws_url=`。ASR/TTS 还支持环境变量
`DASHSCOPE_WEBSOCKET_BASE_URL`；Realtime 使用独立路径，不使用此环境变量。

### 4.6.1 连接复用：`client.ws_connection()` + `connection=`

一条物理连接上跑多个任务（1.x Java 侧常用的复用模式；语音实时对延时
敏感，省一次握手就是首包延迟）：

```python
conn = client.ws_connection()   # 继承 client 的 key/workspace/region/timeout

# 顺序复用：第二个任务免握手
for chunk in (pcm_1, pcm_2):
    with client.audio.asr.recognize(model="paraformer-realtime-v2", connection=conn) as rec:
        rec.send_audio_frame(chunk)
        rec.stop()
        for result in rec.results():
            ...

# 并发：1 路 ASR(二进制上行) + 1 路 TTS(二进制下行) + 任意多 JSON 任务
rec = client.audio.asr.recognize(model="paraformer-realtime-v2", connection=conn).start()
tts = client.audio.tts.connect(model="cosyvoice-v1", connection=conn).start()
conn.close()   # 由调用方关闭；各会话 close() 只摘离任务、不关共享连接
```

支持 `connection=`：`audio.asr.recognize` / `audio.tts.connect` /
`audio.translation.recognize` / `tingwu.realtime`。协议约束：二进制帧不带
task_id，单连接二进制上行/下行各单 owner（并发 TTS 需多条连接）；realtime
族协议无 task_id、会话结束即关连接，不可复用（压首开延迟用提前 `connect()`
预热）。HTTP 侧所有资源默认共享每 Client 的 keep-alive 池，池参数见
`settings.http_max_connections` 等（8.5）。详见
[连接复用指南](dashscope-2.0-connection-reuse-sdk.md)。

### 4.7 管理面：`client.audio.vocabularies` / `phrases` / `voices` / `voice_design`（1.x：`VocabularyService` / `AsrPhraseManager` / `VoiceEnrollmentService` / 声音设计）

1.x 三个管理类的五动词（`create_*` / `list_*` / `query_*` / `update_*` / `delete_*`）
统一为 `create/list/get/update/delete`（`query_*` → `get`）；声音设计在 1.x 无 SDK
类，按官方 REST 建模：

```python
# 定制词表（1.x VocabularyService）：create 返回 vocabulary_id
vocab = client.audio.vocabularies.create(
    target_model="paraformer-realtime-v2",
    prefix="med",
    vocabulary=[{"text": "心肌梗死", "weight": 4}],
)
client.audio.vocabularies.get(vocab.vocabulary_id)
client.audio.vocabularies.list(prefix="med")
client.audio.vocabularies.update(vocab.vocabulary_id, vocabulary=[...])
client.audio.vocabularies.delete(vocab.vocabulary_id)

# paraformer 热词（1.x AsrPhraseManager）：编译任务，phrase_id 供 asr.recognize 使用
phrase = client.audio.phrases.create(model="paraformer-v2", phrases={"下一首": 90})
client.audio.phrases.update(phrase.phrase_id, model="paraformer-v2", phrases={"暂停": 90})
client.audio.phrases.get(phrase.phrase_id)
client.audio.phrases.list(page_no=1)
client.audio.phrases.delete(phrase.phrase_id)

# 声音复刻（1.x VoiceEnrollmentService）：create 返回 voice_id，可作 TTS voice
voice = client.audio.voices.create(
    target_model="cosyvoice-v3-plus",
    prefix="demo",
    url="https://your-audio-url/sample.wav",
)
client.audio.voices.get(voice.voice_id)
client.audio.voices.list(prefix="demo")
client.audio.voices.update(voice.voice_id, url="https://your-audio-url/new.wav")
client.audio.voices.delete(voice.voice_id)

# 声音设计（官方 REST，qwen-voice-design）：文字描述生成音色，返回 voice + 预览音频
designed = client.audio.voice_design.create(
    target_model="qwen3-tts-vd-realtime-2026-01-15",
    voice_prompt="沉稳的中年男性，音色低沉浑厚",
    preview_text="各位听众朋友，大家好",
)
client.audio.voice_design.get(designed.voice)
client.audio.voice_design.list()
client.audio.voice_design.delete(designed.voice)
```

完整参数与返回类型见 [语音管理面 Python SDK](dashscope-2.0-audio-management-sdk.md)。

## 5. 微调 / 部署 / 资源

### 5.1 `client.fine_tunes.*`（1.x：`FineTunes.*`）

```python
# 创建（对应 FineTunes.call）
ft = client.fine_tunes.create(
    model="qwen-turbo",
    training_file_ids=["file-xxx"],
    validation_file_ids=["file-yyy"],                        # 可选
    mode="sft",                                              # 可选
    hyper_parameters={"n_epochs": 2, "learning_rate": "1e-5"},  # 可选
    finetuned_model="my-ft-name",                            # 可选
)
print(ft.job_id, ft.status)

# 等待终态（对应 FineTunes.wait，退避轮询 + 超时）
done = client.fine_tunes.wait(ft, timeout=7200)   # timeout 可选（默认 24h）
print(done.finetuned_model)

# 事件流（对应 FineTunes.stream_events）
for event in client.fine_tunes.stream_events(ft.job_id):
    print(event.output)

# 训练日志（对应 1.x FineTunes.logs）
logs = client.fine_tunes.logs(ft.job_id, offset=1, lines=200)  # offset/lines 可选
print(logs.content)

# 其余管理操作
client.fine_tunes.cancel(ft.job_id)
jobs = client.fine_tunes.list(page_no=1, page_size=10)   # page_no/page_size 可选
one = client.fine_tunes.get(ft.job_id)
client.fine_tunes.delete(ft.job_id)
```

`create/get/wait` 返回 `FineTune`（信封已扁平化，字段直接在对象上）：

```json
{
  "request_id": "...",
  "job_id": "ft-xxx",
  "job_name": "my-ft-name",
  "status": "SUCCEEDED",
  "model": "qwen-turbo",
  "base_model": "qwen-turbo",
  "finetuned_model": "ft-20260827-xxxx",
  "training_file_ids": ["file-xxx"],
  "validation_file_ids": [],
  "hyper_parameters": {"n_epochs": 2},
  "created_at": "...",
  "finished_at": "..."
}
```

`ft.task_id` 是 `job_id` 的别名（与 `Job` 对齐）；`ft.succeeded` / `ft.finished`
判断状态。服务端的 `finetuned_output` / `create_time` / `end_time` 是校验别名，
读到同一份值。

`list` 返回 `FineTuneList`（`jobs` + `page_no`/`page_size`/`total_count`）；
`stream_events` 逐条产出 `FineTuneEvent`（`output` 为事件内容，`is_error` 标记
错误帧）；`logs` 返回 `JobLogs`（`request_id` / `job_id` / `content` 文本）；
`cancel` / `delete` 返回 `JobStatus`（`request_id` / `status`）。

### 5.2 `client.deployments.*`（1.x：`Deployments.*`）

```python
# 部署微调产物（对应 1.x Deployments.call）
dep = client.deployments.create(
    model="qwen-turbo",
    capacity=2,
    model_version="ft-20260827-xxxx",   # 可选，微调出的模型版本
    suffix="my-suffix",                 # 可选
    # 其余可选：plan / template_id / deploy_spec / billing_method
)
print(dep.deployed_model, dep.status)

client.deployments.scale(dep.deployed_model, capacity=4)   # 扩缩容
print(client.deployments.get(dep.deployed_model).status)
for d in client.deployments.list(page_size=20).deployments:  # 分页参数可选
    print(d.deployed_model, d.status)       # d 为 Deployment，字段直接可读
for event in client.deployments.stream_events(dep.deployed_model):  # 部署事件流
    print(event)
client.deployments.delete(dep.deployed_model)
```

`create/get/scale` 返回 `Deployment`（`deployed_model` / `model` / `base_model` /
`status` / `capacity` / `base_capacity` / `ready_capacity` / `workspace_id` /
`charge_type` / `created_at` / `updated_at` + `request_id`）；`list` 返回
`DeploymentList`（`deployments` + `page_no`/`page_size`/`total_count`）；`delete`
返回 `JobStatus`（`request_id` / `status`）。

`deployed_model` 是后续 `get` / `scale` / `delete` 的定位键，作为位置参数传入。

### 5.3 `client.models.list/get`（1.x：`Models.list/get`）

```python
page = client.models.list(page_no=1, page_size=20, provider="qwen")  # 全部可选
for m in page.models:
    print(m.model, m.provider)

m = client.models.get("qwen3.7-plus")       # 不存在抛 NotFoundError
print(m.model)
```

`list` 返回 `ModelList`：

```json
{
  "request_id": "...",
  "models": [
    {"model": "qwen3.7-plus", "source": "...", "provider": "qwen", "open_source": false}
  ],
  "page_no": 1,
  "page_size": 20,
  "total_count": 100
}
```

`get` 返回单个 `Model`（`list(provider=...)` 的过滤在客户端按 provider/source
匹配）。服务端的 `opensource` 是 `open_source` 的校验别名。

### 5.4 `client.files.*`（1.x：`Files.upload/list/get/delete`）

```python
# purpose="fine-tune" 时本地先做 JSONL 校验
up = client.files.upload(
    file_path="./train.jsonl",
    purpose="fine-tune",          # 可选，默认 "fine-tune"
    description="训练集 v1",       # 可选
)
print(up.file_id)

for f in client.files.list(page_no=1, page_size=10).files:  # 分页参数可选
    print(f.file_id, f.name)

info = client.files.get(up.file_id)
client.files.delete(up.file_id)
```

`upload` / `get` 返回 `File`（`file_id` / `name` / `purpose` / `size` /
`uploaded_at`；信封已扁平化，不再需要 `.data["file_id"]`）；`list` 返回
`FileList`（`files` + `page_no`/`page_size`/`total_count`）；`delete` 返回
`DeletionStatus`（`deleted` 布尔值 + `request_id`）。服务端的 `upload_time` 是
`uploaded_at` 的校验别名。

## 6. Agentic RL

对应 1.x CLI `dashscope rl` 子命令的编程化使用（`client.rl`；
`client.reinforcement_learning` 是同一实例的长名别名）。资源面分两层：
`client.rl.*` 是本地准备动作，`client.rl.jobs.*` 是训练任务管理。

```python
# 注册 rollout / reward 函数（rl register_functions）
registered = client.rl.register_functions(
    rollout_classpaths=["my_project.rollout:MyRollout"],   # 可选
    reward_classpaths=["my_project.reward:MyReward"],      # 可选
    group_reward_classpaths=["my_project.group:MyGroup"],  # 可选
    workspace_dir="./",                                    # 可选，默认 "./"
    lazy_load=True,                                        # 可选，默认 True
)
print(registered.rollout_instance_ids)

# 本地测试函数（rl test_functions）：三个参数均必填
client.rl.test_functions(
    instance_id=registered.rollout_instance_ids[0],
    func_type="rollout",
    input_data={"query": "1+1=?"},
)

# 上传训练 / 验证数据（rl upload_data）
ids = client.rl.upload_data(
    training_files=["./rl_train.jsonl"],
    validation_files=["./rl_val.jsonl"],   # 可选
)
print(ids.training_file_ids)

# 启动完整训练流程（rl run）
job = client.rl.jobs.create(config_path="./rl_config.yaml", job_name="my-rl-job")
# 两参数均可选；config 里的其余字段用 **kwargs 透传

# 任务管理（rl get / list / cancel / logs / wait，v2 错误模型）
print(client.rl.jobs.fetch(job.job_id).status)
print(client.rl.jobs.list(page_size=5))                # page_no/page_size 可选
print(client.rl.jobs.logs(job.job_id, lines=100).content)  # offset/lines 可选
client.rl.jobs.cancel(job.job_id)
client.rl.jobs.delete(job.job_id)
done = client.rl.jobs.wait(job)

# 函数实例管理（1.x 直通，同样无 timeout=）
client.rl.query_function_instance_logs(instance_id)       # 单页日志（原始 dict）
client.rl.query_all_function_instance_logs(instance_id)   # 聚合全部分页
client.rl.delete_function_instance(instance_id)
```

返回值：`register_functions` 返回 `RegisteredFunctions`（`rollout_entity_ids` /
`reward_entity_ids` / `group_reward_entity_ids` 及对应的 `*_instance_ids`）；
`upload_data` 返回 `UploadedData`（`training_file_ids` / `validation_file_ids`）；
`test_functions` 返回 1.x 原对象（业务逻辑复用 1.x）；`jobs.create` /
`jobs.fetch` / `jobs.wait` 复用微调资源模型 `FineTune`，`jobs.list` 复用
`FineTuneList`，`jobs.cancel` 返回 `JobStatus`，`jobs.logs` 返回 `JobLogs`
（结构见 §5.1）。

**这些直通方法没有 `timeout=`**：`jobs.create` / `register_functions` /
`test_functions` / `upload_data` / `query_function_instance_logs` /
`query_all_function_instance_logs` / `delete_function_instance` 直通 1.x 业务逻辑
（涉及本地打包上传与 YAML 解析），超时由 1.x 内部控制；`jobs.fetch` /
`jobs.list` / `jobs.cancel` / `jobs.delete` / `jobs.logs` / `jobs.wait` 走 v2
传输层，`timeout=` 可用。

**但两半的错误行为是统一的**：直通方法的 1.x 异常在挂载层翻成 `SdkError`
（背后没有 HTTP 响应，`status_code is None`，另带目录给出的 `solution`），
任务管理走 v2 传输层抛 `APIStatusError` 子类。两者都是 `DashScopeError`
的子类，所以**一个 `except DashScopeError` 覆盖整个 `client.rl`**（示例见 §9.5）。

## 7. AgentStudio（托管 Agent）

2.0 **完整包含** AgentStudio：`client.agentstudio` 是把 1.x 子 SDK
（`dashscope.agentstudio`）原样懒挂载进来，功能、参数、返回模型与 1.x 完全一致，
不是裁剪版。同步 Client 挂 `agentstudio.Client`，异步 Client 挂
`agentstudio.AsyncClient`（方法同名同参，加 `await` / `async for` 即可）。

典型链路：建 Agent → 开会话 → 发消息 → 流式读回复：

```python
as_client = client.agentstudio            # 首次访问时才构造

# 1. 创建托管 Agent（name / model 必填；tools / mcp_servers / skills /
#    multiagent / system_prompt / description / metadata 可选）
agent = as_client.agents.create(
    name="demo",
    model="qwen3.7-plus",
    system_prompt="你是助手",              # 可选
)

# 2. 创建会话（agent 必填，传 id 或 {"id": ..., "version": ...}；
#    environment_id / title / resources / vault_ids /
#    environment_variables / mcp_configs / metadata 可选）
session = as_client.sessions.create(agent=agent.id, title="第一次对话")

# 3. 发送事件 + 流式读取回复（user_message 是客户端事件构造辅助函数，
#    同模块还有 user_interrupt / user_tool_result 等）
from dashscope.agentstudio import user_message

as_client.sessions.events.send(session.id, [user_message("你好")])
with as_client.sessions.events.stream(session.id) as stream:
    for text in stream.text_stream:       # 逐条产出完整文本块
        print(text, end="")
# 增量文本：events.stream(session.id, event_deltas=["message"]) 后用
# stream.text_deltas；直接 for event in stream 则产出类型化 ServerEvent

# 4. 无需单独关闭：client.close()（异步 aclient.aclose()）会连带关闭它
```

资源与方法全集（`*` 为必填；未标注的返回类型即方法名对应模型）：

| 资源 | 方法与关键入参 | 返回 |
| --- | --- | --- |
| `agents` | `create(name*, model*, description/system_prompt/tools/mcp_servers/skills/multiagent/metadata)`、`retrieve(agent_id, version=)`（别名 `get`）、`update(agent_id, version=*, ...)`、`archive(agent_id)`、`list(limit/page/include_archived)`、`list_versions(agent_id)` | `Agent` / `AgentVersion` |
| `sessions` | `create(agent*, environment_id/title/resources/vault_ids/environment_variables/mcp_configs/metadata)`、`retrieve`、`update(title/metadata/vault_ids/...)`、`list(agent_id/statuses/created_at_gt/gte/lt/lte)`、`archive`、`delete` | `Session` / `DeleteResponse` |
| `sessions.events` | `send(session_id, events*)`（空列表抛 `ValueError`）、`list(types/created_at_*/limit/order)`、`stream(session_id, event_deltas=, timeout=)` | `CursorPage[ServerEvent]` / 事件流 |
| `sessions.resources` | `add(session_id, file_id*, resource_type="file", mount_path=)`、`retrieve`、`list`、`delete` | `SessionResource` |
| `sessions.threads` | `list(session_id)`、`retrieve(session_id, thread_id)`、`archive(...)`；`threads.events.list(session_id, thread_id, ...)`（仅 list，无流式） | `SessionThread` |
| `files` | `upload(file*, mime_type=, progress=)`（路径 / 文件对象均可）、`retrieve`、`download(file_id)`（`FileContent` 是 bytes 子类，带 `.write_to_file(path)`）、`list(scope_id=)`、`delete` | `File` |
| `skills` | `create(file_id= 或 file=，二选一)`、`retrieve`、`list(source=)`、`delete`；`skills.versions.create/list/retrieve/download` | `Skill` / `SkillVersion` |
| `vaults` | `create(display_name*, metadata=)`、`retrieve`、`update`、`list(keyword=)`、`archive`、`delete`；`vaults.credentials.create(vault_id, auth*)/retrieve/update/list/archive/delete` | `Vault` / `Credential` |
| `environments` | `create(name*, config*, description/scope/metadata)`（`scope` 默认 `"organization"`）、`retrieve`、`update`、`list`、`archive`、`delete` | `Environment` |
| `deployments` | `create(name*, agent*, initial_events*, description/environment_id/schedule/resources/vault_ids/environment_variables/metadata)`、`retrieve`、`update`、`list(agent_id/keyword/status/created_at_gte/lte)`、`pause` / `unpause` / `archive`、`run(deployment_id)`、`list_runs(deployment_id)` | `Deployment` / `DeploymentRun` |
| `deployment_runs` | `retrieve(deployment_run_id)`、`list` | `DeploymentRun` |
| `webhook_endpoints` | `create(url*, events*, description=)`、`retrieve`、`update`、`delete`、`enable` / `disable`、`test`、`reset_secret`、`list()`、`list_events(webhook_id)` | `WebhookEndpoint` / `WebhookEvent` |
| `security` | `overview()`、`list_agent_logs(current_page/page_size/risk_level/status/risk_name/...)`（页码分页） | `SecurityOverview` / `SecurityAlertList` |

分页统一为 `CursorPage[T]`（异步 `AsyncCursorPage[T]`）：`data` 当前页列表、
`next_page` 下一页游标、`has_next()` / `get_next()` 手动翻页；直接 `for` 迭代
会自动翻完所有页。

agentstudio 各方法的返回类型是 1.x 子 SDK 自带模型（`Agent` / `Session` /
`ServerEvent` 等），不属于 v2 类型体系，字段以 1.x 为准（响应仍是 `output`
信封，不扁平化）。**异常则已对齐 v2**：挂载层用透明代理把 1.x 异常翻成
`dashscope.v2.errors` 一族，分两条路——

- **HTTP 层错误**（1.x `APIStatusError` 一族）：按 `status_code` 进状态映射，
  抛 v2 同名状态类（`BadRequestError` / `AuthenticationError` /
  `PermissionDeniedError` / `NotFoundError` / `RateLimitError` /
  `InternalServerError` / `ServiceUnavailableError`；409 等目录未收录的状态
  落通用 `APIStatusError`），`request_id` 保留。1.x 在本地抛的、没带
  `status_code` 的 HTTP 族错误（如缺 `api_key` 的 `AuthenticationError`）
  按其类代表的状态映射，不会退化成 500。
- **无响应的本地 / 流式错误**（1.x `APIConnectionError` / `APITimeoutError` /
  `StreamError` / `StreamClosedError`）：翻成 `SdkError`，`code` 为
  `sdk.agentstudio.*`，带目录给出的 `solution` / `allow_retry`（见 §9.5）。

所以**一个 `except DashScopeError` 覆盖整个 `client.agentstudio`**：

```python
from dashscope.v2.errors import DashScopeError, NotFoundError, SdkError

try:
    with client.agentstudio.sessions.events.stream(session.id) as stream:
        for text in stream.text_stream:
            print(text, end="")
except NotFoundError:
    ...                # session 不存在（HTTP 404）
except SdkError as e:
    ...                # 连接 / 超时 / 流式协议错误，e.solution 可直接展示
except DashScopeError:
    ...                # 其余全部
```

代理对 `isinstance`（`isinstance(client.agentstudio, AgentStudioClient)` 仍为
真）、`.transport` 访问和上下文管理协议透明；子资源、流式事件流与分页
（`CursorPage` / `AsyncCursorPage`）递归包裹，**迭代与翻页中途抛的错误同样被
翻译**（翻页抓取走的是代理守卫，自动翻页和显式 `get_next()` 都覆盖）。注意
翻译只发生在 v2 挂载层：直接构造 1.x 的 `dashscope.agentstudio.Client` 仍抛
原生异常族。

懒挂载需要 `workspace`（v2 只向它传 api_key / workspace / region，不透传
`base_url`）：workspace 缺失、且未配置 `DASHSCOPE_AGENTSTUDIO_URL` /
`AGENTSTUDIO_URL` 环境变量时，首次访问 `client.agentstudio` 抛 `ValueError`。

## 8. 全局配置

### 8.1 Client 参数或 `dashscope.settings`（1.x：`dashscope.api_key` / `save_api_key()`）

```python
import dashscope

# 方式一：每个 Client 独立配置（推荐，对应 1.x 配置文件 resolve_config）
client = dashscope.Client(
    api_key="sk-xxx",
    workspace="ws-xxx",
    config_path="~/.dashscope/config.toml",
)

# 方式二：全局门面，赋值即时生效（对应 1.x 运行时改模块变量）
dashscope.settings.api_key = "sk-xxx"          # 未显式赋值时懒读 DASHSCOPE_API_KEY
dashscope.settings.base_url = "https://ws-xxx.cn-beijing.maas.aliyuncs.com/api/v1"
dashscope.settings.workspace = "ws-xxx"
dashscope.settings.region = "cn-beijing"
dashscope.settings.protocol = "native"
dashscope.settings.timeout = 30.0
dashscope.settings.max_retries = 3
dashscope.settings.clear()                     # 清掉显式赋值，回落到环境变量
```

配置优先级：**显式参数 > settings 赋值 > 环境变量 > config.toml > 默认值**。
`config.toml` 支持 `api_key` / `base_url` / `workspace` / `region` / `protocol`
等键；`api_key` 的最低兜底还有 1.x 的 key 文件 `~/.dashscope/api_key`。

`with_options()` 从一个已有 Client 派生出只改若干配置的新实例（共享连接池）：

```python
strict = client.with_options(protocol="openai", timeout=10.0)
```

### 8.2 日志级别统一为 `DASHSCOPE_LOG`

```python
# 环境变量：DASHSCOPE_LOG=DEBUG（取代 1.x 的 DASHSCOPE_LOGGING_LEVEL / LOG_LEVEL）
# 代码内：
dashscope.settings.log_level = "DEBUG"
```

### 8.3 websocket 地址

```python
# 全局：环境变量 DASHSCOPE_WEBSOCKET_BASE_URL=wss://...
#       或代码内 dashscope.settings.websocket_base_url = "wss://..."
# 单次调用：资源方法的 ws_url= 参数
audio = client.audio.tts.synthesize(
    "你好",
    ws_url="wss://ws-xxx.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",  # 单次覆盖
)
```

### 8.4 实例级关闭（1.x：`close_shared_aio_session()`）

```python
client.close()                  # 同步
await aclient.aclose()          # 异步
# 也可用上下文管理：
# with dashscope.Client() as client: ...
# async with dashscope.AsyncClient() as aclient: ...
```

### 8.5 HTTP 连接池（keep-alive）

每个 `Client` 持有一个 httpx 连接池，所有资源的 HTTP 请求默认复用连接
（1.x 每次调用新建 `requests.Session`）；`with_options` 未改连接级选项时
共享同一个池。池参数走 settings / 环境变量 / config.toml：

```python
dashscope.settings.http_max_connections = 50            # DASHSCOPE_HTTP_MAX_CONNECTIONS，默认 100
dashscope.settings.http_max_keepalive_connections = 10  # DASHSCOPE_HTTP_MAX_KEEPALIVE_CONNECTIONS，默认 20；0 = 关闭
dashscope.settings.http_keepalive_expiry = 30.0         # DASHSCOPE_HTTP_KEEPALIVE_EXPIRY，默认 5.0 秒
```

池是 per-Client 的：长驻一个 client 才能吃到复用，频繁新建实例会击穿。

### 8.6 环境变量一览

| 环境变量 | 作用 | 默认 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | API Key | — |
| `DASHSCOPE_API_KEY_FILE_PATH` | 从文件读 API Key 的路径 | — |
| `DASHSCOPE_WORKSPACE` | workspace id | —（缺省回退共享网关 + 告警） |
| `DASHSCOPE_REGION` | 地域（六选一） | `cn-beijing` |
| `DASHSCOPE_BASE_URL` | 自定义 HTTP 端点 | — |
| `DASHSCOPE_PROTOCOL` | `native` / `openai` / `anthropic` | `native` |
| `DASHSCOPE_TIMEOUT` | 单次请求超时（秒） | — |
| `DASHSCOPE_MAX_RETRIES` | 自动重试次数 | `2` |
| `DASHSCOPE_CONFIG` | config.toml 路径 | `~/.dashscope/config.toml` |
| `DASHSCOPE_LOG` | 日志级别（debug/info/warning/error） | `warning` |
| `DASHSCOPE_WEBSOCKET_BASE_URL` | ws 端点（不含 realtime 族） | 跟随 workspace/region |
| `DASHSCOPE_HTTP_MAX_CONNECTIONS` | HTTP 连接池上限 | `100` |
| `DASHSCOPE_HTTP_MAX_KEEPALIVE_CONNECTIONS` | keep-alive 连接数（0=关闭） | `20` |
| `DASHSCOPE_HTTP_KEEPALIVE_EXPIRY` | keep-alive 空闲回收（秒） | `5.0` |
| `DASHSCOPE_AGENTSTUDIO_URL` / `AGENTSTUDIO_URL` | agentstudio 端点（无 workspace 时的逃生口） | — |

## 9. 错误处理

规范正文见 [`md/dashscope-2.0-design.md`](dashscope-2.0-design.md) §2.7：异常类名与
HTTP 状态一律从生成的错误目录 `dashscope/v2/_error_registry.py` 派生，不在 SDK 里手写。

### 9.1 异常族

具体异常类从 `dashscope.v2.errors` import；`dashscope.v2` 只便捷再导出
`DashScopeError` 与 `ErrorCode`（包根 `dashscope` 都不导出）：

```python
from dashscope.v2 import DashScopeError              # 基类，够用时只 import 这一个
from dashscope.v2.errors import (
    APIStatusError,          # 网关声明的失败（9.2 的两层都算）
    APIConnectionError,      # 连不上，无响应
    APITimeoutError,         # 超时（APIConnectionError 子类）
    ContentFilterError,      # 内容审核拦截（BadRequestError 子类）
    SdkError,                # SDK 自己报的失败，无响应
    ErrorCode,               # 线缆错误码枚举（13 项）
)
```

9 个状态类是 `APIStatusError` 的**扁平兄弟**：

| 类 | `status_code` | 什么时候抛 |
| --- | --- | --- |
| `BadRequestError` | 400 | 参数非法 / 缺参数 / 文件非法 / URL 非法 |
| `AuthenticationError` | 401 | 认证失败 / API Key 无效 |
| `PermissionDeniedError` | 403 | 无权限、未开通、**欠费** |
| `NotFoundError` | 404 | 模型或资源不存在 |
| `RequestTooLargeError` | 413 | 请求体超限 |
| `RateLimitError` | 429 | 限流、并发超限、配额用尽 |
| `InternalServerError` | 500 | 服务端内部错误 |
| `ServiceUnavailableError` | 503 | 服务不可用 |
| `GatewayTimeoutError` | 504 | 网关超时 |

`ContentFilterError` 是 `BadRequestError` 的**子类**（不是第 10 个兄弟），所以
`except BadRequestError` 也抓得到审核拦截；要单独处理就把
`except ContentFilterError` 写在**前面**：

```python
try:
    client.chat.completions.create(model="qwen3.7-plus", messages=[{"role": "user", "content": "..."}])
except ContentFilterError as e:      # 必须排在 BadRequestError 之前
    print("内容被审核拦截", e.request_id)
except BadRequestError as e:         # 其余 400
    print("请求参数有问题", e.code, e.message)
```

`APIConnectionError` / `APITimeoutError` / `SdkError` 的 `status_code` 是 `None`：
它们背后没有 HTTP 响应。目录未收录的状态码（502 / 507 / 418…）落
`APIStatusError` 本身，SDK 不猜它是哪一类。

### 9.2 两层错误，一个异常族

DashScope 网关除了返回非 2xx，也可能返回 **HTTP 200 但 body 带业务错误码**
（`code` / `message`）。两层抛的是**同一族异常**，实例的 `status_code`
说明是哪一层：

```python
from dashscope.v2.errors import RateLimitError

try:
    resp = client.chat.completions.create(model="qwen3.7-plus", messages=[{"role": "user", "content": "..."}])
except RateLimitError as e:
    # e.status_code == 429 → HTTP 层限流
    # e.status_code == 200 → 业务层限流（线缆码 Throttling / .RateQuota / .AllocationQuota）
    # 两种都该退避重试，所以按类 catch 就够，不必区分
    print(e.code, e.status_code, e.request_id)
```

线缆码（`InvalidApiKey` / `Throttling` / `DataInspectionFailed`…）与目录 key
是两个命名空间：SDK 内部用 crosswalk 把线缆码对到目录项，再由目录项决定异常类。
用户只需认类，`e.code` 保留原始线缆码供排障。业务层出现的**未知线缆码**
（crosswalk 未收录）落 `DashScopeError` 基类本身，`status_code` 同样是 200。

### 9.3 异常上能拿到什么

每个异常都携带同一组字段：

```python
try:
    client.images.generate(model="wan2.2-t2i-flash", prompt="一只猫")
except DashScopeError as e:
    e.message        # 人类可读消息
    e.code           # 线缆错误码，如 "Throttling.RateQuota"；SDK 侧错误是目录名
    e.status_code    # HTTP 状态；业务层是 200；无响应时是 None
    e.request_id     # 排障用，提工单必带
    e.body           # 原始响应体
    e.headers        # 响应头
```

`SdkError` 额外有 `solution` / `allow_retry`（见 9.5）。

### 9.4 重试

`Client(max_retries=N)` 默认 **2** 次，自动重试 408 / 409 / 429 / 500 / 502 /
503 / 504 与连接错误，优先按服务端的 `Retry-After`（含 `retry-after-ms`）等待，
否则指数退避 + 抖动。重试发生在传输层，按**原始状态码**判定，与最终抛出的
异常类无关；服务端也可以用 `x-should-retry` 响应头显式要求或禁止某次重试
（优先于状态码判定）。

SDK 已经重试过还抛出来，说明这一轮没救回来。是否再手工重试要看错误类别：

```python
import time
from dashscope.v2.errors import (
    GatewayTimeoutError,
    RateLimitError,
    ServiceUnavailableError,
)

# 只有这几类值得再来一轮；401/403/404/400 重试没有意义，同样的请求还会同样失败
RETRYABLE = (RateLimitError, ServiceUnavailableError, GatewayTimeoutError)

for attempt in range(3):
    try:
        resp = client.chat.completions.create(
            model="qwen3.7-plus",
            messages=[{"role": "user", "content": "..."}],
        )
        break
    except RETRYABLE:
        if attempt == 2:
            raise
        time.sleep(2**attempt)
```

### 9.5 `SdkError`：SDK 自己报的失败

`client.rl` 的 1.x 直通方法（`jobs.create` / `register_functions` /
`test_functions` / `upload_data` / 三个函数实例管理方法）涉及本地打包上传与
YAML 解析，失败时背后**没有 HTTP 响应**，抛 `SdkError`：

```python
from dashscope.v2.errors import SdkError

try:
    client.rl.upload_data(training_files=["./rl_train.jsonl"])
except SdkError as e:
    e.code          # 目录名，如 "sdk.agentic_rl.OSSUploadError"
    e.solution      # 目录随附的修复建议，可直接展示给用户
    e.allow_retry   # 该错误是否值得重试；SDK 不自动重试这一层
    e.status_code   # None
    e.__cause__     # 原始 1.x 异常，排障时可看
```

`SdkError` 是 `DashScopeError` 的子类，所以 `client.rl` 的两半——直通 1.x 的
工作流步骤、走 v2 传输的任务管理——**一个 `except DashScopeError` 全覆盖**：

```python
try:
    registered = client.rl.register_functions(rollout_classpaths=["my_project.rollout:MyRollout"])
    job = client.rl.jobs.create(config_path="./rl_config.yaml")
    done = client.rl.jobs.wait(job)
except DashScopeError as e:
    # 上传失败 → SdkError；任务查询/取消失败 → APIStatusError 子类
    print(type(e).__name__, e.code, e.message, e.request_id)
```

`client.agentstudio` 同样做了这层翻译（§7）：挂载层代理把 1.x 异常翻成 v2
族——HTTP 层错误进状态映射，无响应的本地 / 流式错误翻成 `SdkError`
（`sdk.agentstudio.*` 一族），`except DashScopeError` 同样全覆盖。

### 9.6 关掉抛异常：`raise_on_error=False`

```python
lenient = client.with_options(raise_on_error=False)
resp = lenient.chat.completions.create(
    model="qwen3.7-plus",
    messages=[{"role": "user", "content": "..."}],
)
if resp.is_error:                      # ResponseBase 从非空 code 自动推导
    print(resp.code, resp.message, resp.request_id)
```

返回值模式下不抛异常，改由每个顶层响应继承的 `ResponseBase` 携带
`code` / `message` / `is_error`。流式则是产出一个 `is_error=True` 的对象后
结束迭代（`client.apps` 对齐 1.x `Application` 语义）。**连接失败与超时仍然
抛异常**：那两种情况根本没有响应对象可返回。
