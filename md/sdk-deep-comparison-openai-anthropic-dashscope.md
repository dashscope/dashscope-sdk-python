# SDK 深度对比：OpenAI vs Anthropic vs DashScope

> 基于源码级分析
> 生成时间：2026-07-31
> OpenAI SDK v1.x | Anthropic SDK v0.39+ | DashScope SDK v1.x
> 来源：钉钉文档（alidocs YMyQA2dXW7gYo6MzcZR0MPdZWzlwrZgb）

## 目录

1. 用户接口友好度
2. 错误码与异常处理
3. 功能使用：流式/非流式/同步/异步
4. 架构设计哲学
5. DashScope SDK 改进建议

---

## 1. 用户接口友好度

### 1.1 最简调用对比

**OpenAI — 极致简洁**

```python
from openai import OpenAI

client = OpenAI()  # api_key 从环境变量读取

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)

print(response.choices[0].message.content)
```

亮点：

- 零配置即可使用（环境变量 `OPENAI_API_KEY`）
- `client.chat.completions.create()` — 点号链式调用，IDE 自动补全完美
- Pydantic 模型，所有字段都有类型提示
- `response.choices[0].message.content` — 属性访问，不是 dict 索引

**Anthropic — 同样简洁**

```python
from anthropic import Anthropic

client = Anthropic()  # api_key 从环境变量读取

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)

print(message.content[0].text)
```

亮点：

- 与 OpenAI 几乎相同的模式
- `message.content[0].text` — 直接返回结构化 content blocks
- Pydantic 模型，完整的类型提示

**DashScope — 多层抽象**

```python
import dashscope
from dashscope import Generation

dashscope.api_key = "sk-xxx"  # 必须手动设置

response = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Hello"}],
)

# 需要先检查 status_code
if response.status_code == 200:
    print(response.output.choices[0].message.content)
else:
    print(f"Error: {response.code} - {response.message}")
```

问题：

- 需要手动设置 api_key（虽然支持环境变量，但不如 OpenAI 自然）
- `Generation.call()` — 类方法调用，不支持 IDE 点号链式补全
- 返回 `DashScopeAPIResponse` 是 dict 子类（DictMixin），不是 Pydantic 模型
- 成功/失败通过 status_code 判断，不抛异常 — 调用方必须每次检查

### 1.2 Client 模式对比

| 特性 | OpenAI | Anthropic | DashScope |
| --- | --- | --- | --- |
| Client 实例化 | `OpenAI(api_key=...)` | `Anthropic(api_key=...)` | 无 Client 模式，使用类方法 |
| API Key 来源 | 环境变量 / 参数 / 文件 | 环境变量 / 参数 | 模块变量 / 环境变量 |
| Base URL 覆盖 | `base_url="..."` | `base_url="..."` | `dashscope.base_http_api_url` |
| 连接池 | httpx.Client 复用 | httpx.Client 复用 | 每次请求新建 requests.Session() |
| 默认超时 | 10 min（connect 5s） | 10 min（connect 5s） | 300s |
| 最大重试 | 2 次（指数退避） | 2 次（指数退避） | 1 次（仅 5xx 重试） |
| IDE 自动补全 | ✅ 完美 | ✅ 完美 | ⚠️ 部分（类方法 + dict 返回） |

### 1.3 响应对象对比

| 特性 | OpenAI | Anthropic | DashScope |
| --- | --- | --- | --- |
| 响应类型 | Pydantic BaseModel | Pydantic BaseModel | DictMixin（dict 子类） |
| 属性访问 | `response.choices[0].message.content` | `message.content[0].text` | `response.output.choices[0].message.content` |
| 字典访问 | `response.model_dump()` | `message.model_dump()` | `response["output"]` |
| 序列化 | `response.model_dump_json()` | `message.model_dump_json()` | `str(response)` → JSON |
| 类型安全 | ✅ 完整 Pydantic 验证 | ✅ 完整 Pydantic 验证 | ⚠️ 无验证（dict 无 schema） |
| 嵌套深度 | `choices[0].message.content` | `content[0].text` | `output.choices[0].message.content`（多一层 output） |
| `__repr__` | 美观的 Pydantic 格式 | 美观的 Pydantic 格式 | `DictMixin({...})` 原始 dict |

### 1.4 API 设计风格对比

| 方面 | OpenAI | Anthropic | DashScope |
| --- | --- | --- | --- |
| API 风格 | REST 资源嵌套 (`client.chat.completions`) | REST 扁平 (`client.messages`) | 类方法扁平 (`Generation.call()`) |
| 方法命名 | create / list / retrieve / delete | create / list / retrieve / delete | call / list / get / delete / update |
| 参数传递 | 命名参数 + Pydantic 验证 | 命名参数 + Pydantic 验证 | `**kwargs` 大量使用，无验证 |
| 模型名传递 | 字符串 | 字符串 | 字符串 |
| Headers | `default_headers` 属性 | `default_headers` 属性 | `default_headers()` 函数 |

---

## 2. 错误码与异常处理

### 2.1 错误处理哲学

| SDK | 哲学 | 模式 |
| --- | --- | --- |
| OpenAI | 失败即抛异常 | try/except 捕获，异常类型区分错误类别 |
| Anthropic | 失败即抛异常 | 同 OpenAI，但异常层次更扁平 |
| DashScope | 返回错误对象 | `if response.status_code != 200` 检查，不抛异常 |

### 2.2 异常类层次

**OpenAI**

```
openai.OpenAIError (基类)
├── openai.APIError                          # 通用 API 错误
├── openai.APIConnectionError                # 连接失败
│   └── openai.APITimeoutError              # 超时
├── openai.RateLimitError                    # 429 限流
├── openai.BadRequestError                   # 400 请求错误
├── openai.AuthenticationError               # 401 认证失败
├── openai.PermissionDeniedError             # 403 权限不足
├── openai.NotFoundError                     # 404 资源不存在
├── openai.ConflictError                     # 409 冲突
├── openai.UnprocessableEntityError          # 422 参数错误
├── openai.InternalServerError               # 500 服务器错误
└── openai.LengthFinishReasonError           # 生成被截断
```

异常属性：

```python
except openai.APIError as e:
    e.status_code    # HTTP 状态码
    e.message        # 错误消息
    e.body           # 响应体（dict）
    e.request_id     # 请求 ID
    e.code           # 错误码
    e.headers        # 响应头
```

**Anthropic**

```
anthropic.AnthropicError (基类)
├── anthropic.APIError                       # 通用 API 错误
│   ├── anthropic.BadRequestError            # 400
│   ├── anthropic.AuthenticationError        # 401
│   ├── anthropic.PermissionDeniedError      # 403
│   ├── anthropic.NotFoundError              # 404
│   ├── anthropic.ConflictError              # 409
│   ├── anthropic.UnprocessableEntityError   # 422
│   ├── anthropic.RateLimitError             # 429
│   └── anthropic.InternalServerError        # 500+
├── anthropic.APIConnectionError             # 连接失败
│   └── anthropic.APITimeoutError           # 超时
└── anthropic.APIStatusError                 # 带完整响应的错误
    └── 所有上面的子类
```

异常属性：

```python
except anthropic.APIError as e:
    e.status_code    # HTTP 状态码
    e.message        # 错误消息
    e.body           # 响应体
    e.response       # 完整 httpx.Response 对象
    e.request_id     # 请求 ID（从 headers 提取）
    e.error          # APIErrorBody 对象（type + message）
```

**DashScope**

无异常类。所有错误通过 `DashScopeAPIResponse` 返回：

```python
response = Generation.call(model="qwen-plus", messages=[...])

# 必须手动检查
if response.status_code != 200:
    print(f"Error: {response.code}")
    print(f"Message: {response.message}")
    print(f"Request ID: {response.request_id}")
```

DashScope 的 common/error.py 仅有 3 个异常：

```python
class Error(Exception): ...
class InputRequired(Error): ...
class ModelRequired(Error): ...
class InvalidParameter(InputRequired): ...
class InvalidTask(Error): ...
class NotSupport(Error): ...
```

这些仅用于客户端参数校验，不用于 API 错误。

### 2.3 错误处理代码对比

**OpenAI — 类型安全**

```python
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

client = OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.choices[0].message.content)

except RateLimitError as e:
    print(f"Rate limited: {e.message}, retry after {e.headers.get('retry-after')}")
except APIConnectionError as e:
    print(f"Connection failed: {e.message}")
except APIError as e:
    print(f"API error {e.status_code}: {e.message}, request_id: {e.request_id}")
```

**Anthropic — 类似但更扁平**

```python
from anthropic import Anthropic, RateLimitError, APIConnectionError

client = Anthropic()

try:
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(message.content[0].text)

except RateLimitError as e:
    print(f"Rate limited: {e.message}")
except APIConnectionError as e:
    print(f"Connection failed: {e.message}")
except Exception as e:
    print(f"Unexpected: {e}")
```

**DashScope — 手动检查**

```python
import dashscope
from dashscope import Generation

response = Generation.call(model="qwen-plus", messages=[...])

# ❌ 容易忘记检查，容易遗漏错误
if response.status_code == 200:
    print(response.output.choices[0].message.content)
else:
    # 错误码是字符串，如 "InvalidApiKey", "DataInspectionFailed"
    print(f"Error {response.code}: {response.message}")
    print(f"Request ID: {response.request_id}")
```

### 2.4 错误码体系

| 特性 | OpenAI | Anthropic | DashScope |
| --- | --- | --- | --- |
| 错误码类型 | Python 异常类 + HTTP status code | Python 异常类 + HTTP status code | 字符串 code（"InvalidApiKey" 等） |
| HTTP 状态码映射 | 400/401/403/404/409/422/429/500 | 400/401/403/404/409/422/429/500 | 全部混在 response.status_code |
| 错误分类 | 按 HTTP 状态自动分类 | 按 HTTP 状态自动分类 | 字符串枚举，无自动分类 |
| 重试策略 | 自动重试 429/5xx/连接错误 | 自动重试 429/5xx/连接错误 | 仅 5xx 重试 1 次 |
| 超时处理 | APITimeoutError 异常 | APITimeoutError 异常 | requests.Timeout 异常（未捕获） |
| 限流信息 | Retry-After header | Retry-After header | 无 |
| 请求 ID | e.request_id | e.request_id | response.request_id |

### 2.5 DashScope 错误码列表（已知）

| code | 含义 | HTTP Status |
| --- | --- | --- |
| InvalidApiKey | API Key 无效 | 401 |
| AccessDenied | 无权限 | 403 |
| InvalidParameter | 参数错误 | 400 |
| DataInspectionFailed | 内容安全审核失败 | 400 |
| Throttling | 限流 | 429 |
| Throttling.AllocationQuota | 配额耗尽 | 429 |
| ModelNotFound | 模型不存在 | 404 |
| RequestTimeout | 请求超时 | 504 |
| InternalError | 内部错误 | 500 |
| OperationDenied.NoStock | 无库存 | 503 |

问题：错误码是字符串，没有类型安全，调用方需要写字符串比较：

```python
if response.code == "Throttling":
    time.sleep(5)
elif response.code == "InvalidApiKey":
    print("Check your API key")
```

对比 OpenAI/Anthropic 的类型安全：

```python
except RateLimitError:   # 类型安全，IDE 可补全
    time.sleep(5)
except AuthenticationError:
    print("Check your API key")
```

---

## 3. 功能使用：流式/非流式/同步/异步

### 3.1 四种模式矩阵

| 模式 | OpenAI | Anthropic | DashScope |
| --- | --- | --- | --- |
| 同步 + 非流式 | ✅ `client.chat.completions.create()` | ✅ `client.messages.create()` | ✅ `Generation.call()` |
| 同步 + 流式 | ✅ stream=True → Stream[ChatCompletionChunk] | ✅ stream=True → Stream[RawMessageStreamEvent] | ✅ stream=True → Generator[DashScopeAPIResponse] |
| 异步 + 非流式 | ✅ `AsyncOpenAI()` + `await client.chat.completions.create()` | ✅ `AsyncAnthropic()` + `await client.messages.create()` | ✅ `AioGeneration.call()` (需手动实现) |
| 异步 + 流式 | ✅ stream=True → AsyncStream[ChatCompletionChunk] | ✅ stream=True → AsyncStream[RawMessageStreamEvent] | ⚠️ 无原生支持 |

### 3.2 同步非流式 — 最常用模式

**OpenAI**

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=1024,
)

# Pydantic 对象，完整类型提示
content = response.choices[0].message.content
usage = response.usage  # CompletionUsage(prompt_tokens=..., completion_tokens=..., total_tokens=...)
```

**Anthropic**

```python
from anthropic import Anthropic

client = Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)

# Pydantic 对象
content = message.content[0].text
usage = message.usage  # Usage(input_tokens=..., output_tokens=..., cache_creation_input_tokens=...)
```

**DashScope**

```python
from dashscope import Generation

response = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=1024,
)

# DictMixin 对象，无类型提示
if response.status_code == 200:
    content = response.output.choices[0].message.content
    usage = response.usage  # GenerationUsage(input_tokens=..., output_tokens=...)
```

### 3.3 同步流式

**OpenAI — 最优雅**

```python
from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)

# Stream 对象实现了 __iter__，每个 chunk 是 ChatCompletionChunk
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

特点：

- Stream 类型实现了 `__iter__`，直接 for 循环
- 每个 chunk 是 ChatCompletionChunk（Pydantic 模型）
- `chunk.choices[0].delta.content` — 增量内容
- 自动处理 SSE 协议解析
- `stream.close()` 可提前终止

**Anthropic — 事件驱动**

```python
from anthropic import Anthropic

client = Anthropic()

with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

    # 流结束后获取完整消息
    message = stream.get_final_message()
    print(f"\nTotal tokens: {message.usage.input_tokens + message.usage.output_tokens}")
```

特点：

- MessageStreamManager 上下文管理器
- `stream.text_stream` — 纯文本迭代器（最简洁）
- `stream.__iter__()` — 原始事件迭代器（RawMessageStreamEvent）
- `stream.get_final_message()` — 累积后的完整消息
- `stream.get_final_text()` — 累积后的纯文本
- 事件类型丰富：RawMessageStartEvent / RawContentBlockDeltaEvent / RawMessageDeltaEvent 等

**DashScope — Generator 模式**

```python
from dashscope import Generation

responses = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    incremental_output=True,  # 增量输出
)

# 返回 Generator[DashScopeAPIResponse]
for response in responses:
    if response.status_code == 200:
        text = response.output.choices[0].message.content
        print(text, end="", flush=True)
    else:
        print(f"Error: {response.code}")
        break
```

特点：

- 返回 Generator[DashScopeAPIResponse]
- 每个 chunk 仍然是 DashScopeAPIResponse（需要检查 status_code）
- `incremental_output=True` 控制是否增量
- SSE 解析在 `StreamEventMixin._handle_stream()` 中实现
- 无 `close()` 方法

### 3.4 异步模式

**OpenAI — 完整异步支持**

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def main():
    # 异步非流式
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.choices[0].message.content)

    # 异步流式
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

asyncio.run(main())
```

特点：

- AsyncOpenAI 使用 httpx.AsyncClient
- AsyncStream 实现了 `__aiter__`
- 连接池复用
- 完整的异步上下文管理

**Anthropic — 完整异步支持**

```python
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

async def main():
    # 异步非流式
    message = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(message.content[0].text)

    # 异步流式
    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="")

asyncio.run(main())
```

**DashScope — 异步支持不完整**

```python
# DashScope 的异步是通过 BaseAsyncAioApi 实现的
# 主要用于异步任务（如图像生成），而非 chat 流式

from dashscope.client.base_api import BaseAsyncAioApi

# 异步任务模式（非 chat）
task_response = await SomeModel.async_call(model="...", input={...})
# 轮询等待
result = await BaseAsyncAioApi.wait(task_response)

# ❌ 没有 AsyncGeneration.call()
# ❌ 没有 async for chunk in stream
# ❌ 没有 AsyncStream 类型
```

问题：

- AioGeneration 存在但功能不完整
- 异步流式（async for）无原生支持
- 底层使用 aiohttp，但上层 API 没有暴露完整的异步接口
- 用户需要在 async 函数中调用同步 API（阻塞事件循环）

### 3.5 流式辅助功能

| 功能 | OpenAI | Anthropic | DashScope |
| --- | --- | --- | --- |
| 累积文本 | 手动拼接 delta | `stream.get_final_text()` | 手动拼接 + incremental_output |
| 累积消息 | 手动拼接 chunks | `stream.get_final_message()` | 手动拼接 |
| Usage 统计 | chunk.usage（最后一个 chunk） | message.usage（最终消息） | 最后一个 chunk 的 usage |
| 提前终止 | `stream.close()` | 退出 with 块 | 无 |
| 事件类型区分 | `chunk.choices[0].finish_reason` | event.type（丰富的枚举） | finish_reason |
| 工具调用流式 | delta.tool_calls | RawContentBlockStartEvent | 无流式工具调用 |

### 3.6 底层实现

| 方面 | OpenAI | Anthropic | DashScope |
| --- | --- | --- | --- |
| HTTP 客户端 | httpx.Client / httpx.AsyncClient | httpx.Client / httpx.AsyncClient | requests.Session / aiohttp.ClientSession |
| 连接池 | ✅ httpx 内置 | ✅ httpx 内置 | ❌ 每次 with requests.Session() 新建 |
| SSE 解析 | 自研 Stream 类 | 自研 MessageStream | StreamEventMixin._handle_stream() |
| 重试机制 | 内置（指数退避 + jitter） | 内置（指数退避 + jitter） | 仅 5xx 重试 1 次 |
| 代理支持 | httpx 原生 | httpx 原生 | requests 原生 |
| 超时粒度 | connect / read / write / pool | connect / read / write / pool | 仅 timeout 一个值 |

---

## 4. 架构设计哲学

### 4.1 设计模式对比

| 模式 | OpenAI | Anthropic | DashScope |
| --- | --- | --- | --- |
| 客户端模式 | 实例化 Client → 调用方法 | 实例化 Client → 调用方法 | 类方法直接调用 |
| 响应模式 | Pydantic 模型 | Pydantic 模型 | DictMixin（dict 子类） |
| 错误模式 | 异常驱动 | 异常驱动 | 返回值驱动 |
| 类型系统 | Pydantic v2 + Pydantic Core | Pydantic v1 兼容 | 无类型验证 |
| HTTP 层 | httpx（现代） | httpx（现代） | requests（传统） |
| 代码生成 | 大部分自动生成 | 大部分自动生成 | 手写 |

### 4.2 代码量对比

| 指标 | OpenAI | Anthropic | DashScope |
| --- | --- | --- | --- |
| SDK 包大小 | ~1MB | ~500KB | ~2MB |
| 核心文件数 | ~50 | ~30 | ~80 |
| 依赖项 | httpx, pydantic, anyio, typing-extensions | httpx, pydantic, anyio, typing-extensions | requests, aiohttp, pyyaml |
| 类型存根 | ✅ 内置 .pyi | ✅ 内置 .pyi | ⚠️ 刚添加（PR #170） |
| 自动重试 | ✅ 内置 | ✅ 内置 | ⚠️ 基础实现 |

### 4.3 OpenAI 和 Anthropic 为何选择相同架构

两者 SDK 架构惊人相似，原因：

1. Anthropic 早期 API 兼容 OpenAI — 降低用户迁移成本
2. Pydantic 成为行业标准 — 类型安全 + 自动验证 + 序列化
3. httpx 替代 requests — 原生异步支持 + 连接池
4. 异常驱动错误处理 — Python 社区最佳实践（EAFP > LBYL）

DashScope 选择不同架构的历史原因：

1. 早期设计 — 基于 requests + dict，追求简单
2. 多任务支持 — 异步任务（图像生成、语音合成）需要 async_call / wait / fetch 模式
3. 兼容性考虑 — 保持向后兼容，不敢大改 API

---

## 5. DashScope SDK 改进建议

### 5.1 优先级 P0 — 必须修复

| # | 改进 | 说明 | 参考 |
| --- | --- | --- | --- |
| 1 | 添加 Client 模式 | `DashScope()` 实例化 + `client.chat.completions.create()` | OpenAI `OpenAI()` |
| 2 | 添加异常类层次 | 按 HTTP 状态码自动抛出对应异常 | OpenAI 异常体系 |
| 3 | Pydantic 响应模型 | 替换 DictMixin，提供完整类型验证和 IDE 补全 | OpenAI ChatCompletion |
| 4 | 完善异步流式 | `AsyncDashScope()` + `async for chunk in stream` | OpenAI AsyncOpenAI |

### 5.2 优先级 P1 — 强烈建议

| # | 改进 | 说明 | 参考 |
| --- | --- | --- | --- |
| 5 | httpx 替换 requests | 原生异步 + 连接池复用 + 现代超时控制 | OpenAI / Anthropic |
| 6 | 内置自动重试 | 指数退避 + jitter + Retry-After 支持 | OpenAI `_should_retry()` |
| 7 | 流式辅助 API | `stream.get_final_message()` / `stream.close()` | Anthropic MessageStream |
| 8 | 增量输出默认开启 | `incremental_output=True` 应为默认 | Anthropic 默认增量 |

### 5.3 优先级 P2 — 锦上添花

| # | 改进 | 说明 | 参考 |
| --- | --- | --- | --- |
| 9 | @tool 装饰器 | 从函数签名 + docstring 自动生成 tool schema | OpenAI Agents SDK |
| 10 | Responses API | 兼容 OpenAI Responses API 格式 | OpenAI `client.responses.create()` |
| 11 | Anthropic 兼容层 | 兼容 Anthropic Messages API 格式 | Anthropic `client.messages.create()` |
| 12 | Dashboard 追踪 | 请求追踪和用量统计 Web UI | OpenAI Agents Dashboard |

### 5.4 渐进式迁移方案

DashScope 不能一步到位重构（破坏现有用户），建议分阶段：

**阶段 1：兼容层（不破坏现有 API）**

```python
# 现有 API 保持不变
from dashscope import Generation
response = Generation.call(model="qwen-plus", messages=[...])

# 新增 Client 模式（并行可用）
from dashscope import DashScope
client = DashScope()  # api_key 从环境变量读取
response = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Hello"}],
)
# response 是 Pydantic 模型
print(response.choices[0].message.content)
```

**阶段 2：异常层（可选启用）**

```python
# 默认保持返回值模式（向后兼容）
client = DashScope()
response = client.chat.completions.create(...)
if response.status_code != 200: ...

# 启用异常模式
client = DashScope(raise_on_error=True)
try:
    response = client.chat.completions.create(...)
except RateLimitError as e:
    ...
```

**阶段 3：完全现代化（下一代）**

```python
# dashscope v2
from dashscope import DashScope
from dashscope.errors import RateLimitError, AuthenticationError

client = DashScope()  # httpx + Pydantic + 异常驱动

try:
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.choices[0].message.content)
except RateLimitError as e:
    print(f"Rate limited: {e.message}")
```

### 5.5 关键差异总结

| 维度 | OpenAI/Anthropic | DashScope 现状 | 差距 |
| --- | --- | --- | --- |
| 初始化 | Client() 实例化 | 类方法 + 模块变量 | 缺少 Client 模式 |
| 调用风格 | client.resource.method() | Class.call() | 缺少资源嵌套 |
| 响应 | Pydantic 模型 | DictMixin (dict) | 缺少类型验证 |
| 错误 | 异常驱动 | 返回值驱动 | 缺少异常体系 |
| 异步 | 完整支持 | 部分支持 | 缺少异步流式 |
| HTTP | httpx（连接池） | requests（无连接池） | 性能差距 |
| 重试 | 自动 + 智能 | 基础 1 次 | 可靠性差距 |
| 类型提示 | 完整 .pyi | 刚添加 | IDE 体验差距 |

---

## 6. 日志与错误可理解性

### 6.1 日志系统对比

**OpenAI — 最小化 + 敏感信息过滤**

```python
# 日志配置（_utils/_logs.py）
logger = logging.getLogger("openai")

def setup_logging():
    env = os.environ.get("OPENAI_LOG")
    if env == "debug":
        logger.setLevel(logging.DEBUG)
        httpx_logger.setLevel(logging.DEBUG)  # 同时开启 httpx 日志
```

日志内容示例（DEBUG 模式）：

```
[2026-07-31 14:12:26 - openai._base_client:1033 - DEBUG] Sending HTTP Request: POST https://api.openai.com/v1/chat/completions
[2026-07-31 14:12:27 - openai._base_client:1074 - DEBUG] HTTP Response: POST https://api.openai.com/v1/chat/completions "200 OK" <Headers({...})>
[2026-07-31 14:12:27 - openai._base_client:1082 - DEBUG] request_id: req_abc123
```

亮点：

- SensitiveHeadersFilter — 自动过滤 Authorization 和 api-key 头，防止日志泄露密钥
- 重试日志非常详细："Retrying due to status code 429" / "Retrying as header x-should-retry is set to true"
- 环境变量一键开启（`OPENAI_LOG=debug`），无需代码修改
- 同时输出 httpx 底层日志（连接/超时/重试细节）

**Anthropic — 与 OpenAI 几乎一致**

```python
# _base_client.py
log = logging.getLogger(__name__)  # anthropic._base_client

# 重试日志
log.debug("Retrying due to status code %i", response.status_code)
log.info("Retrying request to %s in %f seconds", options.url, timeout)
log.debug("request_id: %s", response.headers.get("request-id"))
```

亮点：

- 日志格式和粒度与 OpenAI 一致（同一套 Stainless 代码生成器）
- request-id header 追踪
- 超时错误消息自带文档链接："Request timed out or interrupted. See https://docs.anthropic.com/en/api/errors#long-requests"

**DashScope — 基础日志，信息量不足**

```python
# common/logging.py
logger = logging.getLogger("dashscope")

# 日志格式
"%(asctime)s - %(name)s - %(filename)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s"

# 环境变量: DASHSCOPE_LOGGING_LEVEL=info/debug
```

日志内容示例（DEBUG 模式）：

```
2026-07-31 14:12:26 - dashscope - base_api.py - _get - 599 - DEBUG - Starting request: https://dashscope.aliyuncs.com/api/v1/tasks/task_123
2026-07-31 14:12:27 - dashscope - base_api.py - _get - 605 - DEBUG - Starting processing response: https://dashscope.aliyuncs.com/api/v1/tasks/task_123
```

问题：

- ❌ 无敏感信息过滤 — API Key 可能出现在日志中
- ❌ 无 request_id 日志 — 请求 ID 只在 response 对象中，不在日志里
- ❌ 无 HTTP 状态码日志 — 只记录 "Starting request" 和 "Starting processing response"
- ❌ 无重试原因日志 — 重试时只记录 "The task xxx is RUNNING"，不记录重试次数/等待时间
- ❌ 日志格式冗长 — 包含 filename + funcName + lineno，但信息密度低
- ⚠️ 默认关闭 — 不设置 DASHSCOPE_LOGGING_LEVEL 环境变量则无日志

### 6.2 日志对比矩阵

| 特性 | OpenAI | Anthropic | DashScope |
| --- | --- | --- | --- |
| 开启方式 | OPENAI_LOG=debug | ANTHROPIC_LOG=debug | DASHSCOPE_LOGGING_LEVEL=debug |
| 日志级别 | DEBUG / INFO | DEBUG / INFO | DEBUG / INFO |
| 请求 URL | ✅ POST https://... | ✅ POST https://... | ✅ Starting request: https://... |
| HTTP 状态码 | ✅ "200 OK" | ✅ "200 OK" | ❌ 不记录 |
| Request ID | ✅ request_id: req_xxx | ✅ request_id: xxx | ❌ 不记录 |
| Response Headers | ✅ 完整输出 | ✅ 完整输出 | ❌ 不记录 |
| 敏感信息过滤 | ✅ SensitiveHeadersFilter | ✅ 同源 | ❌ 无 |
| 重试原因 | ✅ "Retrying due to 429" | ✅ "Retrying due to 429" | ❌ 不记录 |
| 重试等待时间 | ✅ "Retrying in 2.5s" | ✅ "Retrying in 2.5s" | ❌ 不记录 |
| 超时异常详情 | ✅ exc_info=True | ✅ exc_info=True | ❌ 无 |
| 连接异常详情 | ✅ exc_info=True | ✅ exc_info=True | ❌ 无 |
| httpx 底层日志 | ✅ 联动开启 | ✅ 联动开启 | N/A（用 requests） |

### 6.3 错误消息可理解性

**OpenAI — 结构化错误消息**

```python
# 实际错误消息格式
try:
    client.chat.completions.create(model="gpt-4o", messages=[...])
except openai.RateLimitError as e:
    print(e)
    # Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4o
    # in organization org-xxx on requests per min (RPM): Limit 60, Used 60,
    # Requested 1. Please try again in 1s. Visit ...',
    # 'type': 'requests', 'code': 'rate_limit_exceeded'}}

    # 结构化访问
    e.status_code     # 429
    e.message         # "Rate limit reached..."
    e.code            # "rate_limit_exceeded"
    e.type            # "requests"
    e.request_id      # "req_abc123"
    e.body            # 完整响应体 dict
    e.response        # 完整 httpx.Response 对象
    e.headers         # 响应头（含 Retry-After）
```

亮点：

- 错误消息包含完整上下文：哪个组织、哪个模型、限制是多少、已用多少、建议等待多久
- 所有字段都可以通过属性访问，无需解析字符串
- `e.response.headers.get("Retry-After")` 直接获取重试等待时间

**Anthropic — 同样结构化 + 文档链接**

```python
try:
    client.messages.create(model="claude-sonnet-4-20250514", messages=[...])
except anthropic.RateLimitError as e:
    print(e)
    # Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error',
    # 'message': 'Number of request tokens has exceeded your per-minute rate limit'}}

    e.status_code     # 429
    e.message         # "Number of request tokens..."
    e.request_id      # "req_xxx"
    e.type            # "rate_limit_error"（Anthropic 特有的错误类型枚举）
    e.body            # 完整响应体
    e.response        # 完整 httpx.Response

# 超时错误自带文档链接
except anthropic.APITimeoutError as e:
    print(e.message)
    # "Request timed out or interrupted. This could be due to a network timeout,
    # dropped connection, or request cancellation.
    # See https://docs.anthropic.com/en/api/errors#long-requests for more details."
```

亮点：

- `e.type` — Anthropic 特有的 ErrorType 枚举（overloaded_error / rate_limit_error / invalid_request_error 等），比 HTTP 状态码更精确
- 超时错误消息包含排查文档链接
- 额外异常类型：OverloadedError(529) / DeadlineExceededError(504) / ServiceUnavailableError(503)

**DashScope — 字符串拼接，信息碎片化**

```python
response = Generation.call(model="qwen-plus", messages=[...])

# 成功时
print(response)
# {"status_code": 200, "request_id": "req-xxx", "code": "", "message": "",
#  "output": {...}, "usage": {...}}

# 失败时
if response.status_code != 200:
    print(f"code: {response.code}")       # "Throttling"
    print(f"message: {response.message}")  # "Requests rate limit"
    print(f"request_id: {response.request_id}")  # "req-xxx"

    # ❌ 没有结构化的错误对象
    # ❌ 没有 Retry-After 建议
    # ❌ 没有文档链接
    # ❌ 没有错误分类（限流 vs 认证 vs 参数错误 需要自己判断）
```

问题：

- 错误消息过于简短："Requests rate limit" — 不告诉用户限制是多少、已用多少、建议等多久
- 错误码是裸字符串："Throttling" / "InvalidApiKey" / "DataInspectionFailed" — 无类型安全
- 同一个 DashScopeAPIResponse 对象既表示成功又表示失败 — 容易忘记检查
- response.code 可能是 None、空字符串 ""、或错误码字符串 — 三种状态混乱

### 6.4 实际排障场景对比

**场景 1：API Key 无效**

OpenAI：

```
AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided:
sk-xxxx...xxxx. You can find your API key at https://platform.openai.com/account/api-keys.',
'type': 'invalid_request_error', 'code': 'invalid_api_key'}}
```

→ 直接告诉用户去哪里找正确的 Key

DashScope：

```
status_code: 401
code: "InvalidApiKey"
message: "Invalid API-key provided."
```

→ 只说无效，不说去哪里获取

**场景 2：限流**

OpenAI：

```
RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4o
in organization org-xxx on requests per min (RPM): Limit 60, Used 60, Requested 1.
Please try again in 1s. Visit https://platform.openai.com/account/rate-limits for more.',
'type': 'requests', 'code': 'rate_limit_exceeded'}}
```

→ 告诉你限制值、已用量、建议等待时间、文档链接

DashScope：

```
status_code: 429
code: "Throttling"
message: "Requests rate limit"
```

→ 4 个单词，没有任何可操作信息

**场景 3：内容安全审核**

DashScope：

```
status_code: 400
code: "DataInspectionFailed"
message: "Input data may contain inappropriate content."
```

→ 不告诉用户哪段内容触发了审核

**场景 4：网络超时**

OpenAI：

```
APITimeoutError: Request timed out.
```

→ 简洁明确，可通过 `except APITimeoutError` 精确捕获

DashScope：

```
requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='dashscope.aliyuncs.com',
port=443): Read timed out. (read timeout=300)
```

→ 底层 requests 异常直接泄露，用户需要 `except requests.exceptions.ReadTimeout` 捕获

### 6.5 异常可追溯性

| 特性 | OpenAI | Anthropic | DashScope |
| --- | --- | --- | --- |
| 异常链 | `raise ... from err` 保留原始异常 | 同 | ❌ 不保留（直接返回 response） |
| 原始请求 | e.request (httpx.Request) | e.request (httpx.Request) | ❌ 不保留 |
| 原始响应 | e.response (httpx.Response) | e.response (httpx.Response) | ❌ 不保留 |
| 请求体 | e.request.content | e.request.content | ❌ 不保留 |
| 响应体 | e.body | e.body | response 本身就是 |
| 堆栈追踪 | ✅ 完整 Python traceback | ✅ 完整 | ❌ 无异常，无 traceback |
| 可序列化 | e.body 是 dict | e.body 是 dict | response 是 dict |

### 6.6 调试体验对比

开启 DEBUG 日志后的输出

**OpenAI（信息密度高）**：

```
[2026-07-31 14:12:26 - openai._base_client:1033 - DEBUG] Sending HTTP Request: POST https://api.openai.com/v1/chat/completions
[2026-07-31 14:12:27 - openai._base_client:1074 - DEBUG] HTTP Response: POST https://api.openai.com/v1/chat/completions "200 OK" Headers({'content-type': 'application/json', 'x-request-id': 'req_abc123', ...})
[2026-07-31 14:12:27 - openai._base_client:1082 - DEBUG] request_id: req_abc123
```

**DashScope（信息密度低）**：

```
2026-07-31 14:12:26 - dashscope - api_request_factory.py - _build_api_request - 128 - DEBUG - encryption enabled
2026-07-31 14:12:26 - dashscope - base_api.py - call - 548 - DEBUG - Starting request: https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation/
2026-07-31 14:12:27 - dashscope - base_api.py - call - 558 - DEBUG - Starting processing response: https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation/
```

→ 不知道响应状态码、不知道 request_id、不知道响应内容

### 6.7 DashScope 日志改进建议

| # | 改进 | 优先级 | 说明 |
| --- | --- | --- | --- |
| 1 | 添加 SensitiveHeadersFilter | P0 | 过滤 Authorization header，防止密钥泄露 |
| 2 | 日志中记录 request_id | P0 | 每个请求/响应都记录 request_id，便于找阿里云工单 |
| 3 | 日志中记录 HTTP 状态码 | P0 | "HTTP Response: POST ... 200 OK" |
| 4 | 重试日志增加原因和等待时间 | P1 | "Retrying due to 429, waiting 2.5s" |
| 5 | 异常链保留 | P1 | `raise DashScopeError(...) from original_error` |
| 6 | 错误消息增加可操作信息 | P1 | 限流时告知限制值和等待时间 |
| 7 | 超时错误消息添加排查文档链接 | P2 | 参考 Anthropic 的做法 |
| 8 | 日志格式简化 | P2 | 去掉 filename + funcName + lineno（用 %(name)s 即可） |

### 6.8 错误处理最佳实践建议

**当前 DashScope 用户被迫写的代码**

```python
# 每个调用点都要写这套 — 容易遗漏，容易不一致
response = Generation.call(model="qwen-plus", messages=[...])
if response.status_code == 200:
    result = response.output.choices[0].message.content
elif response.code == "Throttling":
    time.sleep(5)
    # 重试...
elif response.code == "InvalidApiKey":
    print("Check your DASHSCOPE_API_KEY")
elif response.code == "DataInspectionFailed":
    print("Content blocked by safety filter")
else:
    print(f"Unknown error: {response.status_code} {response.code}: {response.message}")
```

**OpenAI/Anthropic 用户写的代码**

```python
# 一处 try/except，全局生效 — 不可能遗漏
try:
    response = client.chat.completions.create(model="gpt-4o", messages=[...])
    result = response.choices[0].message.content  # 无需检查 status_code
except RateLimitError:
    time.sleep(int(e.headers.get("Retry-After", 5)))
except AuthenticationError:
    print("Check your API key")
```

**如果 DashScope 采用异常驱动后的理想代码**

```python
from dashscope import DashScope
from dashscope.errors import RateLimitError, AuthenticationError, ContentFilterError

client = DashScope()

try:
    response = client.chat.completions.create(model="qwen-plus", messages=[...])
    result = response.choices[0].message.content
except RateLimitError as e:
    print(f"Rate limited. Limit: {e.limit}, Used: {e.used}, Retry after: {e.retry_after}s")
    time.sleep(e.retry_after)
except AuthenticationError:
    print(f"Invalid API key. Get yours at https://dashscope.console.aliyun.com/apiKey")
except ContentFilterError as e:
    print(f"Content blocked: {e.category}")
```

---

## 参考

- OpenAI Python SDK
- Anthropic Python SDK
- DashScope Python SDK
- Pydantic V2
- httpx
