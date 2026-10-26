# Anthropic Python SDK vs DashScope SDK 对比分析

> 基于 Anthropic Python SDK 文档 和 DashScope SDK 源码的深度对比
> 生成时间：2026-07-31
> 来源：钉钉文档（alidocs YMyQA2dXW7gYo6MzcZR0MPdZWzlwrZgb）

---

## 1. 初始化方式

| 维度 | Anthropic | DashScope |
| --- | --- | --- |
| API Key | `Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))` 或自动读取环境变量 | `dashscope.api_key = "..."` 全局变量，或每次调用传入 |
| Client 实例 | `client = Anthropic()` — 面向对象，创建实例 | `Generation.call(...)` — 全静态类方法，无需实例化 |
| Base URL | `client = Anthropic(base_url="...")` | `dashscope.base_http_api_url = "..."` 全局变量 |
| 异步 Client | `client = AsyncAnthropic()` — 独立的异步类 | `AioGeneration` 等独立的异步类 |

**评价**：Anthropic 的 Client 实例模式明显更优 — 可以创建多个不同配置的 client（不同 key/不同 region），支持依赖注入自定义 HTTP client（httpx/aiohttp）。DashScope 的全局变量方式在多租户场景下容易出错。

---

## 2. 最简单的调用（Hello World）

**Anthropic**

```python
from anthropic import Anthropic

client = Anthropic()
message = client.messages.create(
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ],
    model="claude-opus-5",
)

for block in message.content:
    if block.type == "text":
        print(block.text)
```

**DashScope**

```python
import dashscope

response = dashscope.Generation.call(
    model="qwen-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)

print(response.output.choices[0].message.content)
```

**评价**：

- Anthropic 需要遍历 `message.content` blocks，稍显繁琐，但支持多模态输出（图片、工具调用等）
- DashScope 路径更深：`output.choices[0].message.content` 四层穿透
- 两者都不如 OpenAI 的 `response.output_text` 一行直达

---

## 3. 流式输出

**Anthropic**

```python
# 方式一：基础流式
stream = client.messages.create(
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
    model="claude-opus-5",
    stream=True,
)
for event in stream:
    print(event.type)

# 方式二：高级流式助手（推荐）
async with client.messages.stream(
    max_tokens=1024,
    messages=[{"role": "user", "content": "Say hello"}],
    model="claude-opus-5",
) as stream:
    async for text in stream.text_stream:
        print(text, end="", flush=True)
    message = await stream.get_final_message()
```

**DashScope**

```python
responses = Generation.call(
    model="qwen-turbo",
    messages=[...],
    stream=True,
    incremental_output=True,  # 需要手动设置
)
for response in responses:
    print(response.output.choices[0].message.content, end="")
```

**评价**：Anthropic 的流式体验明显更好：

- 提供 `stream.text_stream` 直接输出文本（无需解析 event type）
- `get_final_message()` 获取完整消息
- 支持 context manager 管理资源
- DashScope 需要额外理解 `incremental_output` 参数，且路径深

---

## 4. Function Calling / Tool Use

**Anthropic**

```python
from anthropic import Anthropic, beta_tool

client = Anthropic()

@beta_tool
def get_weather(location: str) -> str:
    """Get the weather for a given location.

    Args:
        location: The city and state, for example, San Francisco, CA

    Returns:
        A JSON-encoded string with location, temperature, and weather condition.
    """
    return json.dumps({"location": location, "temperature": "68°F", "condition": "Sunny"})

# 使用 tool_runner 自动处理工具调用
runner = client.beta.messages.tool_runner(
    max_tokens=1024,
    model="claude-opus-5",
    tools=[get_weather],
    messages=[{"role": "user", "content": "What is the weather in SF?"}],
)

for message in runner:
    print(message)
```

**DashScope**

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state"}
            },
            "required": ["location"]
        }
    }
}]

response = Generation.call(
    model="qwen-turbo",
    messages=[{"role": "user", "content": "SF weather?"}],
    tools=tools,
)

# 需要手动解析 tool_calls 并调用函数
tool_calls = response.output.choices[0].message.tool_calls
```

**评价**：Anthropic 大幅领先：

- `@beta_tool` 装饰器：从函数签名和 docstring 自动生成 schema
- `tool_runner`：自动处理工具调用循环（调用 → 返回结果 → 再次调用模型）
- DashScope 需要手动定义 JSON schema，手动解析 tool_calls，手动循环调用

---

## 5. 异步任务（图像生成等）

**Anthropic**：没有暴露异步任务概念，调用就是同步/流式。

**DashScope**

```python
# 提交任务
resp = ImageSynthesis.async_call(model="wanx-v1", prompt="...")
# 轮询状态
while True:
    result = ImageSynthesis.fetch(resp.output.task_id)
    if result.output.task_status == "SUCCEEDED":
        break
```

**评价**：DashScope 的异步任务模式增加了使用复杂度，但这是服务端架构决定的（图像生成耗时长），属于合理设计。Anthropic 主要做文本生成，不需要此类功能。

---

## 6. 错误处理

**Anthropic**

```python
import anthropic

try:
    message = client.messages.create(...)
except anthropic.APIConnectionError as e:
    print("The server could not be reached")
except anthropic.RateLimitError as e:
    print("A 429 status code was received; we should back off a bit.")
except anthropic.APIStatusError as e:
    print(f"Status code: {e.status_code}")
```

提供完整的异常类型层级：

- BadRequestError (400)
- AuthenticationError (401)
- PermissionDeniedError (403)
- NotFoundError (404)
- ConflictError (409)
- UnprocessableEntityError (422)
- RateLimitError (429)
- InternalServerError (5xx)
- APIConnectionError (网络错误)

**DashScope**

```python
response = Generation.call(...)
if response.status_code != 200:
    print(f"Error: {response.code}, {response.message}")
    print(f"Request ID: {response.request_id}")
```

**评价**：Anthropic 明显更优：

- 异常驱动的错误处理，符合 Python 惯例
- 细粒度的异常类型，可以针对不同错误做不同处理
- DashScope 使用返回值检查（status_code + code + message），容易遗漏错误处理

---

## 7. 高级功能

**Anthropic**

| 功能 | 说明 |
| --- | --- |
| Token 计数 | `client.messages.count_tokens(...)` 请求前预计算 token 数 |
| Message Batches | 批量处理 API，支持异步大批量请求 |
| File Uploads | 支持 PathLike、bytes tuple、BinaryIO 等多种上传方式 |
| Request ID | `message._request_id` 便于调试和报告 |
| 自动重试 | 408/409/429/5xx 自动重试 2 次，指数退避 |
| 多平台支持 | Bedrock、Vertex、AWS、Microsoft Foundry 多平台 Client |
| HTTP Client 定制 | 支持 httpx/aiohttp 自定义 HTTP client |

**DashScope**

| 功能 | 说明 |
| --- | --- |
| 丰富的模型类型 | 文本/图像/视频/语音/Embedding/Rerank 全覆盖 |
| 异步任务 | async_call + fetch 模式对长耗时任务实用 |
| 多模态格式 | `{"text": ..., "image": ...}` 直观简洁 |
| Session 管理 | 支持 WebSocket、SSE 等多种连接方式 |

**评价**：Anthropic 在工程化程度上领先（token 计数、批量 API、自动重试、多平台），DashScope 在模型多样性上有优势。

---

## 总结：DashScope 可改进的方向

### 🔴 高优先级

| # | 改进点 | 现状 | 建议 |
| --- | --- | --- | --- |
| 1 | 异常处理 | 返回值检查 `if response.status_code != 200` | 添加异常驱动错误处理：APIError、RateLimitError、AuthenticationError 等 |
| 2 | Client 实例化 | 全局变量 `dashscope.api_key` | 支持 `DashScope(api_key=..., base_url=...)` 实例，多租户友好 |
| 3 | 流式体验 | 需要 `incremental_output` 参数 + 深层路径 | 提供 `client.messages.stream()` 高级 API，直接输出文本 |

### 🟡 中优先级

| # | 改进点 | 现状 | 建议 |
| --- | --- | --- | --- |
| 4 | Tool Use 简化 | 手动定义 JSON schema + 手动循环调用 | 添加 `@tool` 装饰器 + tool_runner 自动循环 |
| 5 | 自动重试 | 无内置重试机制 | 添加可配置的自动重试（429/5xx 等） |
| 6 | Token 计数 | 无预计算功能 | 添加 `Generation.count_tokens(...)` API |
| 7 | HTTP Client 定制 | 内部封装 httpx | 支持传入自定义 HTTP client（httpx/aiohttp） |

### 🟢 低优先级（DashScope 做得好的地方）

| # | 亮点 | 说明 |
| --- | --- | --- |
| 1 | 多模态 content 格式 | `{"text": ..., "image": ...}` 比 Anthropic 的 block 结构更直观 |
| 2 | 内置异步任务 | async_call + fetch 模式对长耗时任务（图像/视频）很实用 |
| 3 | 丰富的模型类型 | 文本/图像/视频/语音/Embedding/Rerank 全覆盖，Anthropic 主要是文本 |

---

## 参考资源

- Anthropic Python SDK
- OpenAI Python SDK
- DashScope SDK 源码：dashscope-sdk-python-part/
