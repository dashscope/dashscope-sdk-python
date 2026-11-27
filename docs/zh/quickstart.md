---
description: "安装 dashscope、配置 API Key，几分钟内完成第一次对话补全。"
---
# 快速开始

dashscope API 以带类型标注的 [Client](reference/client.md)（及其异步版本
`AsyncClient`）为核心，通过 `client.chat`、`client.images`、`client.audio`
等资源命名空间组织各能力。本页带你从安装走到第一次对话补全。

## 安装

需要 Python 3.9 及以上版本。

```bash
python -m pip install -U dashscope
```

## 配置 API Key

推荐通过环境变量 `DASHSCOPE_API_KEY` 提供密钥——不要把密钥硬编码进源码：

```bash
export DASHSCOPE_API_KEY="<your-api-key>"
```

也可以调用 `dashscope.save_api_key("sk-xxx")` 持久化保存密钥
（写入 `~/.dashscope/api_key`），或在运行时通过 `dashscope.settings.api_key`
赋值。

配置按以下顺序解析，先命中者生效：

1. `Client(...)` 显式参数
2. 进程级 `dashscope.settings`
3. 环境变量（`DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL` 等）
4. `config.toml`
5. API Key 文件

!!! tip "CI 与容器"
    在 CI 中把 `DASHSCOPE_API_KEY` 作为秘密环境变量注入；
    `save_api_key` 仅适用于交互式本地机器。

## 一行调用对话

脚本和 notebook 场景下，`dashscope.chat` 会自动管理客户端：

```python
import dashscope

resp = dashscope.chat("qwen-plus", "讲个笑话")
print(resp.output_text)
```

额外的关键字参数会透传给 `client.chat.completions.create`。

## 使用 Client

```python
import dashscope

client = dashscope.Client()  # 读取 DASHSCOPE_API_KEY
resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "你好！"}],
)
print(resp.output_text)
```

`Client` 持有 HTTP 连接池，用完后请关闭，或使用上下文管理器：

```python
with dashscope.Client() as client:
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "你好！"}],
    )
```

### 流式输出

传入 `stream=True` 并迭代增量块：

```python
with dashscope.Client() as client:
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "写一首关于大海的俳句。"}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### 异步调用

```python
import asyncio
import dashscope


async def main():
    async with dashscope.AsyncClient() as client:
        resp = await client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "你好！"}],
        )
        print(resp.output_text)


asyncio.run(main())
```

??? example "完整可运行示例：chat.py（自动同步）"
    ```python
    --8<-- "examples/01_atomic_capabilities/text_generation/chat.py"
    ```

## 错误处理

调用失败（连接失败、HTTP 错误与服务端业务错误）会抛出
[DashScopeError][dashscope.errors.DashScopeError] 的子类，
携带 `code`、`status_code`、`request_id` 和响应 `body`：

```python
from dashscope import Client, DashScopeError

try:
    with Client() as client:
        client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "你好"}],
        )
except DashScopeError as exc:
    print(exc.code, exc.status_code, exc.request_id, exc.message)
```

!!! note "两个边界"
    - 本地配置与参数校验可能抛出 `ValueError`（如非法 region、
      空 `file_id`），不属于 `DashScopeError`。
    - 流式调用分两个阶段：建流失败按上述规则抛出 `DashScopeError`；
      建流成功后，迭代中断流等底层传输异常（如 `httpx.ReadTimeout`）
      会原样传播，SDK 不会自动重连，需要调用方自行捕获处理。

完整的异常层级（限流、鉴权、超时、WebSocket 错误等）见 [错误（Errors）](reference/errors.md)。

## 下一步

- [客户端与配置](reference/client.md) —— `Client` / `AsyncClient` 构造参数、
  `Config` / `Settings`、地域切换。
- [对话（Chat）](reference/chat.md) —— 对话补全、流式输出与响应模型。
- [语音（Audio）](reference/audio.md) —— TTS、ASR、实时与 Qwen Live 会话；完整接入流程见
  [Qwen Live 指南](guides/qwen-live-python-sdk.md)。
- [资源（Resources）](reference/resources.md) —— 向量嵌入、重排序、图像、视频、文件、模型、
  微调、部署、应用等。
- [类型（Types）](reference/types.md) —— 共享的消息 / 响应 / 分页模型。