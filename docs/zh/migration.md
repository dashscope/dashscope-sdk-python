---
description: "逐项对照 1.x 与 2.0，把既有代码迁移到资源式 dashscope 客户端。"
---
# 从 1.x 迁移

2.0 客户端取代了 1.x 的模块级静态类，但完整 1.x 接口仍可从历史导入
路径继续使用。可以按自己的节奏迁移——本页把旧习惯映射到新写法。

## 逐项对照

| | 1.x | 2.0 |
|---|---|---|
| 入口 | 模块级静态类：`Generation.call(...)` | 资源式客户端：`client.chat.completions.create(...)` |
| 一行式 | — | `dashscope.chat("qwen-plus", "讲个笑话")` |
| 响应 | 原始 dict 式；手动检查 `status_code == HTTPStatus.OK` | 类型化对象；读 `resp.output_text`、`resp.usage.total_tokens` |
| 错误 | 手动查看 `resp.code` / `resp.message` | 默认抛出类型化 `DashScopeError` 异常 |
| 流式 | 原始帧生成器 | `ChatStream` 提供 `.text_stream`、`.reasoning_stream`、`.get_final_message()` |
| 协议 | DashScope 原生 + 独立的兼容模式入口 | 一个客户端通过 `protocol=` 使用 `native`、`openai` 或 `anthropic` |
| 异步 | 独立的 `Aio*` 类 | `dashscope.AsyncClient()` 镜像同步接口 |
| 连接 | 每次调用新建会话 | 每客户端一个 keep-alive 连接池，跨资源共享 |
| 配置 | 模块全局变量（`dashscope.api_key`、`set_region`） | 分层：参数 → `dashscope.settings` → 环境变量 → `~/.dashscope/config.toml` |

## 调用映射

| 1.x 类 | 2.0 资源 |
| --- | --- |
| `Generation` / `AioGeneration` | `client.chat.completions` |
| `MultiModalConversation` | `client.multimodal` |
| `ImageSynthesis` / `ImageGeneration` | `client.images` |
| `VideoSynthesis` | `client.videos` |
| `TextEmbedding` / `BatchTextEmbedding` | `client.embeddings` |
| `MultiModalEmbedding` | `client.multimodal`（embeddings） |
| `TextReRank` | `client.rerank` |
| `SpeechSynthesizer`（tts / tts_v2） | `client.audio.tts` |
| `Recognition` / `Transcription` | `client.audio.asr` |
| `Files` | `client.files` |
| `Models` | `client.models` |
| `FineTunes` | `client.fine_tunes` |
| `Deployments` | `client.deployments` |
| `Application` | `client.apps` |

所有 1.x 类仍完整记录在[旧版 1.x](reference/legacy.md) 下。

## 迁移前后

1.x —— 调用后手动检查状态：

```python
from http import HTTPStatus
from dashscope import Generation

resp = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "你好！"}],
)
if resp.status_code != HTTPStatus.OK:
    print(resp.code, resp.message)
else:
    print(resp.output.text)
```

2.0 —— 类型化响应，失败抛异常：

```python
from dashscope import Client, DashScopeError

try:
    with Client() as client:
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "你好！"}],
        )
    print(resp.output_text)
except DashScopeError as exc:
    print(exc.code, exc.message)
```

## 保持不变的部分

- 导入路径：`from dashscope import Generation` 仍可用（冻结门面）。
- `DASHSCOPE_API_KEY` 环境变量与控制台签发的密钥。
- 模型名与请求体——native 协议下线上契约不变。

!!! tip "渐进迁移"
    两套接口可在同一进程中共存。逐个迁移调用点即可，没有统一的
    切换截止日。各 2.x 版本的变更见[发行说明](release-notes.md)。