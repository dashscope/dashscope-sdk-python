---
description: "阿里云百炼（DashScope）官方 Python SDK：类型化同步/异步客户端、示例与特征总览。"
---
# DashScope Python SDK

`dashscope` SDK 是
[阿里云百炼（Model Studio）](https://help.aliyun.com/zh/model-studio/)
的官方 Python 客户端：对话补全、多模态理解、向量嵌入、重排序、
图像与视频生成、语音合成与识别、实时语音、微调、智能体强化学习、
托管智能体与应用调用。

2.0 是类型化、资源式客户端——`dashscope.Client()`——参照 OpenAI SDK
的人体工学：一个客户端持有共享连接池，每种能力以类型化资源挂在
其上，响应是类型化对象，失败默认抛出异常。

!!! tip "从哪里开始"
    初次使用？跟随[快速开始](quickstart.md)。迁移既有 1.x 代码？
    见[从 1.x 迁移](migration.md)。

## 特征

- **类型化资源** —— `client.chat`、`client.images`、`client.audio`……
- **同步与异步** —— `AsyncClient` 镜像同步接口。
- **三种协议** —— 一个客户端支持 `native`、`openai`、`anthropic`。
- **像读文本一样的流式** —— `.text_stream`、`.reasoning_stream`。
- **类型化错误 + 自诊断**，以及离线 **dev mock**。
- **真正的命令行**与可选交互式 AI 助手。

完整列表见[特征](features.md)。

## 要求

- Python 3.9+（可选交互式助手需要 3.10+）。
- 百炼 API Key——在[百炼控制台](https://bailian.console.aliyun.com/)
  创建后导出：

```shell
export DASHSCOPE_API_KEY="sk-..."
```

## 安装

```shell
pip install -U dashscope
```

基础安装覆盖 SDK API 调用、`dashscope` CLI、Agentic RL 微调与本地
分词器。可选 extras 用于配置交互式 AI 助手，例如
`pip install "dashscope[acli]"`（见[命令行工具](cli.md)）。

## 例子

### 创建它

```python
import dashscope

client = dashscope.Client()  # 读取 DASHSCOPE_API_KEY
```

### 运行它

```python
resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "讲个笑话"}],
)
print(resp.output_text)
print(resp.usage.total_tokens)
```

或一行式：`dashscope.chat("qwen-plus", "讲个笑话")`。

### 流式运行它

```python
with client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "写一首关于海的俳句"}],
    stream=True,
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final = stream.get_final_message()
```

### 异步运行它

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

### 和它对话（实时语音）

`client.audio.live` 通过 WebSocket 运行全双工语音会话——完整接入
流程见 [Qwen Live 指南](guides/qwen-live-python-sdk.md)。

## 命令行（CLI 2.0）

安装 SDK 的同时会安装 `dashscope` 命令。2.0 CLI 与 SDK 资源一一对应，
`Client` 能做的任何事情都可以直接在 shell 里脚本化：

```bash
dashscope version                 # 打印 SDK 版本
dashscope auth login              # 保存 API Key
dashscope chat create --model qwen-plus --prompt "你好！"
dashscope -o json ft list         # 机器可读输出，便于脚本处理
```

覆盖对话、多模态、向量嵌入、图像/视频、音频、文件、微调、智能体
强化学习等能力，并提供协议、区域与输出格式等全局选项。不带任何
参数运行 `dashscope` 会启动可选的交互式 AI 助手
（`pip install "dashscope[acli]"`）。冻结的 1.x 命令树仍以
`dashscope1` 提供。详见[命令行工具](cli.md)。

## 协议

同一个调用可以使用三种传输协议：

```python
client = dashscope.Client(protocol="openai")  # 或 "native"、"anthropic"
```

见[协议适配器](reference/protocols.md)。

## 错误与诊断

失败抛出类型化 `DashScopeError` 子类，携带 `code`、`request_id` 与
响应体；可选的本地诊断在 `error.diagnosis` 上附加结构化修复建议。
见[错误（Errors）](reference/errors.md)与[开发工具](reference/devkit.md)。

## 旧版 1.x

完整 1.x 接口仍可从历史导入路径使用
（`from dashscope import Generation`），由冻结门面实现。
见[旧版 1.x](reference/legacy.md)。

## 依赖关系

基础安装引入 `requests`、`aiohttp`、`httpx` + `httpx-sse`、
`websocket-client`、`pydantic`（v2）、`tenacity`、`tiktoken`，以及
CLI 所需的 `typer` + `rich`。可选助手 extras 见上文「安装」一节。

## 执照

本项目采用
[Apache License 2.0](https://github.com/dashscope/dashscope-sdk-python/blob/main/LICENSE)
授权。