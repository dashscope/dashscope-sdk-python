---
description: "dashscope 2.0 SDK 特征总览：类型化资源、同步/异步、协议、流式、错误、dev mock、CLI 与 1.x 兼容。"
---
# 特征

dashscope 2.0 SDK 是面向阿里云百炼（Model Studio / DashScope）的
类型化、资源式客户端。亮点如下：

## 类型化资源客户端

- 一个 `Client()` 持有 keep-alive 连接池，供所有资源共享。
- 能力以类型化命名空间挂在客户端上：`client.chat`、`client.images`、
  `client.audio`、`client.files`、`client.fine_tunes`……
- 响应是类型化对象：直接读 `resp.output_text`、`resp.usage.total_tokens`，
  不再翻原始 dict。
- 失败默认抛出类型化异常——无需手动检查 `status_code`。

见[客户端与配置](reference/client.md)。

## 同步与异步，同一套接口

- `dashscope.AsyncClient()` 逐方法镜像同步客户端。
- 共享的分层配置：参数 → `dashscope.settings` → 环境变量 →
  `~/.dashscope/config.toml`。
- 例外：双工语音 WebSocket 会话仅支持同步。

## 一个客户端，三种协议

- 同一个 `client.chat.completions` 调用可使用 DashScope `native`、
  OpenAI 兼容或 Anthropic 兼容传输协议。
- 通过 `Client(protocol="openai")`、`protocol` 配置项或
  `DASHSCOPE_PROTOCOL` 选择。

见[协议适配器](reference/protocols.md)。

## 像读文本一样的流式

- `ChatStream` 提供 `.text_stream`、`.reasoning_stream` 与
  `.get_final_message()`——迭代纯字符串，结束时拿到组装好的类型化消息。
- SSE 传输，自动重试与重连处理。

见[对话（Chat）](reference/chat.md)。

## WebSocket 实时语音

- `client.audio` 下的语音合成（TTS）、识别（ASR）、omni 实时与
  Qwen Live 会话。
- 音色与热词定制资源。

见[语音（Audio）](reference/audio.md)与
[Qwen Live 指南](guides/qwen-live-python-sdk.md)。

## 类型化错误与自诊断

- `DashScopeError` 异常体系覆盖传输、校验、HTTP 状态与 WebSocket 失败，
  附带 `code`、`request_id` 与响应体。
- 可选的本地诊断把失败转化为结构化双语修复建议，见 `error.diagnosis`。

见[错误（Errors）](reference/errors.md)与[开发工具](reference/devkit.md)。

## 用 dev mock 离线开发

- `static` / `replay` 模式离线走通 HTTP 请求路径，不消耗云端 token。
- `record` 会发送真实请求，可能产生费用，并保存响应供后续离线回放。
- 关闭时零开销；成功路径上不导入任何相关代码。

见[开发工具](reference/devkit.md)。

## 托管智能体与智能体强化学习

- `dashscope.applications.agentstudio`：独立子 SDK，覆盖托管智能体、
  会话、环境、部署、记忆、技能与密钥库。
- `dashscope.applications.finetune.AgenticRL` 加装饰器，
  本地编写 rollout / reward 函数。

见 [AgentStudio](reference/agentstudio.md)与
[强化学习微调](reference/agentic-rl.md)。

## 真正的命令行

- `dashscope` CLI 覆盖对话、语音、图像、文件、微调、部署、RL、应用等，
  全局选项一致，支持机器可读输出（`-o json`）。
- 可选交互式 AI 助手：`pip install "dashscope[acli]"`。

见[命令行工具](cli.md)。

## 1.x 兼容，已冻结

- 完整 1.x 公共接口仍可从历史路径导入
  （`from dashscope import Generation`），由 `dashscope.legacy_1x` 下
  的冻结门面实现。

见[旧版 1.x](reference/legacy.md)。
