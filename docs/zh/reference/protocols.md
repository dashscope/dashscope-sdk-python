---
description: "native、OpenAI 兼容与 Anthropic 兼容协议适配器 API 参考。"
---
# 协议适配器（Protocols）

`dashscope.protocols` 在不同传输协议之间转换对话请求与响应，
同一个 `client.chat.completions` 调用可以使用 DashScope 原生协议、
OpenAI 兼容协议或 Anthropic 兼容协议。通过 `Client(protocol="openai")`
（或 `protocol` 配置项 / `DASHSCOPE_PROTOCOL` 环境变量）选择；
合法值为 `native`、`openai`、`anthropic`。

!!! tip "优先级"
    显式 `Client(protocol=...)` 参数优先于 `protocol` 配置项，
    后者优先于 `DASHSCOPE_PROTOCOL` 环境变量。

## 基础契约

::: dashscope.protocols.base.ChatProtocol

::: dashscope.protocols.base.ChatRequest

## 适配器

::: dashscope.protocols.native.NativeProtocol

::: dashscope.protocols.openai.OpenAIProtocol

::: dashscope.protocols.anthropic.AnthropicProtocol

## 注册表

::: dashscope.protocols.get_protocol
