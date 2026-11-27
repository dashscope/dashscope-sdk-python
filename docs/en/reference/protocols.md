---
description: "API reference for native, OpenAI-compatible and Anthropic-compatible protocol adapters."
---
# Protocol Adapters

`dashscope.protocols` translates chat requests and responses between wire
protocols, so the same `client.chat.completions` call can speak the DashScope
native protocol, the OpenAI-compatible protocol or the Anthropic-compatible
protocol. Select one with `Client(protocol="openai")` (or the `protocol`
setting / `DASHSCOPE_PROTOCOL` environment variable); valid values are
`native`, `openai` and `anthropic`.

!!! tip "Precedence"
    An explicit `Client(protocol=...)` argument wins over the `protocol`
    setting, which wins over `DASHSCOPE_PROTOCOL`.

## Base contract

::: dashscope.protocols.base.ChatProtocol

::: dashscope.protocols.base.ChatRequest

## Adapters

::: dashscope.protocols.native.NativeProtocol

::: dashscope.protocols.openai.OpenAIProtocol

::: dashscope.protocols.anthropic.AnthropicProtocol

::: dashscope.protocols.anthropic.detect_web_search_tool

::: dashscope.protocols.anthropic.system_has_billing_header

::: dashscope.protocols.anthropic.ensure_billing_header

## Registry

::: dashscope.protocols.get_protocol
