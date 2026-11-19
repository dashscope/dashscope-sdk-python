---
description: "对话补全 API 参考：同步/异步资源与流式类。"
---
# 对话（Chat）

`client.chat.completions` 用于创建对话补全，支持流式与非流式。
响应模型见[类型（Types）](types.md)。

## 资源类

::: dashscope.resources.chat.Chat

::: dashscope.resources.chat.AsyncChat

::: dashscope.resources.chat.completions.Completions

::: dashscope.resources.chat.completions.AsyncCompletions

::: dashscope.resources.chat.completions.ChatStream

::: dashscope.resources.chat.completions.AsyncChatStream

## 另请参阅

- [ChatCompletion][dashscope.types.completions.ChatCompletion] /
  [ChatCompletionChunk][dashscope.types.completions.ChatCompletionChunk] —— 响应模型。
- [类型（Types）](types.md) —— 消息、选项、增量与用量。