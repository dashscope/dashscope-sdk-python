---
description: "共享 pydantic 模型 API 参考：消息、补全、分页与辅助函数。"
---
# 类型（Types）

消息、补全、响应与分页的共享 pydantic 模型，以及消息规范化辅助函数。
大部分名称可从 `dashscope.types` 直接导入。

## 消息与内容片段

::: dashscope.types.messages.Message

::: dashscope.types.messages.TextPart

::: dashscope.types.messages.ImagePart

::: dashscope.types.messages.AudioPart

::: dashscope.types.messages.VideoPart

::: dashscope.types.messages.ToolCall

::: dashscope.types.messages.ToolCallFunction

## 补全模型

::: dashscope.types.completions.ChatCompletion

::: dashscope.types.completions.ChatCompletionChunk

::: dashscope.types.completions.ChatCompletionMessage

::: dashscope.types.completions.Choice

::: dashscope.types.completions.StreamChoice

::: dashscope.types.completions.Delta

::: dashscope.types.completions.SearchInfo

::: dashscope.types.completions.SearchResult

## 通用与分页

::: dashscope.types.common.ResponseBase

::: dashscope.types.common.Usage

::: dashscope.types.common.JobStatus

::: dashscope.types.pagination.BasePage

## 辅助函数

::: dashscope.types.messages.normalize_messages

::: dashscope.types.messages.extract_text

::: dashscope.types.completions.parse_completion

::: dashscope.types.completions.parse_chunk