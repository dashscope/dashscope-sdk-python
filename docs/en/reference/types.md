---
description: "API reference for shared pydantic models: messages, completions, pagination and helpers."
---
# Types

Shared pydantic models for messages, completions, responses and pagination,
plus message-normalization helpers. Most names are importable directly from
`dashscope.types`.

## Messages & content parts

::: dashscope.types.messages.Message

::: dashscope.types.messages.TextPart

::: dashscope.types.messages.ImagePart

::: dashscope.types.messages.AudioPart

::: dashscope.types.messages.VideoPart

::: dashscope.types.messages.ToolCall

::: dashscope.types.messages.ToolCallFunction

## Completion models

::: dashscope.types.completions.ChatCompletion

::: dashscope.types.completions.ChatCompletionChunk

::: dashscope.types.completions.ChatCompletionMessage

::: dashscope.types.completions.Choice

::: dashscope.types.completions.StreamChoice

::: dashscope.types.completions.Delta

::: dashscope.types.completions.SearchInfo

::: dashscope.types.completions.SearchResult

## Common & pagination

::: dashscope.types.common.ResponseBase

::: dashscope.types.common.Usage

::: dashscope.types.common.JobStatus

::: dashscope.types.pagination.BasePage

## Helpers

::: dashscope.types.messages.normalize_messages

::: dashscope.types.messages.extract_text

::: dashscope.types.completions.parse_completion

::: dashscope.types.completions.parse_chunk