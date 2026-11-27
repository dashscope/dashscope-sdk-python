---
description: "共享 pydantic 模型 API 参考：消息、补全、分页、辅助函数、原始响应与解析模式。"
---
# 类型（Types）

消息、补全、响应与分页的共享 pydantic 模型，以及消息规范化辅助函数。
大部分名称可从 `dashscope.types` 直接导入。最后一节说明所有资源
共享的原始响应包装器与解析/错误模式契约。

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

::: dashscope.types.common.with_headers

::: dashscope.types.common.flatten_envelope

::: dashscope.types.common.response_body

::: dashscope.types.common.response_envelope

::: dashscope.types.pagination.BasePage

## 辅助函数

::: dashscope.types.messages.normalize_messages

::: dashscope.types.messages.extract_text

::: dashscope.types.messages.has_media_parts

::: dashscope.types.completions.parse_completion

::: dashscope.types.completions.parse_chunk

## 原始响应与解析 { #raw-responses-parsing }

资源调用传 `raw=True` 会在任何错误检查与解析之前返回 `APIResponse`：
可读取 `status_code`、`headers`、`elapsed`、`request_id`，用 `json()`
解码响应体，或用 `parse(Model)` 构建类型化模型。下面四个开关
相互独立：

| 开关 | 作用域 | 效果 |
| --- | --- | --- |
| 降级解析（默认） | 每个类型化响应 | 漂移字段被修复为默认值/零值并记录 `debug` 日志；未知键保留。服务端字段变化不会让调用失败。 |
| `strict_validation=True` | `Client(...)` / `dashscope.settings` / 配置文件 | 关闭降级：漂移响应抛出 `APIResponseValidationError`（携带 `response` 与原始 `body`）。即使在不抛异常模式下也会触发。 |
| `raise_on_error=False` | `Client(...)` | HTTP/业务失败以类型化信封返回（`is_error`、`code`、`message`），不再抛出。不会吞掉严格校验异常或本地 `ValueError`。 |
| `raw=True` | 单次调用 | 在错误检查与解析之前返回 `APIResponse`；随后 `parse(Model)` 按客户端的严格度解析。raw 响应不带分页回调，`parse()` 也不会重放资源自定义的响应整形。 |

::: dashscope.transport.APIResponse

::: dashscope.types._parse.parse_model

## Anthropic Messages

::: dashscope.types.anthropic_messages.WebSearchResult

::: dashscope.types.anthropic_messages.AnthropicContentBlock

::: dashscope.types.anthropic_messages.AnthropicServerToolUsage

::: dashscope.types.anthropic_messages.AnthropicCacheCreation

::: dashscope.types.anthropic_messages.AnthropicUsage

::: dashscope.types.anthropic_messages.ThinkingConfig

::: dashscope.types.anthropic_messages.OutputConfig

::: dashscope.types.anthropic_messages.CacheControl

::: dashscope.types.anthropic_messages.AnthropicMessage

::: dashscope.types.anthropic_messages.AnthropicStreamEvent

::: dashscope.types.anthropic_messages.parse_message

::: dashscope.types.anthropic_messages.parse_message_stream_event

## Conversations

::: dashscope.types.conversations.ConversationContentPart

::: dashscope.types.conversations.Conversation

::: dashscope.types.conversations.ConversationDeleted

::: dashscope.types.conversations.ConversationItem

::: dashscope.types.conversations.ConversationItemDeleted

::: dashscope.types.conversations.parse_conversation

::: dashscope.types.conversations.parse_conversation_deleted

::: dashscope.types.conversations.parse_conversation_item

::: dashscope.types.conversations.parse_conversation_item_deleted

## Responses

::: dashscope.types.responses.ResponseContentPart

::: dashscope.types.responses.SummaryText

::: dashscope.types.responses.ResponseOutputItem

::: dashscope.types.responses.InputTokensDetails

::: dashscope.types.responses.OutputTokensDetails

::: dashscope.types.responses.ResponseUsage

::: dashscope.types.responses.ToolUsageCount

::: dashscope.types.responses.ToolUsage

::: dashscope.types.responses.ResponseError

::: dashscope.types.responses.Reasoning

::: dashscope.types.responses.PromptCacheOptions

::: dashscope.types.responses.WebSearchTool

::: dashscope.types.responses.WebExtractorTool

::: dashscope.types.responses.CodeInterpreterTool

::: dashscope.types.responses.WebSearchImageTool

::: dashscope.types.responses.ImageSearchTool

::: dashscope.types.responses.FileSearchTool

::: dashscope.types.responses.McpTool

::: dashscope.types.responses.FunctionTool

::: dashscope.types.responses.Response

::: dashscope.types.responses.ResponseDeleted

::: dashscope.types.responses.InputItem

::: dashscope.types.responses.ResponseStreamEvent

::: dashscope.types.responses.parse_response

::: dashscope.types.responses.parse_response_deleted

::: dashscope.types.responses.parse_stream_event
