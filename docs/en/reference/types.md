---
description: "API reference for shared pydantic models: messages, completions, pagination, helpers, raw responses and parsing modes."
---
# Types

Shared pydantic models for messages, completions, responses and pagination,
plus message-normalization helpers. Most names are importable directly from
`dashscope.types`. The last section documents the raw-response wrapper and
the parsing / error-mode contract shared by all resources.

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

::: dashscope.types.common.with_headers

::: dashscope.types.common.flatten_envelope

::: dashscope.types.common.response_body

::: dashscope.types.common.response_envelope

::: dashscope.types.pagination.BasePage

## Helpers

::: dashscope.types.messages.normalize_messages

::: dashscope.types.messages.extract_text

::: dashscope.types.messages.has_media_parts

::: dashscope.types.completions.parse_completion

::: dashscope.types.completions.parse_chunk

## Raw responses & parsing { #raw-responses-parsing }

Passing `raw=True` to a resource call returns an `APIResponse` before any
error check or parsing: read `status_code`, `headers`, `elapsed` and
`request_id`, decode the body with `json()`, or build a typed model with
`parse(Model)`. The four switches below are independent of each other:

| Switch | Scope | Effect |
| --- | --- | --- |
| Lenient parse (default) | every typed response | Drifted fields are repaired to their defaults / zero values and logged at `debug` level; unknown keys are kept. A server-side schema change does not break the call. |
| `strict_validation=True` | `Client(...)` / `dashscope.settings` / config file | Opts out of the repair: a drifted body raises `APIResponseValidationError` (carrying `response` and the raw `body`) instead of degrading. Fires even in non-raising mode. |
| `raise_on_error=False` | `Client(...)` | HTTP / business failures come back as the typed envelope (`is_error`, `code`, `message`) instead of raising. Does not suppress strict-validation errors or local `ValueError`. |
| `raw=True` | per call | Returns `APIResponse` before the error check and parsing; `parse(Model)` then applies the client's strictness. A raw response carries no pagination callback, and `parse()` does not replay a resource's custom reshaping. |

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
