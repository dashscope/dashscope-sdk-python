# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared typed models for the dashscope 2.0 SDK."""

from dashscope.v2.types.common import JobStatus, ResponseBase, Usage
from dashscope.v2.types.completions import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    Delta,
    StreamChoice,
    parse_chunk,
    parse_completion,
)
from dashscope.v2.types.messages import (
    AudioPart,
    ImagePart,
    Message,
    TextPart,
    ToolCall,
    ToolCallFunction,
    VideoPart,
    extract_text,
    normalize_messages,
)

__all__ = [
    "AudioPart",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionMessage",
    "Choice",
    "Delta",
    "ImagePart",
    "JobStatus",
    "Message",
    "ResponseBase",
    "StreamChoice",
    "TextPart",
    "ToolCall",
    "ToolCallFunction",
    "Usage",
    "VideoPart",
    "extract_text",
    "normalize_messages",
    "parse_chunk",
    "parse_completion",
]
