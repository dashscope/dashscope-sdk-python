# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Typed models for the Anthropic-compatible Messages API (C3).

These are *native* Anthropic types (a ``message`` object whose ``content``
array carries text / thinking / tool_use / server-tool blocks), distinct
from the canonical chat :class:`dashscope.types.Message` used by
``client.chat``. The ``client.messages`` resource returns these directly;
the ``protocol="anthropic"`` chat adapter keeps normalizing to chat types.

Every model is permissive (``extra="allow"``, optional defaulted fields):
Bailian adds extensions (``output_config`` echo, two-generation cache
``usage``, ``server_tool_use`` counters, ``x_tools``) and the content-block
shape varies by ``type``. A single permissive block model keeps every type
readable without a discriminated union the tolerant parser would have to
guess -- the same approach :mod:`dashscope.types.responses` takes.

Contract:
https://help.aliyun.com/zh/model-studio/anthropic-api-messages
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from dashscope.types._parse import parse_model
from dashscope.types.common import ResponseBase, response_envelope

if TYPE_CHECKING:
    import httpx

__all__ = [
    "AnthropicCacheCreation",
    "AnthropicContentBlock",
    "AnthropicMessage",
    "AnthropicServerToolUsage",
    "AnthropicStreamEvent",
    "AnthropicUsage",
    "CacheControl",
    "OutputConfig",
    "ThinkingConfig",
    "WebSearchResult",
    "parse_message",
    "parse_message_stream_event",
]


# --------------------------------------------------------------------------
# Content blocks (one permissive model covers every wire type)
# --------------------------------------------------------------------------
class WebSearchResult(BaseModel):
    """One ``web_search_tool_result.content[]`` element."""

    model_config = ConfigDict(extra="allow")

    type: str = "web_search_result"
    title: Optional[str] = None
    url: Optional[str] = None
    encrypted_content: Optional[str] = None
    page_age: Optional[str] = None


class AnthropicContentBlock(BaseModel):
    """One ``content[]`` block, covering every wire type.

    Fields are conditional on ``type`` (the service omits the rest):
    ``text`` -> text; ``thinking`` -> thinking/signature; ``redacted_thinking``
    -> data; ``tool_use`` / ``server_tool_use`` -> id/name/input (+caller);
    ``web_search_tool_result`` -> tool_use_id/content (a list of
    :class:`WebSearchResult`); ``web_fetch_tool_result`` -> tool_use_id/content
    (a document object); ``tool_result`` -> tool_use_id/content (string or
    blocks). ``cache_control`` may ride on input blocks (explicit cache).
    """

    model_config = ConfigDict(extra="allow")

    type: Optional[str] = None
    text: Optional[str] = None
    thinking: Optional[str] = None
    signature: Optional[str] = None
    data: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    tool_use_id: Optional[str] = None
    content: Optional[Any] = None
    caller: Optional[Dict[str, Any]] = None
    cache_control: Optional[Dict[str, Any]] = None

    @property
    def search_results(self) -> List[WebSearchResult]:
        """Typed view of a ``web_search_tool_result`` block's content."""
        content = self.content
        if not isinstance(content, list):
            return []
        # list() materializes the Optional[Any] so type checkers see an
        # iterable (astroid cannot narrow it through the isinstance guard).
        return [
            parse_model(WebSearchResult, item)
            for item in list(content)
            if isinstance(item, dict)
        ]


# --------------------------------------------------------------------------
# Usage (cache + server-tool counters; optional-field union)
# --------------------------------------------------------------------------
class AnthropicServerToolUsage(BaseModel):
    """Server-side tool call counts, returned only when such a tool ran."""

    model_config = ConfigDict(extra="allow")

    web_search_requests: int = 0
    web_fetch_requests: int = 0


class AnthropicCacheCreation(BaseModel):
    """Cache-creation tokens split by TTL (qwen4-max explicit cache)."""

    model_config = ConfigDict(extra="allow")

    ephemeral_5m_input_tokens: int = 0
    ephemeral_1h_input_tokens: int = 0


class AnthropicUsage(BaseModel):
    """Token usage. ``prompt_tokens_details.cached_tokens`` equals
    ``cache_read_input_tokens`` -- do not double-count. ``input_tokens``
    excludes cached/created tokens on qwen4-max."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    server_tool_use: Optional[AnthropicServerToolUsage] = None
    cache_creation: Optional[AnthropicCacheCreation] = None
    prompt_tokens_details: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------
# Request-side configuration (Bailian extensions; also accepted as dicts)
# --------------------------------------------------------------------------
class ThinkingConfig(BaseModel):
    """Deep-thinking control. ``type``: enabled / disabled.
    ``budget_tokens`` is deprecated in favour of ``output_config.effort``.
    ``display="summarized"`` (qwen4-max) returns a summary + signature."""

    model_config = ConfigDict(extra="allow")

    type: Optional[str] = None
    budget_tokens: Optional[int] = None
    display: Optional[str] = None


class OutputConfig(BaseModel):
    """Bailian extension. ``effort`` controls reasoning strength (per-model
    tiers differ); ``format`` (``{type: json_schema, schema}``) enables
    structured output."""

    model_config = ConfigDict(extra="allow")

    effort: Optional[str] = None
    format: Optional[Dict[str, Any]] = None


class CacheControl(BaseModel):
    """Explicit-cache breakpoint (qwen4-max). ``type`` is ``ephemeral``;
    ``ttl`` is 5m (default) / 1h. Rides on a block or the request top level."""

    model_config = ConfigDict(extra="allow")

    type: str = "ephemeral"
    ttl: Optional[str] = None


# --------------------------------------------------------------------------
# Top-level message
# --------------------------------------------------------------------------
class AnthropicMessage(ResponseBase):
    """A Messages reply (create). ``stop_reason``: end_turn / max_tokens /
    tool_use / stop_sequence. ``content`` carries the typed blocks."""

    id: Optional[str] = None
    type: Optional[str] = "message"
    role: Optional[str] = "assistant"
    model: Optional[str] = None
    content: List[AnthropicContentBlock] = []
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Optional[AnthropicUsage] = None

    @property
    def text(self) -> str:
        """Concatenated text of every ``text`` block (skips thinking/tools)."""
        return "".join(
            block.text
            for block in self.content
            if block.type == "text" and block.text
        )


class AnthropicStreamEvent(ResponseBase):
    """One streaming event. ``type`` identifies it: ``message_start``
    (carries ``message``), ``content_block_start`` (``index`` +
    ``content_block``), ``content_block_delta`` (``index`` + ``delta``),
    ``content_block_stop``, ``message_delta`` (``delta`` stop_reason +
    full ``usage``), ``message_stop``, ``ping``. All fields optional."""

    type: Optional[str] = None
    message: Optional[AnthropicMessage] = None
    index: Optional[int] = None
    content_block: Optional[AnthropicContentBlock] = None
    delta: Optional[Dict[str, Any]] = None
    usage: Optional[AnthropicUsage] = None


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------
def parse_message(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional[httpx.Response] = None,
) -> AnthropicMessage:
    payload = dict(body) if isinstance(body, dict) else {}
    return parse_model(
        AnthropicMessage,
        {**payload, **response_envelope(payload)},
        strict=strict,
        response=response,
    )


def parse_message_stream_event(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional[httpx.Response] = None,
) -> AnthropicStreamEvent:
    payload = dict(body) if isinstance(body, dict) else {}
    return parse_model(
        AnthropicStreamEvent,
        {**payload, **response_envelope(payload)},
        strict=strict,
        response=response,
    )
