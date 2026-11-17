# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Typed models for the OpenAI-compatible Responses API (C1).

The Responses wire is a flat OpenAI ``response`` object whose ``output``
array carries ten heterogeneous item types (message, reasoning,
function_call and the seven built-in tool calls). Chat's choice/message
model cannot express those, so they are modelled independently here.

Every model is permissive (``extra="allow"``, optional defaulted fields):
the service echoes OpenAI-protocol fields it does not support as ``null``
and adds Bailian extensions (``tool_usage``, ``ocr_result``, two-generation
``usage``). Parsing degrades rather than raises so a drifted field never
sinks an otherwise healthy response.

Contract:
https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-responses
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict

from dashscope.types._parse import parse_model
from dashscope.types.common import ResponseBase, response_envelope

if TYPE_CHECKING:
    import httpx

__all__ = [
    "CodeInterpreterTool",
    "FileSearchTool",
    "FunctionTool",
    "ImageSearchTool",
    "InputItem",
    "InputTokensDetails",
    "McpTool",
    "OutputTokensDetails",
    "PromptCacheOptions",
    "Reasoning",
    "Response",
    "ResponseContentPart",
    "ResponseDeleted",
    "ResponseError",
    "ResponseOutputItem",
    "ResponseStreamEvent",
    "ResponseUsage",
    "SummaryText",
    "ToolUsage",
    "ToolUsageCount",
    "WebExtractorTool",
    "WebSearchImageTool",
    "WebSearchTool",
    "parse_response",
    "parse_response_deleted",
    "parse_stream_event",
]


# --------------------------------------------------------------------------
# Content blocks
# --------------------------------------------------------------------------
class ResponseContentPart(BaseModel):
    """A content block. Output uses ``output_text``; input items reuse the
    same shape with ``input_text``. ``logprobs`` is currently always null."""

    model_config = ConfigDict(extra="allow")

    type: Optional[str] = None
    text: Optional[str] = None
    annotations: List[Dict[str, Any]] = []
    logprobs: Optional[Any] = None


class SummaryText(BaseModel):
    """One reasoning-summary element (``reasoning.summary[]``)."""

    model_config = ConfigDict(extra="allow")

    type: str = "summary_text"
    text: Optional[str] = None


class ResponseOutputItem(BaseModel):
    """One ``output[]`` item, covering all ten wire types.

    Fields are conditional on ``type`` (the service omits the rest):
    ``message`` -> role/content; ``reasoning`` -> summary/encrypted_content;
    ``function_call`` -> name/arguments/call_id; ``web_search_call`` ->
    action; ``code_interpreter_call`` -> code/outputs/container_id;
    ``web_extractor_call`` -> goal/output/urls; ``web_search_image_call`` /
    ``image_search_call`` -> name/arguments/output; ``mcp_call`` ->
    server_label/name/arguments/output; ``file_search_call`` ->
    queries/results. A single permissive model keeps every type readable
    without a discriminated union the tolerant parser would have to guess.
    """

    model_config = ConfigDict(extra="allow")

    type: Optional[str] = None
    id: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    call_id: Optional[str] = None
    content: List[ResponseContentPart] = []
    summary: List[SummaryText] = []
    encrypted_content: Optional[str] = None
    action: Optional[Dict[str, Any]] = None
    code: Optional[str] = None
    outputs: List[Dict[str, Any]] = []
    container_id: Optional[str] = None
    goal: Optional[str] = None
    output: Optional[str] = None
    urls: List[str] = []
    server_label: Optional[str] = None
    queries: List[str] = []
    results: List[Dict[str, Any]] = []


# --------------------------------------------------------------------------
# Usage (two generations share one optional-field union)
# --------------------------------------------------------------------------
class CacheWriteTokensDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    ttl_1h_tokens: int = 0
    ttl_5m_tokens: int = 0


class CacheCreation(BaseModel):
    model_config = ConfigDict(extra="allow")

    ephemeral_5m_input_tokens: int = 0
    ephemeral_1h_input_tokens: int = 0
    cache_type: Optional[str] = None


class InputTokensDetails(BaseModel):
    """Old gen: ``cached_tokens`` (+ multimodal text/image/audio). New gen
    (Qwen4 / Kimi-K3.1+): ``cache_write_tokens`` + ``cache_write_tokens_details``.
    Session cache adds ``cache_creation_input_tokens`` / ``cache_creation``."""

    model_config = ConfigDict(extra="allow")

    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_tokens_details: Optional[CacheWriteTokensDetails] = None
    cache_creation_input_tokens: int = 0
    cache_creation: Optional[CacheCreation] = None
    text_tokens: Optional[int] = None
    image_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None


class OutputTokensDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    reasoning_tokens: int = 0
    text_tokens: Optional[int] = None


class ResponseUsage(BaseModel):
    """Token usage. ``x_details`` / ``x_tools`` / ``plugins`` are old-gen
    only; the new structure omits them. No model hardcoding -- whichever
    fields the service returns are surfaced."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: Optional[InputTokensDetails] = None
    output_tokens_details: Optional[OutputTokensDetails] = None
    x_details: List[Dict[str, Any]] = []
    x_tools: Optional[Dict[str, Any]] = None
    plugins: Optional[Dict[str, Any]] = None
    prompt_tokens_details: Optional[Dict[str, Any]] = None


class ToolUsageCount(BaseModel):
    model_config = ConfigDict(extra="allow")

    num_requests: int = 0
    count: Optional[int] = None


class ToolUsage(BaseModel):
    """Built-in tool call counts, billed separately from token ``usage``.
    Empty object ``{}`` when no tool ran."""

    model_config = ConfigDict(extra="allow")

    web_search: Optional[ToolUsageCount] = None
    code_interpreter: Optional[ToolUsageCount] = None
    web_extractor: Optional[ToolUsageCount] = None
    image_search: Optional[ToolUsageCount] = None
    web_search_image: Optional[ToolUsageCount] = None
    pdf_page_parser: Optional[ToolUsageCount] = None


class ResponseError(BaseModel):
    """Failure object (``error``); null on success. Create uses
    ``{code, message}``, retrieve/delete use ``{message, type}``."""

    model_config = ConfigDict(extra="allow")

    code: Optional[str] = None
    message: Optional[str] = None
    type: Optional[str] = None


# --------------------------------------------------------------------------
# Request-side configuration (also echoed on the response)
# --------------------------------------------------------------------------
class Reasoning(BaseModel):
    """Thinking control. ``effort``: none/minimal/low/medium/high/xhigh/max
    (per-model mapping differs; not hard-validated so new tiers pass through).
    ``summary="auto"`` (qwen4-max) returns a summary + encrypted_content."""

    model_config = ConfigDict(extra="allow")

    effort: Optional[str] = None
    summary: Optional[str] = None


class PromptCacheOptions(BaseModel):
    """Explicit cache (qwen4-max). ``mode``: implicit (default) / explicit;
    ``ttl``: 5m (default) / 1h."""

    model_config = ConfigDict(extra="allow")

    mode: Optional[str] = None
    ttl: Optional[str] = None


class OcrOptions(BaseModel):
    """OCR task config (qwen3.5-ocr). ``task=document_parsing`` raises the
    PDF page cap from 10 to 50."""

    model_config = ConfigDict(extra="allow")

    task: Optional[str] = None


class PromptCacheBreakpoint(BaseModel):
    """Manual cache breakpoint on an input block (qwen4-max)."""

    model_config = ConfigDict(extra="allow")

    mode: str = "explicit"


# --------------------------------------------------------------------------
# Tools (eight built-in/custom types; plain dicts are accepted too)
# --------------------------------------------------------------------------
class WebSearchTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["web_search"] = "web_search"


class WebExtractorTool(BaseModel):
    """Must be declared together with ``web_search``."""

    model_config = ConfigDict(extra="allow")

    type: Literal["web_extractor"] = "web_extractor"


class CodeInterpreterTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["code_interpreter"] = "code_interpreter"


class WebSearchImageTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["web_search_image"] = "web_search_image"


class ImageSearchTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["image_search"] = "image_search"


class FileSearchTool(BaseModel):
    """Knowledge-base search. ``vector_store_ids`` currently accepts one id."""

    model_config = ConfigDict(extra="allow")

    type: Literal["file_search"] = "file_search"
    vector_store_ids: List[str] = []
    max_num_results: Optional[int] = None


class McpTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["mcp"] = "mcp"
    server_protocol: Optional[str] = None
    server_label: Optional[str] = None
    server_url: Optional[str] = None
    server_description: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


class FunctionTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["function"] = "function"
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None


Tool = Union[
    WebSearchTool,
    WebExtractorTool,
    CodeInterpreterTool,
    WebSearchImageTool,
    ImageSearchTool,
    FileSearchTool,
    McpTool,
    FunctionTool,
    Dict[str, Any],
]


# --------------------------------------------------------------------------
# Top-level responses
# --------------------------------------------------------------------------
class Response(ResponseBase):
    """A stored response (create / get). ``id`` is valid for 7 days and can
    seed ``previous_response_id``. ``created_at`` is Unix seconds."""

    id: Optional[str] = None
    object: Optional[str] = "response"
    created_at: Optional[int] = None
    completed_at: Optional[int] = None
    status: Optional[str] = None
    model: Optional[str] = None
    output: List[ResponseOutputItem] = []
    usage: Optional[ResponseUsage] = None
    tool_usage: Optional[ToolUsage] = None
    error: Optional[ResponseError] = None
    incomplete_details: Optional[Dict[str, Any]] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[Any] = None
    store: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    instructions: Optional[str] = None
    tools: List[Dict[str, Any]] = []
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    top_logprobs: Optional[int] = None
    service_tier: Optional[str] = None
    background: Optional[bool] = None
    reasoning: Optional[Reasoning] = None
    prompt_cache_options: Optional[PromptCacheOptions] = None
    text: Optional[Dict[str, Any]] = None
    ocr_result: Optional[Any] = None

    @property
    def output_text(self) -> str:
        """Concatenated text of every ``message`` item's output_text blocks."""
        parts: List[str] = []
        for item in self.output:
            if item.type != "message":
                continue
            for part in item.content:
                if part.type == "output_text" and part.text:
                    parts.append(part.text)
        return "".join(parts)


class ResponseDeleted(ResponseBase):
    """Result of ``delete``. The wire returns only ``{id, deleted}`` (no
    ``object`` field, unlike the OpenAI SDK's ``response.deleted``)."""

    id: Optional[str] = None
    deleted: bool = False


class InputItem(BaseModel):
    """One ``input_items`` entry: a user/assistant message whose content
    blocks are ``input_text`` (user) or ``output_text`` (assistant)."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    type: Optional[str] = "message"
    role: Optional[str] = None
    status: Optional[str] = None
    content: List[ResponseContentPart] = []


class ResponseStreamEvent(ResponseBase):
    """One streaming event. ``type`` + ``sequence_number`` identify it;
    ``response.completed`` carries the full :class:`Response` and ends the
    stream. Which of response/item/part/delta/text is populated depends on
    ``type`` (all optional, permissive)."""

    type: Optional[str] = None
    sequence_number: Optional[int] = None
    response: Optional[Response] = None
    item: Optional[ResponseOutputItem] = None
    part: Optional[ResponseContentPart] = None
    delta: Optional[str] = None
    text: Optional[str] = None
    item_id: Optional[str] = None
    output_index: Optional[int] = None
    content_index: Optional[int] = None
    logprobs: Optional[Any] = None


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------
def parse_response(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional[httpx.Response] = None,
) -> Response:
    payload = dict(body) if isinstance(body, dict) else {}
    return parse_model(
        Response,
        {**payload, **response_envelope(payload)},
        strict=strict,
        response=response,
    )


def parse_response_deleted(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional[httpx.Response] = None,
) -> ResponseDeleted:
    payload = dict(body) if isinstance(body, dict) else {}
    return parse_model(
        ResponseDeleted,
        {**payload, **response_envelope(payload)},
        strict=strict,
        response=response,
    )


def parse_stream_event(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional[httpx.Response] = None,
) -> ResponseStreamEvent:
    payload = dict(body) if isinstance(body, dict) else {}
    return parse_model(
        ResponseStreamEvent,
        {**payload, **response_envelope(payload)},
        strict=strict,
        response=response,
    )