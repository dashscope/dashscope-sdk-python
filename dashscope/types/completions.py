# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Typed chat response models (canonical, protocol-agnostic)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from dashscope.types._parse import parse_model
from dashscope.types.common import ResponseBase, Usage
from dashscope.types.messages import ToolCall

if TYPE_CHECKING:
    import httpx

__all__ = [
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionMessage",
    "Choice",
    "Delta",
    "SearchInfo",
    "SearchResult",
    "StreamChoice",
]


class SearchResult(BaseModel):
    """One web-search citation (``output.search_info.search_results[]``)."""

    model_config = ConfigDict(extra="allow")

    title: Optional[str] = None
    url: Optional[str] = None
    site_name: Optional[str] = None
    index: Optional[int] = None


class SearchInfo(BaseModel):
    """Web-search citations, present when search is enabled."""

    model_config = ConfigDict(extra="allow")

    search_results: List[SearchResult] = []


class ChatCompletionMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = "assistant"
    content: Optional[str] = None
    # Non-text parts of a list-form content (image / audio / ...). Text
    # already reads through ``content``; without this, flattening the
    # list would drop generated media such as wan2.7 image URLs.
    content_parts: Optional[List[Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None
    reasoning_signature: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class Delta(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Optional[str] = None
    content: Optional[str] = None
    content_parts: Optional[List[Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None
    reasoning_signature: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    finish_reason: Optional[str] = None
    message: ChatCompletionMessage


class StreamChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    finish_reason: Optional[str] = None
    delta: Delta


class ChatCompletion(ResponseBase):
    model: Optional[str] = None
    choices: List[Choice] = []
    usage: Optional[Usage] = None
    search_info: Optional[SearchInfo] = None

    @property
    def output_text(self) -> str:
        if not self.choices:
            return ""
        return self.choices[0].message.content or ""


class ChatCompletionChunk(ResponseBase):
    model: Optional[str] = None
    choices: List[StreamChoice] = []
    usage: Optional[Usage] = None
    search_info: Optional[SearchInfo] = None

    @property
    def delta(self) -> Optional[Delta]:
        if not self.choices:
            return None
        return self.choices[0].delta


def _flatten_content(content: Any) -> Any:
    # Multimodal endpoints return list-form content ([{"text": "..."}]);
    # join the text parts so canonical models stay string-typed.
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(part["text"])
        return "".join(parts) or None
    return content


def _media_parts(content: Any) -> Optional[List[Dict[str, Any]]]:
    """The non-text parts of a list-form content.

    ``content`` stays a string for compatibility, so image / audio parts
    need their own field -- dropping them loses generated media URLs.
    """
    if not isinstance(content, list):
        return None
    parts = [
        part
        for part in content
        if isinstance(part, dict) and not part.get("text")
    ]
    return parts or None


def _normalize_message_content(message: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten text and keep media parts on one message / delta."""
    normalized = dict(message)
    content = normalized.get("content")
    normalized["content"] = _flatten_content(content)
    parts = _media_parts(content)
    if parts is not None:
        normalized["content_parts"] = parts
    return normalized


def _output(body: Dict[str, Any]) -> Dict[str, Any]:
    """``body["output"]`` when it is a mapping, else empty.

    A drifted ``output`` (a string, a number) has no choices or
    ``search_info`` to read, and ``.get`` on it would escape as a bare
    ``AttributeError`` instead of reaching ``parse_model``.
    """
    output = body.get("output")
    return output if isinstance(output, dict) else {}


def _canonical_choices(output: Dict[str, Any]) -> List[Any]:
    choices = output.get("choices")
    if choices:
        canonical = []
        for raw in choices:
            if not isinstance(raw, dict):
                # ``dict(raw)`` on a string or number would escape as a
                # bare ValueError/TypeError. Passing it through keeps the
                # element count, so parse_model repairs it in place and
                # the healthy choices stay positionally aligned.
                canonical.append(raw)
                continue
            raw = dict(raw)
            for key in ("message", "delta"):
                msg = raw.get(key)
                if isinstance(msg, dict):
                    raw[key] = _normalize_message_content(msg)
            canonical.append(raw)
        return canonical
    text = output.get("text")
    if text is not None:
        return [
            {
                "index": 0,
                "finish_reason": output.get("finish_reason"),
                "message": {"role": "assistant", "content": text},
            },
        ]
    return []


def parse_completion(
    body: Dict[str, Any],
    *,
    strict_validation: bool = False,
    response: Optional[httpx.Response] = None,
) -> ChatCompletion:
    output = _output(body)
    return parse_model(
        ChatCompletion,
        {
            "request_id": body.get("request_id"),
            "model": body.get("model"),
            "choices": _canonical_choices(output),
            "usage": body.get("usage"),
            "search_info": output.get("search_info"),
        },
        strict=strict_validation,
        response=response,
    )


def parse_chunk(
    body: Dict[str, Any],
    *,
    strict_validation: bool = False,
    response: Optional[httpx.Response] = None,
) -> ChatCompletionChunk:
    output = _output(body)
    choices = []
    for raw in _canonical_choices(output):
        if isinstance(raw, dict):
            raw = dict(raw)
            if "message" in raw:
                raw["delta"] = raw.pop("message")
        choices.append(raw)
    return parse_model(
        ChatCompletionChunk,
        {
            "request_id": body.get("request_id"),
            "model": body.get("model"),
            "choices": choices,
            "usage": body.get("usage"),
            "search_info": output.get("search_info"),
        },
        strict=strict_validation,
        response=response,
    )
