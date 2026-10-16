# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Typed chat response models (canonical, protocol-agnostic)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from dashscope.v2.types.common import ResponseBase, Usage
from dashscope.v2.types.messages import ToolCall

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
    reasoning_content: Optional[str] = None
    reasoning_signature: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class Delta(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Optional[str] = None
    content: Optional[str] = None
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


def _canonical_choices(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    choices = output.get("choices")
    if choices:
        canonical = []
        for raw in choices:
            raw = dict(raw)
            for key in ("message", "delta"):
                msg = raw.get(key)
                if isinstance(msg, dict):
                    msg = dict(msg)
                    msg["content"] = _flatten_content(msg.get("content"))
                    raw[key] = msg
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


def parse_completion(body: Dict[str, Any]) -> ChatCompletion:
    output = body.get("output") or {}
    return ChatCompletion(
        request_id=body.get("request_id"),
        model=body.get("model"),
        choices=_canonical_choices(output),
        usage=body.get("usage"),
        search_info=output.get("search_info"),
    )


def parse_chunk(body: Dict[str, Any]) -> ChatCompletionChunk:
    output = body.get("output") or {}
    choices = []
    for raw in _canonical_choices(output):
        raw = dict(raw)
        if "message" in raw:
            raw["delta"] = raw.pop("message")
        choices.append(raw)
    return ChatCompletionChunk(
        request_id=body.get("request_id"),
        model=body.get("model"),
        choices=choices,
        usage=body.get("usage"),
        search_info=output.get("search_info"),
    )
