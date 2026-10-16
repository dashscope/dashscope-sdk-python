# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Protocol adapter interface for the dashscope 2.0 SDK.

A protocol adapter translates between the SDK's canonical chat models
and one wire format (native / openai / anthropic, see design doc §3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Union,
)

if TYPE_CHECKING:
    # Annotation-only: avoid importing resources at runtime, which would
    # cycle back through completions.py into this partially-loaded module.
    from dashscope.v2.types.completions import (
        ChatCompletion,
        ChatCompletionChunk,
    )

__all__ = [
    "ChatProtocol",
    "ChatRequest",
]


@dataclass(frozen=True)
class ChatRequest:
    """Wire-level description of one chat request."""

    path: str
    body: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)


class ChatProtocol(ABC):
    """Translates canonical chat calls to one wire protocol."""

    name: ClassVar[str]

    # Endpoint for messages carrying media parts; None when the adapter
    # serves them from its single chat path.
    MULTIMODAL_PATH: ClassVar[Optional[str]] = None

    def __init__(self, *, strict: bool = False) -> None:
        self._strict = strict

    @property
    def strict(self) -> bool:
        """Reject (True) or pass through (False) unknown parameters.

        Inert on the native adapter: every known parameter maps into
        its ``parameters`` bag, so nothing is ever "unknown".
        """
        return self._strict

    @abstractmethod
    def build_chat_request(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Union[str, List[str], None] = None,
        seed: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        enable_thinking: Optional[bool] = None,
        response_format: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        tool_choice: Union[str, Dict[str, Any], None] = None,
        n: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        enable_search: Optional[bool] = None,
        search_options: Optional[Dict[str, Any]] = None,
        parallel_tool_calls: Optional[bool] = None,
        extra_parameters: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
    ) -> ChatRequest:
        """Build the wire request (path, headers, body) for one call.

        ``path`` overrides the adapter's default endpoint; callers pass
        ``MULTIMODAL_PATH`` when the messages carry media parts.
        """

    @abstractmethod
    def parse_completion(
        self,
        payload: Dict[str, Any],
    ) -> ChatCompletion:
        """Translate a non-streaming response body."""

    @abstractmethod
    def parse_chunk(
        self,
        payload: Dict[str, Any],
    ) -> Optional[ChatCompletionChunk]:
        """Translate one streaming event body; ``None`` skips the event
        (keepalives and control frames carry no user-facing content)."""

    def make_chunk_parser(
        self,
    ) -> Callable[[Dict[str, Any]], Optional[ChatCompletionChunk]]:
        """Return the chunk parser for one new stream.

        Stateless adapters reuse ``parse_chunk``; stateful adapters
        (e.g. anthropic tool-use accumulation) return a fresh parser
        per stream so concurrent streams never share state.
        """
        return self.parse_chunk
