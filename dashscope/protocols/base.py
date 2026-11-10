# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Protocol adapter interface for the dashscope 2.0 SDK.

A protocol adapter translates between the SDK's canonical chat models
and one wire format (native / openai / anthropic, see design doc §3).
"""

from __future__ import annotations

import functools
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
    import httpx

    from dashscope.types.completions import (
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
        *,
        strict_validation: bool = False,
        response: Optional[httpx.Response] = None,
    ) -> ChatCompletion:
        """Translate a non-streaming response body.

        ``strict_validation`` raises on a body that does not fit the
        typed model instead of degrading it; it is unrelated to
        :attr:`strict`, which governs *request* construction.
        ``response`` is only carried onto that raised error, so the
        caller can inspect the status and headers of the body that
        failed; websocket payloads have none and leave it ``None``.
        """

    @abstractmethod
    def parse_chunk(
        self,
        payload: Dict[str, Any],
        *,
        strict_validation: bool = False,
        response: Optional[httpx.Response] = None,
    ) -> Optional[ChatCompletionChunk]:
        """Translate one streaming event body; ``None`` skips the event
        (keepalives and control frames carry no user-facing content)."""

    def make_chunk_parser(
        self,
        *,
        strict_validation: bool = False,
        response: Optional[httpx.Response] = None,
    ) -> Callable[[Dict[str, Any]], Optional[ChatCompletionChunk]]:
        """Return the chunk parser for one new stream.

        Stateless adapters reuse ``parse_chunk``; stateful adapters
        (e.g. anthropic tool-use accumulation) return a fresh parser
        per stream so concurrent streams never share state.

        The returned callable takes only the payload, so the validation
        mode and the response a strict failure should carry are bound
        here rather than passed per event.
        """
        if not strict_validation and response is None:
            return self.parse_chunk
        return functools.partial(
            self.parse_chunk,
            strict_validation=strict_validation,
            response=response,
        )
