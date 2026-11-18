# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.messages -- Anthropic-compatible Messages API (C3).

A *native* Anthropic Messages resource: ``create`` returns an
:class:`~dashscope.types.AnthropicMessage` whose ``content`` array carries
typed blocks (text / thinking / tool_use / server-tool results), and
``stream=True`` returns a :class:`MessageStream` of native SSE events.
This is distinct from ``Client(protocol="anthropic").chat``, which
normalizes the same wire onto canonical chat types; both paths share the
web-search billing-header injection so they behave identically.

Routing differs from the compatible-mode resources: Messages lives at
``{base}/apps/anthropic/v1/messages`` (base does not end in ``/v1``), via
``anthropic_base_url`` -- the chat ``protocol`` setting cannot move it.
Every call tags ``sdk_module`` ``messages`` for the
``x-dashscope-sdk-client`` header.

Bailian specifics honoured here: ``temperature`` range is [0, 2) (not the
official [0, 1]); ``output_config`` (effort + structured-output format),
block-level and top-level ``cache_control``, ``document`` / ``video``
blocks and ``thinking.display`` are first-class; ``thinking.budget_tokens``
is deprecated in favour of ``output_config.effort``. There is no
``count_tokens`` and no ``/v1/models`` on this endpoint (use
``client.models`` for discovery).

Contract:
https://help.aliyun.com/zh/model-studio/anthropic-api-messages
"""

from __future__ import annotations

import functools
import json
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
    overload,
)

from pydantic import BaseModel

from dashscope._config import Config, anthropic_base_url
from dashscope.errors import DashScopeError, response_error
from dashscope.protocols.anthropic import (
    AnthropicProtocol,
    ensure_billing_header,
)
from dashscope.transport._options import request_options
from dashscope.transport.sse import AsyncStream, Stream
from dashscope.types.anthropic_messages import (
    AnthropicMessage,
    AnthropicStreamEvent,
    parse_message,
    parse_message_stream_event,
)
from dashscope.types.common import with_headers

if TYPE_CHECKING:
    import httpx

    from dashscope.transport._response import APIResponse

_MESSAGES_PATH = "/v1/messages"

MODULE = "messages"

# Bailian widens the official Anthropic [0, 1] range to [0, 2).
_MIN_TEMPERATURE = 0.0
_MAX_TEMPERATURE = 2.0

_BUDGET_TOKENS_WARNED = False

MessagesInput = Sequence[Union[Mapping[str, Any], BaseModel]]
SystemInput = Union[str, Sequence[Union[Mapping[str, Any], BaseModel]]]
Tools = Sequence[Union[Mapping[str, Any], BaseModel]]
ConfigInput = Union[Mapping[str, Any], BaseModel]
ToolChoiceInput = Union[str, Mapping[str, Any], BaseModel]


def _endpoint_base(config: Config) -> str:
    # Messages sits under /apps/anthropic on every gateway; the chat
    # protocol setting must not move it.
    return anthropic_base_url(
        config.workspace,
        config.region,
        config.configured_base_url,
    )


def _normalize_items(
    value: Sequence[Union[Mapping[str, Any], BaseModel]],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, BaseModel):
            items.append(item.model_dump(exclude_none=True))
        elif isinstance(item, Mapping):
            items.append(dict(item))
        else:
            raise ValueError(
                "each item must be a mapping or a pydantic model"
            )
    return items


def _normalize_config(value: Optional[ConfigInput]) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    return dict(value)


def _normalize_system(value: Optional[SystemInput]) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return _normalize_items(value)


def _normalize_tool_choice(value: ToolChoiceInput) -> Dict[str, Any]:
    if isinstance(value, str):
        return {"type": value}
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError(
        "tool_choice must be a string, mapping or pydantic model"
    )


def _validate_temperature(temperature: Optional[float]) -> None:
    if temperature is None:
        return
    if not _MIN_TEMPERATURE <= temperature < _MAX_TEMPERATURE:
        raise ValueError(
            "temperature must be in [0, 2) on the Bailian Messages API "
            "(the official Anthropic [0, 1] range does not apply)"
        )


def _warn_budget_tokens() -> None:
    global _BUDGET_TOKENS_WARNED  # pylint: disable=global-statement
    if _BUDGET_TOKENS_WARNED:
        return
    _BUDGET_TOKENS_WARNED = True
    warnings.warn(
        "thinking.budget_tokens is deprecated on the Messages API; use "
        "output_config={'effort': ...} instead",
        category=DeprecationWarning,
        stacklevel=3,
    )


def _build_create_body(
    model: str,
    max_tokens: int,
    messages: MessagesInput,
    *,
    system: Optional[SystemInput],
    tools: Optional[Tools],
    tool_choice: Optional[ToolChoiceInput],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    stop_sequences: Optional[Sequence[str]],
    thinking: Optional[ConfigInput],
    output_config: Optional[ConfigInput],
    cache_control: Optional[ConfigInput],
    stream: bool,
) -> Dict[str, Any]:
    if not model or not model.strip():
        raise ValueError("model must be a non-empty string")
    if max_tokens is None or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    normalized_messages = _normalize_items(messages)
    if not normalized_messages:
        raise ValueError("messages must be a non-empty sequence")
    _validate_temperature(temperature)
    body: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": normalized_messages,
        "stream": stream,
    }
    normalized_tools = _normalize_items(tools) if tools is not None else None
    if normalized_tools is not None:
        body["tools"] = normalized_tools
    # Inject the web-search billing identifier when a server web_search
    # tool is declared (a caller-provided marker wins); shared with the
    # chat protocol="anthropic" adapter so both paths enable search alike.
    injected_system = ensure_billing_header(
        _normalize_system(system), normalized_tools
    )
    if injected_system is not None:
        body["system"] = injected_system
    optional = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    body.update({k: v for k, v in optional.items() if v is not None})
    if stop_sequences is not None:
        body["stop_sequences"] = list(stop_sequences)
    if tool_choice is not None:
        body["tool_choice"] = _normalize_tool_choice(tool_choice)
    thinking_dict = _normalize_config(thinking)
    if thinking_dict is not None:
        if thinking_dict.get("budget_tokens") is not None:
            _warn_budget_tokens()
        body["thinking"] = thinking_dict
    for key, value in (
        ("output_config", output_config),
        ("cache_control", cache_control),
    ):
        normalized = _normalize_config(value)
        if normalized is not None:
            body[key] = normalized
    return body


def _checked(client, resp, model_cls, parser):
    """Raise on failure, or fill the error envelope when not raising.

    Mirrors the per-resource pattern of responses/conversations: route the
    reply through ``response_error`` (HTTP >= 400, then HTTP-200 business
    code) so ``raise_on_error=False`` reports the failure via ``is_error``
    / ``code`` / ``message``. Anthropic failures nest ``message`` under
    ``error``, which ``error_from_status`` understands.
    """
    http_resp = resp.http_response
    error = response_error(http_resp)
    if error is None:
        return with_headers(
            parser(
                resp.json(),
                strict=client.config.strict_validation,
                response=http_resp,
            ),
            http_resp,
        )
    if client.raise_on_error:
        raise error
    return with_headers(
        model_cls(
            request_id=error.request_id,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            is_error=True,
        ),
        http_resp,
    )


def _parse_tool_json(raw: str) -> Dict[str, Any]:
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class _MessageBuilder:
    """Reassemble native SSE events into a final :class:`AnthropicMessage`.

    Anthropic streams carry no terminal full-message event (unlike
    Responses), so the message is accumulated: ``message_start`` seeds the
    skeleton, ``content_block_start`` opens a block, ``content_block_delta``
    appends text/thinking/signature or tool-input JSON, ``message_delta``
    supplies stop_reason + full usage, ``message_stop`` ends the stream.
    """

    def __init__(self) -> None:
        self._skeleton: Dict[str, Any] = {}
        self._blocks: Dict[int, Dict[str, Any]] = {}
        self._json: Dict[int, str] = {}
        self._usage: Dict[str, Any] = {}
        self._stop_reason: Optional[str] = None
        self._stop_sequence: Optional[str] = None
        self._saw_stop = False

    def feed(self, event: AnthropicStreamEvent) -> None:
        kind = event.type
        if kind == "message_start":
            message = event.message
            if message is not None:
                self._skeleton = {
                    "id": message.id,
                    "type": message.type,
                    "role": message.role,
                    "model": message.model,
                }
                if message.usage is not None:
                    self._usage = message.usage.model_dump(exclude_none=True)
        elif kind == "content_block_start":
            index = event.index
            block = (
                event.content_block.model_dump(exclude_none=True)
                if event.content_block is not None
                else {}
            )
            self._blocks[index] = block
            if block.get("type") == "tool_use":
                # tool_use input arrives incrementally via input_json_delta;
                # server_tool_use / web_search_tool_result carry complete
                # input/content in this start event (no deltas to buffer).
                self._json[index] = ""
                block.setdefault("input", {})
        elif kind == "content_block_delta":
            index = event.index
            delta = event.delta or {}
            block = self._blocks.setdefault(index, {})
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                block["text"] = (block.get("text") or "") + (
                    delta.get("text") or ""
                )
            elif delta_type == "thinking_delta":
                block["thinking"] = (block.get("thinking") or "") + (
                    delta.get("thinking") or ""
                )
            elif delta_type == "signature_delta":
                block["signature"] = delta.get("signature") or ""
            elif delta_type == "input_json_delta":
                self._json[index] = self._json.get(index, "") + (
                    delta.get("partial_json") or ""
                )
        elif kind == "content_block_stop":
            index = event.index
            if index in self._json:
                raw = self._json.pop(index)
                block = self._blocks.setdefault(index, {})
                if raw.strip():
                    block["input"] = _parse_tool_json(raw)
                elif not block.get("input"):
                    block["input"] = {}
        elif kind == "message_delta":
            delta = event.delta or {}
            if "stop_reason" in delta:
                self._stop_reason = delta.get("stop_reason")
            if "stop_sequence" in delta:
                self._stop_sequence = delta.get("stop_sequence")
            if event.usage is not None:
                self._usage.update(event.usage.model_dump(exclude_none=True))
        elif kind == "message_stop":
            self._saw_stop = True

    def build(self) -> AnthropicMessage:
        if not self._saw_stop:
            raise DashScopeError(
                "message stream ended without a message_stop event"
            )
        for index, raw in self._json.items():
            block = self._blocks.setdefault(index, {})
            if raw.strip():
                block["input"] = _parse_tool_json(raw)
            elif not block.get("input"):
                block["input"] = {}
        content = [self._blocks[i] for i in sorted(self._blocks)]
        return parse_message(
            {
                **self._skeleton,
                "content": content,
                "stop_reason": self._stop_reason,
                "stop_sequence": self._stop_sequence,
                "usage": self._usage or None,
            }
        )


class MessageStream:
    """Typed iterator over Messages SSE events.

    Consume it one way: iterate the events, read ``text_stream`` /
    ``thinking_stream``, or call ``get_final_message()`` -- each reads the
    live socket once. ``get_final_message`` reassembles the full
    :class:`AnthropicMessage` from the event sequence.
    """

    def __init__(self, inner: Stream[AnthropicStreamEvent]) -> None:
        self._inner = inner

    def __iter__(self) -> Iterator[AnthropicStreamEvent]:
        return iter(self._inner)

    @property
    def text_stream(self) -> Iterator[str]:
        for event in self:
            if event.type == "content_block_delta":
                delta = event.delta or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield delta["text"]

    @property
    def thinking_stream(self) -> Iterator[str]:
        for event in self:
            if event.type == "content_block_delta":
                delta = event.delta or {}
                if delta.get("type") == "thinking_delta" and delta.get(
                    "thinking"
                ):
                    yield delta["thinking"]

    def get_final_message(self) -> AnthropicMessage:
        builder = _MessageBuilder()
        for event in self:
            builder.feed(event)
        final = builder.build()
        final.headers = self.headers
        return final

    @property
    def headers(self) -> Dict[str, str]:
        """Raw HTTP response headers of the stream's envelope."""
        return self._inner.headers

    def close(self) -> None:
        self._inner.close()


class AsyncMessageStream:
    """Async twin of :class:`MessageStream`."""

    def __init__(self, inner: AsyncStream[AnthropicStreamEvent]) -> None:
        self._inner = inner

    def __aiter__(self):
        return self._inner.__aiter__()

    @property
    def text_stream(self):
        return self._text_stream()

    async def _text_stream(self):
        async for event in self:
            if event.type == "content_block_delta":
                delta = event.delta or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield delta["text"]

    @property
    def thinking_stream(self):
        return self._thinking_stream()

    async def _thinking_stream(self):
        async for event in self:
            if event.type == "content_block_delta":
                delta = event.delta or {}
                if delta.get("type") == "thinking_delta" and delta.get(
                    "thinking"
                ):
                    yield delta["thinking"]

    async def get_final_message(self) -> AnthropicMessage:
        builder = _MessageBuilder()
        async for event in self:
            builder.feed(event)
        final = builder.build()
        final.headers = self.headers
        return final

    @property
    def headers(self) -> Dict[str, str]:
        """Raw HTTP response headers of the stream's envelope."""
        return self._inner.headers

    async def aclose(self) -> None:
        await self._inner.aclose()


class Messages:
    """``client.messages`` -- the native Anthropic Messages resource."""

    def __init__(self, client) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: MessagesInput,
        stream: Literal[True],
        system: Optional[SystemInput] = None,
        tools: Optional[Tools] = None,
        tool_choice: Optional[ToolChoiceInput] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        thinking: Optional[ConfigInput] = None,
        output_config: Optional[ConfigInput] = None,
        cache_control: Optional[ConfigInput] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> MessageStream:
        ...

    @overload
    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: MessagesInput,
        stream: Literal[False] = False,
        system: Optional[SystemInput] = None,
        tools: Optional[Tools] = None,
        tool_choice: Optional[ToolChoiceInput] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        thinking: Optional[ConfigInput] = None,
        output_config: Optional[ConfigInput] = None,
        cache_control: Optional[ConfigInput] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> AnthropicMessage:
        ...

    @overload
    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: MessagesInput,
        stream: Literal[False] = False,
        system: Optional[SystemInput] = None,
        tools: Optional[Tools] = None,
        tool_choice: Optional[ToolChoiceInput] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        thinking: Optional[ConfigInput] = None,
        output_config: Optional[ConfigInput] = None,
        cache_control: Optional[ConfigInput] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: MessagesInput,
        stream: bool = False,
        system: Optional[SystemInput] = None,
        tools: Optional[Tools] = None,
        tool_choice: Optional[ToolChoiceInput] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        thinking: Optional[ConfigInput] = None,
        output_config: Optional[ConfigInput] = None,
        cache_control: Optional[ConfigInput] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[AnthropicMessage, MessageStream, "APIResponse"]:
        """Create a message (``POST /apps/anthropic/v1/messages``).

        ``messages`` are native Anthropic turns (role + content string or
        blocks); ``system`` is a string or block array. Bailian extensions
        are first-class: ``output_config`` (effort / structured-output
        format), ``cache_control`` (top-level auto cache; block-level
        markers ride inside system/messages blocks), ``thinking``
        (type / display). ``temperature`` must be in [0, 2). Declaring a
        server ``web_search`` tool auto-injects the billing identifier into
        system (a caller-provided marker wins). ``thinking.budget_tokens``
        is deprecated in favour of ``output_config.effort`` and emits a
        ``DeprecationWarning``.

        ``stream=True`` returns a :class:`MessageStream`; ``raw=True``
        returns the :class:`~dashscope.transport.APIResponse` before any
        error check (only with ``stream=False``; combining raises
        ``ValueError``).
        """
        body = _build_create_body(
            model,
            max_tokens,
            messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            thinking=thinking,
            output_config=output_config,
            cache_control=cache_control,
            stream=stream,
        )
        headers = {"anthropic-version": AnthropicProtocol.API_VERSION}
        options = request_options(
            extra_headers,
            extra_query,
            extra_body,
            max_retries=max_retries,
        )
        transport = self._client.transport
        base_url = _endpoint_base(self._client.config)
        if stream:
            if raw:
                raise ValueError("raw=True is not supported with stream=True")
            response = transport.open_stream(
                "POST",
                _MESSAGES_PATH,
                json=body,
                headers=headers,
                base_url=base_url,
                api_key=api_key,
                workspace=workspace,
                sdk_module=MODULE,
                timeout=timeout,
                options=options,
            )
            return MessageStream(
                Stream(
                    response,
                    functools.partial(
                        parse_message_stream_event,
                        strict=self._client.config.strict_validation,
                        response=response,
                    ),
                )
            )
        resp = transport.request_api(
            "POST",
            _MESSAGES_PATH,
            json=body,
            headers=headers,
            base_url=base_url,
            api_key=api_key,
            workspace=workspace,
            sdk_module=MODULE,
            timeout=timeout,
            options=options,
            client=self._client,
        )
        if raw:
            return resp
        return _checked(self._client, resp, AnthropicMessage, parse_message)


class AsyncMessages:
    """Async twin of :class:`Messages`."""

    def __init__(self, client) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: MessagesInput,
        stream: Literal[True],
        system: Optional[SystemInput] = None,
        tools: Optional[Tools] = None,
        tool_choice: Optional[ToolChoiceInput] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        thinking: Optional[ConfigInput] = None,
        output_config: Optional[ConfigInput] = None,
        cache_control: Optional[ConfigInput] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> AsyncMessageStream:
        ...

    @overload
    async def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: MessagesInput,
        stream: Literal[False] = False,
        system: Optional[SystemInput] = None,
        tools: Optional[Tools] = None,
        tool_choice: Optional[ToolChoiceInput] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        thinking: Optional[ConfigInput] = None,
        output_config: Optional[ConfigInput] = None,
        cache_control: Optional[ConfigInput] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> AnthropicMessage:
        ...

    @overload
    async def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: MessagesInput,
        stream: Literal[False] = False,
        system: Optional[SystemInput] = None,
        tools: Optional[Tools] = None,
        tool_choice: Optional[ToolChoiceInput] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        thinking: Optional[ConfigInput] = None,
        output_config: Optional[ConfigInput] = None,
        cache_control: Optional[ConfigInput] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: MessagesInput,
        stream: bool = False,
        system: Optional[SystemInput] = None,
        tools: Optional[Tools] = None,
        tool_choice: Optional[ToolChoiceInput] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        thinking: Optional[ConfigInput] = None,
        output_config: Optional[ConfigInput] = None,
        cache_control: Optional[ConfigInput] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[AnthropicMessage, AsyncMessageStream, "APIResponse"]:
        """Async twin of :meth:`Messages.create`."""
        body = _build_create_body(
            model,
            max_tokens,
            messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            thinking=thinking,
            output_config=output_config,
            cache_control=cache_control,
            stream=stream,
        )
        headers = {"anthropic-version": AnthropicProtocol.API_VERSION}
        options = request_options(
            extra_headers,
            extra_query,
            extra_body,
            max_retries=max_retries,
        )
        transport = self._client.transport
        base_url = _endpoint_base(self._client.config)
        if stream:
            if raw:
                raise ValueError("raw=True is not supported with stream=True")
            response = await transport.open_stream(
                "POST",
                _MESSAGES_PATH,
                json=body,
                headers=headers,
                base_url=base_url,
                api_key=api_key,
                workspace=workspace,
                sdk_module=MODULE,
                timeout=timeout,
                options=options,
            )
            return AsyncMessageStream(
                AsyncStream(
                    response,
                    functools.partial(
                        parse_message_stream_event,
                        strict=self._client.config.strict_validation,
                        response=response,
                    ),
                )
            )
        resp = await transport.request_api(
            "POST",
            _MESSAGES_PATH,
            json=body,
            headers=headers,
            base_url=base_url,
            api_key=api_key,
            workspace=workspace,
            sdk_module=MODULE,
            timeout=timeout,
            options=options,
            client=self._client,
        )
        if raw:
            return resp
        return _checked(self._client, resp, AnthropicMessage, parse_message)


__all__ = [
    "AsyncMessageStream",
    "AsyncMessages",
    "MessageStream",
    "Messages",
]