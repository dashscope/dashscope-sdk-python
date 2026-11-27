# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.responses -- OpenAI-compatible Responses API (C1).

A stored-response lifecycle independent of chat: ``create`` / ``get`` /
``delete`` plus ``input_items.list``, and SSE streaming. The wire is a flat
OpenAI ``response`` object whose ``output`` array carries ten item types
(message, reasoning, function_call and the built-in tool calls), so it is
modelled in :mod:`dashscope.types.responses` rather than reusing chat's
choice/message shape.

Like ``client.batches`` / ``client.decision``, every call derives its own
base URL via ``compatible_mode_base_url`` (``{gateway}/compatible-mode/v1``)
so the chat ``protocol`` setting cannot move it, and tags ``sdk_module``
``responses`` for the ``x-dashscope-sdk-client`` header.

State lives on the server (a stored response id is valid for 7 days and can
seed ``previous_response_id``); the SDK stays a stateless wrapper -- no
client-side history or memory.

Contract:
https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-responses
"""

from __future__ import annotations

import functools
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
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
from urllib.parse import quote

from pydantic import BaseModel

from dashscope._config import Config, compatible_mode_base_url
from dashscope.errors import DashScopeError, response_error
from dashscope.resources.batches import _CursorPage
from dashscope.transport._options import request_options
from dashscope.transport.sse import AsyncStream, Stream
from dashscope.types._parse import parse_model
from dashscope.types.common import response_envelope, with_headers
from dashscope.types.responses import (
    InputItem,
    Response,
    ResponseDeleted,
    ResponseStreamEvent,
    parse_response,
    parse_response_deleted,
    parse_stream_event,
)

if TYPE_CHECKING:
    import httpx

    from dashscope.transport._response import APIResponse

__all__ = [
    "AsyncInputItems",
    "AsyncResponseStream",
    "AsyncResponses",
    "InputItemList",
    "InputItems",
    "ResponseStream",
    "Responses",
]

_RESPONSES_PATH = "/responses"

MODULE = "responses"

# input_items caps limit at 100 (default 20); order defaults to desc.
_MAX_LIMIT = 100

_SESSION_CACHE_HEADER = "x-dashscope-session-cache"

# Terminal stream events carry the full response object.
_TERMINAL_EVENTS = frozenset(
    {"response.completed", "response.failed", "response.incomplete"},
)

_ENABLE_THINKING_WARNED = False

Input = Union[str, Sequence[Union[Mapping[str, Any], BaseModel]]]
Tools = Sequence[Union[Mapping[str, Any], BaseModel]]
ConfigInput = Union[Mapping[str, Any], BaseModel]


def _endpoint_base(config: Config) -> str:
    # The compatible-mode Responses API sits under /compatible-mode/v1 on
    # every gateway; the chat protocol setting must not move it.
    return compatible_mode_base_url(
        config.workspace,
        config.region,
        config.configured_base_url,
    )


def _response_id(value: Union[Response, str]) -> str:
    rid = value.id if isinstance(value, Response) else value
    if not rid or not str(rid).strip():
        raise ValueError("response_id must be a non-empty string")
    # Percent-encode so a '/', '?' or '#' inside the id cannot retarget
    # the request to another path or query.
    return quote(str(rid), safe="")


def _normalize_config(
    value: Optional[ConfigInput],
) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    return dict(value)


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
                "each item must be a mapping or a pydantic model",
            )
    return items


def _normalize_input(value: Input) -> Union[str, List[Dict[str, Any]]]:
    if isinstance(value, str):
        return value
    return _normalize_items(value)


def _validate_function_call_sequence(items: List[Dict[str, Any]]) -> None:
    # The server rejects a function_call_output that does not immediately
    # follow its matching function_call (same call_id); mirror that here so
    # the error surfaces before the round-trip.
    prev: Optional[Dict[str, Any]] = None
    for item in items:
        if item.get("type") == "function_call_output":
            if (
                prev is None
                or prev.get("type") != "function_call"
                or prev.get("call_id") != item.get("call_id")
            ):
                raise ValueError(
                    "each function_call_output must immediately follow its "
                    "matching function_call (same call_id) in the input array",
                )
        prev = item


def _warn_enable_thinking() -> None:
    global _ENABLE_THINKING_WARNED  # pylint: disable=global-statement
    if _ENABLE_THINKING_WARNED:
        return
    _ENABLE_THINKING_WARNED = True
    warnings.warn(
        "enable_thinking is deprecated on the Responses API; use "
        "reasoning={'effort': ...} instead",
        category=DeprecationWarning,
        stacklevel=3,
    )


def _session_headers(session_cache: Optional[bool]) -> Dict[str, str]:
    if session_cache is None:
        return {}
    return {_SESSION_CACHE_HEADER: "enable" if session_cache else "disable"}


def _build_create_body(
    model: str,
    input: Input,  # pylint: disable=redefined-builtin
    *,
    instructions: Optional[str],
    previous_response_id: Optional[str],
    conversation: Optional[str],
    store: Optional[bool],
    tools: Optional[Tools],
    tool_choice: Union[str, Mapping[str, Any], None],
    parallel_tool_calls: Optional[bool],
    temperature: Optional[float],
    top_p: Optional[float],
    max_output_tokens: Optional[int],
    reasoning: Optional[ConfigInput],
    prompt_cache_options: Optional[ConfigInput],
    ocr_options: Optional[ConfigInput],
    enable_thinking: Optional[bool],
    text: Optional[ConfigInput],
    stream: bool,
) -> Dict[str, Any]:
    if not model or not model.strip():
        raise ValueError("model must be a non-empty string")
    if input is None:
        raise ValueError("input is required")
    if previous_response_id is not None and conversation is not None:
        raise ValueError(
            "previous_response_id and conversation are mutually exclusive",
        )
    normalized_input = _normalize_input(input)
    if isinstance(normalized_input, list):
        _validate_function_call_sequence(normalized_input)
    if enable_thinking is not None:
        _warn_enable_thinking()
    body: Dict[str, Any] = {
        "model": model,
        "input": normalized_input,
        "stream": stream,
    }
    optional = {
        "instructions": instructions,
        "previous_response_id": previous_response_id,
        "conversation": conversation,
        "store": store,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "enable_thinking": enable_thinking,
    }
    body.update({k: v for k, v in optional.items() if v is not None})
    if tools is not None:
        body["tools"] = _normalize_items(tools)
    for key, value in (
        ("reasoning", reasoning),
        ("prompt_cache_options", prompt_cache_options),
        ("ocr_options", ocr_options),
        ("text", text),
    ):
        normalized = _normalize_config(value)
        if normalized is not None:
            body[key] = normalized
    return body


def _input_items_params(
    *,
    after: Optional[str],
    limit: Optional[int],
    order: Optional[str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if after is not None:
        params["after"] = after
    if limit is not None:
        if not 1 <= limit <= _MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {_MAX_LIMIT}")
        params["limit"] = limit
    if order is not None:
        if order not in ("asc", "desc"):
            raise ValueError("order must be 'asc' or 'desc'")
        params["order"] = order
    return params


def _checked(client, resp, model_cls, parser):
    """Raise on failure, or fill the error envelope when not raising.

    Compatible-mode failures nest ``code`` / ``message`` under ``error``
    (OpenAI shape), which the flat success parser cannot see; routing the
    reply through ``response_error`` keeps ``raise_on_error=False``
    reporting the failure via ``is_error`` / ``code`` / ``message``.
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


class InputItemList(_CursorPage):
    """One cursor page of ``input_items``.

    The wire carries ``first_id`` / ``last_id`` (item ids, ``msg_xxx``); the
    next-page cursor prefers ``last_id``. ``id`` is the parent response id,
    ``model`` the model that produced it, and ``created_at`` is Unix
    **milliseconds** (unlike the response object's seconds).
    """

    data: List[InputItem] = []
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    id: Optional[str] = None
    model: Optional[str] = None
    created_at: Optional[int] = None

    @property
    def items(self) -> List[InputItem]:
        return self.data

    @property
    def _next_cursor(self) -> Optional[str]:
        if self.last_id:
            return self.last_id
        return self.data[-1].id if self.data else None

    def __iter__(self) -> Iterator[InputItem]:
        return iter(self.data)

    def __aiter__(self) -> AsyncIterator[InputItem]:
        async def _gen() -> AsyncIterator[InputItem]:
            for item in self.data:
                yield item

        return _gen()


def _parse_input_item_list(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> InputItemList:
    payload = dict(body) if isinstance(body, dict) else {}
    data = payload.get("data")
    if not isinstance(data, list):
        data = []
    return parse_model(
        InputItemList,
        {
            "data": data,
            "first_id": payload.get("first_id"),
            "last_id": payload.get("last_id"),
            "id": payload.get("id"),
            "model": payload.get("model"),
            "created_at": payload.get("created_at"),
            "has_more": bool(payload.get("has_more", False)),
            **response_envelope(payload),
        },
        strict=strict,
        response=response,
    )


class ResponseStream:
    """Typed iterator over Responses SSE events.

    Consume it one way: iterate the events, read ``text_stream`` /
    ``reasoning_stream``, or call ``get_final_response()`` -- each reads the
    live socket once. The terminal event (``response.completed`` /
    ``failed`` / ``incomplete``) carries the full :class:`Response`.
    """

    def __init__(self, inner: Stream[ResponseStreamEvent]) -> None:
        self._inner = inner

    def __iter__(self) -> Iterator[ResponseStreamEvent]:
        return iter(self._inner)

    @property
    def text_stream(self) -> Iterator[str]:
        for event in self:
            if event.type == "response.output_text.delta" and event.delta:
                yield event.delta

    @property
    def reasoning_stream(self) -> Iterator[str]:
        for event in self:
            if (
                event.type
                in (
                    "response.reasoning_text.delta",
                    "response.reasoning_summary_text.delta",
                )
                and event.delta
            ):
                yield event.delta

    def get_final_response(self) -> Response:
        final: Optional[Response] = None
        for event in self:
            if event.type in _TERMINAL_EVENTS and event.response is not None:
                final = event.response
        if final is None:
            raise DashScopeError(
                "response stream ended without a terminal response event",
            )
        final.headers = self.headers
        return final

    @property
    def headers(self) -> Dict[str, str]:
        """Raw HTTP response headers of the stream's envelope."""
        return self._inner.headers

    def close(self) -> None:
        self._inner.close()


class AsyncResponseStream:
    """Async twin of :class:`ResponseStream`."""

    def __init__(self, inner: AsyncStream[ResponseStreamEvent]) -> None:
        self._inner = inner

    def __aiter__(self):
        return self._inner.__aiter__()

    @property
    def text_stream(self):
        return self._text_stream()

    async def _text_stream(self):
        async for event in self:
            if event.type == "response.output_text.delta" and event.delta:
                yield event.delta

    @property
    def reasoning_stream(self):
        return self._reasoning_stream()

    async def _reasoning_stream(self):
        async for event in self:
            if (
                event.type
                in (
                    "response.reasoning_text.delta",
                    "response.reasoning_summary_text.delta",
                )
                and event.delta
            ):
                yield event.delta

    async def get_final_response(self) -> Response:
        final: Optional[Response] = None
        async for event in self:
            if event.type in _TERMINAL_EVENTS and event.response is not None:
                final = event.response
        if final is None:
            raise DashScopeError(
                "response stream ended without a terminal response event",
            )
        final.headers = self.headers
        return final

    @property
    def headers(self) -> Dict[str, str]:
        """Raw HTTP response headers of the stream's envelope."""
        return self._inner.headers

    async def aclose(self) -> None:
        await self._inner.aclose()


class InputItems:
    """``client.responses.input_items`` -- list a response's input items."""

    def __init__(self, client) -> None:
        self._client = client

    def list(  # pylint: disable=redefined-builtin
        self,
        response_id: Union[Response, str],
        *,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[InputItemList, "APIResponse"]:
        """List input items, one cursor page (``GET /responses/{id}/input_items``).

        ``after`` is an item id (``msg_xxx``) -- the previous page's
        ``last_id``; ``limit`` is in [1, 100] (default 20); ``order`` is
        ``asc`` / ``desc`` (default ``desc``). Cross pages with
        ``iter_pages()`` / ``next_page()`` on the parsed result.
        """
        rid = _response_id(response_id)
        params = _input_items_params(after=after, limit=limit, order=order)
        resp = self._client.transport.request_api(
            "GET",
            f"{_RESPONSES_PATH}/{rid}/input_items",
            params=params,
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        result = _checked(
            self._client,
            resp,
            InputItemList,
            _parse_input_item_list,
        )
        if not result.is_error:
            result.bind_pagination(
                lambda cursor: self.list(
                    response_id,
                    after=cursor,
                    limit=limit,
                    order=order,
                    timeout=timeout,
                    api_key=api_key,
                    workspace=workspace,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    max_retries=max_retries,
                ),
            )
        return result


class AsyncInputItems:
    """Async twin of :class:`InputItems`."""

    def __init__(self, client) -> None:
        self._client = client

    async def list(  # pylint: disable=redefined-builtin
        self,
        response_id: Union[Response, str],
        *,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[InputItemList, "APIResponse"]:
        """Async twin of :meth:`InputItems.list`."""
        rid = _response_id(response_id)
        params = _input_items_params(after=after, limit=limit, order=order)
        resp = await self._client.transport.request_api(
            "GET",
            f"{_RESPONSES_PATH}/{rid}/input_items",
            params=params,
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        result = _checked(
            self._client,
            resp,
            InputItemList,
            _parse_input_item_list,
        )
        if not result.is_error:
            result.bind_pagination(
                lambda cursor: self.list(
                    response_id,
                    after=cursor,
                    limit=limit,
                    order=order,
                    timeout=timeout,
                    api_key=api_key,
                    workspace=workspace,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    max_retries=max_retries,
                ),
            )
        return result


class Responses:
    """``client.responses`` -- the OpenAI-compatible Responses lifecycle."""

    def __init__(self, client) -> None:
        self._client = client
        self.input_items = InputItems(client)

    @overload
    def create(  # pylint: disable=redefined-builtin
        self,
        *,
        model: str,
        input: Input,
        stream: Literal[True],
        instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        conversation: Optional[str] = None,
        store: Optional[bool] = None,
        tools: Optional[Tools] = None,
        tool_choice: Union[str, Mapping[str, Any], None] = None,
        parallel_tool_calls: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[ConfigInput] = None,
        prompt_cache_options: Optional[ConfigInput] = None,
        ocr_options: Optional[ConfigInput] = None,
        enable_thinking: Optional[bool] = None,
        text: Optional[ConfigInput] = None,
        session_cache: Optional[bool] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> ResponseStream:
        ...

    @overload
    def create(  # pylint: disable=redefined-builtin
        self,
        *,
        model: str,
        input: Input,
        stream: Literal[False] = False,
        instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        conversation: Optional[str] = None,
        store: Optional[bool] = None,
        tools: Optional[Tools] = None,
        tool_choice: Union[str, Mapping[str, Any], None] = None,
        parallel_tool_calls: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[ConfigInput] = None,
        prompt_cache_options: Optional[ConfigInput] = None,
        ocr_options: Optional[ConfigInput] = None,
        enable_thinking: Optional[bool] = None,
        text: Optional[ConfigInput] = None,
        session_cache: Optional[bool] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Response:
        ...

    @overload
    def create(  # pylint: disable=redefined-builtin
        self,
        *,
        model: str,
        input: Input,
        stream: Literal[False] = False,
        instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        conversation: Optional[str] = None,
        store: Optional[bool] = None,
        tools: Optional[Tools] = None,
        tool_choice: Union[str, Mapping[str, Any], None] = None,
        parallel_tool_calls: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[ConfigInput] = None,
        prompt_cache_options: Optional[ConfigInput] = None,
        ocr_options: Optional[ConfigInput] = None,
        enable_thinking: Optional[bool] = None,
        text: Optional[ConfigInput] = None,
        session_cache: Optional[bool] = None,
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

    def create(  # pylint: disable=redefined-builtin
        self,
        *,
        model: str,
        input: Input,
        stream: bool = False,
        instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        conversation: Optional[str] = None,
        store: Optional[bool] = None,
        tools: Optional[Tools] = None,
        tool_choice: Union[str, Mapping[str, Any], None] = None,
        parallel_tool_calls: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[ConfigInput] = None,
        prompt_cache_options: Optional[ConfigInput] = None,
        ocr_options: Optional[ConfigInput] = None,
        enable_thinking: Optional[bool] = None,
        text: Optional[ConfigInput] = None,
        session_cache: Optional[bool] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Response, ResponseStream, "APIResponse"]:
        """Create a stored response (``POST /responses``).

        ``input`` is a string or a sequence of typed input items.
        ``previous_response_id`` and ``conversation`` are mutually exclusive
        (server-side context); a ``function_call_output`` item must
        immediately follow its matching ``function_call``. Bailian
        extensions are first-class: ``reasoning`` (effort/summary),
        ``prompt_cache_options``, ``ocr_options``, ``text`` (structured
        output) and ``session_cache`` (-> ``x-dashscope-session-cache``
        header). ``enable_thinking`` is deprecated in favour of
        ``reasoning.effort`` and emits a ``DeprecationWarning``.

        ``stream=True`` returns a :class:`ResponseStream`; ``raw=True``
        returns the :class:`~dashscope.transport.APIResponse` before any
        error check (only with ``stream=False``; combining raises
        ``ValueError``).
        """
        body = _build_create_body(
            model,
            input,
            instructions=instructions,
            previous_response_id=previous_response_id,
            conversation=conversation,
            store=store,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            reasoning=reasoning,
            prompt_cache_options=prompt_cache_options,
            ocr_options=ocr_options,
            enable_thinking=enable_thinking,
            text=text,
            stream=stream,
        )
        headers = _session_headers(session_cache)
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
                _RESPONSES_PATH,
                json=body,
                headers=headers or None,
                base_url=base_url,
                api_key=api_key,
                workspace=workspace,
                sdk_module=MODULE,
                timeout=timeout,
                options=options,
            )
            return ResponseStream(
                Stream(
                    response,
                    functools.partial(
                        parse_stream_event,
                        strict=self._client.config.strict_validation,
                        response=response,
                    ),
                ),
            )
        resp = transport.request_api(
            "POST",
            _RESPONSES_PATH,
            json=body,
            headers=headers or None,
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
        return _checked(self._client, resp, Response, parse_response)

    def get(
        self,
        response_id: Union[Response, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Response, "APIResponse"]:
        """Retrieve a stored response (``GET /responses/{id}``)."""
        rid = _response_id(response_id)
        resp = self._client.transport.request_api(
            "GET",
            f"{_RESPONSES_PATH}/{rid}",
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(self._client, resp, Response, parse_response)

    def delete(
        self,
        response_id: Union[Response, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ResponseDeleted, "APIResponse"]:
        """Delete a stored response (``DELETE /responses/{id}``).

        The wire returns ``{id, deleted}`` (no ``object`` field).
        """
        rid = _response_id(response_id)
        resp = self._client.transport.request_api(
            "DELETE",
            f"{_RESPONSES_PATH}/{rid}",
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(
            self._client,
            resp,
            ResponseDeleted,
            parse_response_deleted,
        )


class AsyncResponses:
    """Async twin of :class:`Responses`."""

    def __init__(self, client) -> None:
        self._client = client
        self.input_items = AsyncInputItems(client)

    @overload
    async def create(  # pylint: disable=redefined-builtin
        self,
        *,
        model: str,
        input: Input,
        stream: Literal[True],
        instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        conversation: Optional[str] = None,
        store: Optional[bool] = None,
        tools: Optional[Tools] = None,
        tool_choice: Union[str, Mapping[str, Any], None] = None,
        parallel_tool_calls: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[ConfigInput] = None,
        prompt_cache_options: Optional[ConfigInput] = None,
        ocr_options: Optional[ConfigInput] = None,
        enable_thinking: Optional[bool] = None,
        text: Optional[ConfigInput] = None,
        session_cache: Optional[bool] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> AsyncResponseStream:
        ...

    @overload
    async def create(  # pylint: disable=redefined-builtin
        self,
        *,
        model: str,
        input: Input,
        stream: Literal[False] = False,
        instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        conversation: Optional[str] = None,
        store: Optional[bool] = None,
        tools: Optional[Tools] = None,
        tool_choice: Union[str, Mapping[str, Any], None] = None,
        parallel_tool_calls: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[ConfigInput] = None,
        prompt_cache_options: Optional[ConfigInput] = None,
        ocr_options: Optional[ConfigInput] = None,
        enable_thinking: Optional[bool] = None,
        text: Optional[ConfigInput] = None,
        session_cache: Optional[bool] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Response:
        ...

    @overload
    async def create(  # pylint: disable=redefined-builtin
        self,
        *,
        model: str,
        input: Input,
        stream: Literal[False] = False,
        instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        conversation: Optional[str] = None,
        store: Optional[bool] = None,
        tools: Optional[Tools] = None,
        tool_choice: Union[str, Mapping[str, Any], None] = None,
        parallel_tool_calls: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[ConfigInput] = None,
        prompt_cache_options: Optional[ConfigInput] = None,
        ocr_options: Optional[ConfigInput] = None,
        enable_thinking: Optional[bool] = None,
        text: Optional[ConfigInput] = None,
        session_cache: Optional[bool] = None,
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

    async def create(  # pylint: disable=redefined-builtin
        self,
        *,
        model: str,
        input: Input,
        stream: bool = False,
        instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        conversation: Optional[str] = None,
        store: Optional[bool] = None,
        tools: Optional[Tools] = None,
        tool_choice: Union[str, Mapping[str, Any], None] = None,
        parallel_tool_calls: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[ConfigInput] = None,
        prompt_cache_options: Optional[ConfigInput] = None,
        ocr_options: Optional[ConfigInput] = None,
        enable_thinking: Optional[bool] = None,
        text: Optional[ConfigInput] = None,
        session_cache: Optional[bool] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Response, AsyncResponseStream, "APIResponse"]:
        """Async twin of :meth:`Responses.create`."""
        body = _build_create_body(
            model,
            input,
            instructions=instructions,
            previous_response_id=previous_response_id,
            conversation=conversation,
            store=store,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            reasoning=reasoning,
            prompt_cache_options=prompt_cache_options,
            ocr_options=ocr_options,
            enable_thinking=enable_thinking,
            text=text,
            stream=stream,
        )
        headers = _session_headers(session_cache)
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
                _RESPONSES_PATH,
                json=body,
                headers=headers or None,
                base_url=base_url,
                api_key=api_key,
                workspace=workspace,
                sdk_module=MODULE,
                timeout=timeout,
                options=options,
            )
            return AsyncResponseStream(
                AsyncStream(
                    response,
                    functools.partial(
                        parse_stream_event,
                        strict=self._client.config.strict_validation,
                        response=response,
                    ),
                ),
            )
        resp = await transport.request_api(
            "POST",
            _RESPONSES_PATH,
            json=body,
            headers=headers or None,
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
        return _checked(self._client, resp, Response, parse_response)

    async def get(
        self,
        response_id: Union[Response, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Response, "APIResponse"]:
        """Async twin of :meth:`Responses.get`."""
        rid = _response_id(response_id)
        resp = await self._client.transport.request_api(
            "GET",
            f"{_RESPONSES_PATH}/{rid}",
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(self._client, resp, Response, parse_response)

    async def delete(
        self,
        response_id: Union[Response, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ResponseDeleted, "APIResponse"]:
        """Async twin of :meth:`Responses.delete`."""
        rid = _response_id(response_id)
        resp = await self._client.transport.request_api(
            "DELETE",
            f"{_RESPONSES_PATH}/{rid}",
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(
            self._client,
            resp,
            ResponseDeleted,
            parse_response_deleted,
        )
