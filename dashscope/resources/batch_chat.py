# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.batch_chat -- Batch Chat (queued, synchronous, discounted).

Batch Chat keeps the real-time Chat API's synchronous call shape but
routes to a dedicated batch gateway: the request queues, the connection
is held, and the complete result returns at once. It is OpenAI-compatible
(``POST {batch}/compatible-mode/v1/chat/completions``) regardless of the
client's chat ``protocol`` setting, so this resource always builds and
parses through the OpenAI adapter and derives its own batch base URL --
like ``client.decision``, the chat ``protocol`` cannot move it.

Single request only; for many requests in one file use ``client.batches``
(the asynchronous file-input Batch Job). Non-streaming: the server returns
the full completion once the request is scheduled, so there is no
``stream`` option here (use ``client.chat`` for streaming).

Billing/timeout semantics differ from real-time Chat: only successfully
executed requests are billed (a request that times out unscheduled is not
billed and may be retried); the catalog price matches real-time but Batch
Chat carries a limited-time 50% discount; it is an independent billing
item that does not support prepaid plans, new-user free quota or context
cache. The server holds the connection up to 3600s; a custom ``timeout``
must be within 60..3600s and defaults to 3600s.

Contract:
https://help.aliyun.com/zh/model-studio/openai-compatible-batch-chat
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
    overload,
)

from typing_extensions import Literal

from dashscope._config import Config
from dashscope.errors import response_error
from dashscope.protocols import get_protocol
from dashscope.protocols.openai import OpenAIProtocol
from dashscope.transport._options import request_options
from dashscope.types._parse import parse_model
from dashscope.types.common import with_headers
from dashscope.types.completions import ChatCompletion
from dashscope.types.messages import MessageInput, normalize_messages

if TYPE_CHECKING:
    from dashscope.transport._response import APIResponse

_PATH = "/chat/completions"
# Functional-area tag for x-dashscope-sdk-client (emitted as "batch_chat2").
MODULE = "batch_chat"

# The documented Batch Chat gateway. Batch Chat is served on this shared
# batch host (currently Beijing); the official contract documents no
# workspace/international batch host, so anything else must be supplied as
# an explicit base_url rather than invented here.
BATCH_GATEWAY_BASE = "https://batch.dashscope.aliyuncs.com/compatible-mode/v1"

# Server holds the connection up to 3600s; custom timeout range 60..3600s.
BATCH_DEFAULT_TIMEOUT = 3600.0
BATCH_MIN_TIMEOUT = 60.0
BATCH_MAX_TIMEOUT = 3600.0

# Batch Chat is always OpenAI-compatible, independent of the client's chat
# protocol. Response parsing is stateless, so one shared instance is safe
# there (strictness is passed per call). Request building is resolved per
# client instead, so protocol_options such as ``strict`` are honored.
_PROTOCOL = OpenAIProtocol()

__all__ = [
    "AsyncBatchChat",
    "AsyncBatchChatCompletions",
    "BatchChat",
    "BatchChatCompletions",
]


def _batch_base_url(config: Config) -> str:
    # An explicit base_url stays the service root (escape hatch for a
    # custom/workspace batch endpoint); otherwise the documented batch
    # gateway. The chat protocol setting must not move it.
    if config.configured_base_url is not None:
        return config.configured_base_url
    return BATCH_GATEWAY_BASE


def _resolve_timeout(timeout: Optional[float]) -> float:
    if timeout is None:
        return BATCH_DEFAULT_TIMEOUT
    if not BATCH_MIN_TIMEOUT <= timeout <= BATCH_MAX_TIMEOUT:
        raise ValueError(
            f"batch_chat timeout must be between {BATCH_MIN_TIMEOUT:.0f} "
            f"and {BATCH_MAX_TIMEOUT:.0f} seconds (got {timeout!r}); the "
            f"server holds the connection up to "
            f"{BATCH_MAX_TIMEOUT:.0f}s",
        )
    return float(timeout)


def _messages(
    messages: Union[str, List[MessageInput]],
) -> List[Dict[str, Any]]:
    normalized = normalize_messages(messages)
    if not normalized:
        raise ValueError("messages must not be empty")
    return [m.model_dump(exclude_none=True) for m in normalized]


def _build_body(
    *,
    protocol: OpenAIProtocol,
    model: str,
    messages: Union[str, List[MessageInput]],
    tools: Optional[List[Dict[str, Any]]],
    temperature: Optional[float],
    top_p: Optional[float],
    max_tokens: Optional[int],
    stop: Union[str, List[str], None],
    seed: Optional[int],
    repetition_penalty: Optional[float],
    presence_penalty: Optional[float],
    enable_thinking: Optional[bool],
    response_format: Optional[Dict[str, Any]],
    top_k: Optional[int],
    tool_choice: Union[str, Dict[str, Any], None],
    n: Optional[int],
    logprobs: Optional[bool],
    top_logprobs: Optional[int],
    thinking_budget: Optional[int],
    enable_search: Optional[bool],
    search_options: Optional[Dict[str, Any]],
    parallel_tool_calls: Optional[bool],
    parameters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not model or not model.strip():
        raise ValueError("model must be a non-empty string")
    request = protocol.build_chat_request(
        model=model,
        messages=_messages(messages),
        stream=False,
        tools=tools,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
        seed=seed,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        enable_thinking=enable_thinking,
        response_format=response_format,
        top_k=top_k,
        tool_choice=tool_choice,
        n=n,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        thinking_budget=thinking_budget,
        enable_search=enable_search,
        search_options=search_options,
        parallel_tool_calls=parallel_tool_calls,
        extra_parameters=parameters,
    )
    return request.body


def _checked(client, resp) -> ChatCompletion:
    """Raise on failure, or fill the error envelope when not raising.

    Batch Chat is OpenAI-compatible, so a failure nests ``code`` /
    ``message`` under ``error``; routing through ``response_error`` keeps
    ``raise_on_error=False`` reporting it via ``is_error`` / ``code`` /
    ``message`` instead of an empty success.
    """
    http_resp = resp.http_response
    error = response_error(http_resp)
    if error is None:
        return with_headers(
            _PROTOCOL.parse_completion(
                resp.json(),
                strict_validation=client.config.strict_validation,
                response=http_resp,
            ),
            http_resp,
        )
    if client.raise_on_error:
        raise error
    return with_headers(
        parse_model(
            ChatCompletion,
            {
                "request_id": error.request_id,
                "status_code": error.status_code,
                "code": error.code,
                "message": error.message,
                "is_error": True,
                "headers": dict(http_resp.headers),
            },
            strict=client.config.strict_validation,
            response=http_resp,
        ),
        http_resp,
    )


class BatchChatCompletions:
    """Synchronous Batch Chat calls (``client.batch_chat.completions``)."""

    def __init__(self, client) -> None:
        self._client = client
        self._protocol = get_protocol(
            "openai",
            protocol_options=client._protocol_options,
        )

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Union[str, List[MessageInput]],
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
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> ChatCompletion:
        ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Union[str, List[MessageInput]],
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
        parameters: Optional[Dict[str, Any]] = None,
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
        messages: Union[str, List[MessageInput]],
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
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ChatCompletion, "APIResponse"]:
        """Run one Batch Chat completion (queued, synchronous, discounted).

        Same parameters as ``client.chat.completions.create`` minus
        streaming: the request queues on the batch gateway and the full
        :class:`~dashscope.types.completions.ChatCompletion` returns once
        scheduled. ``timeout`` defaults to 3600s and must be within
        60..3600s (the server's max connection hold).

        Only successful requests are billed; a request that times out
        unscheduled is not billed and may be retried (``max_retries``
        follows the client default unless overridden). Batch Chat is an
        independent billing item at a limited-time 50% discount, with no
        prepaid plans, new-user free quota or context cache.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check.
        """
        body = _build_body(
            protocol=self._protocol,
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            enable_thinking=enable_thinking,
            response_format=response_format,
            top_k=top_k,
            tool_choice=tool_choice,
            n=n,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            thinking_budget=thinking_budget,
            enable_search=enable_search,
            search_options=search_options,
            parallel_tool_calls=parallel_tool_calls,
            parameters=parameters,
        )
        resp = self._client.transport.request_api(
            "POST",
            _PATH,
            json=body,
            base_url=_batch_base_url(self._client.config),
            sdk_module=MODULE,
            timeout=_resolve_timeout(timeout),
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
        return _checked(self._client, resp)


class AsyncBatchChatCompletions:
    """Asynchronous Batch Chat calls (``client.batch_chat.completions``)."""

    def __init__(self, client) -> None:
        self._client = client
        self._protocol = get_protocol(
            "openai",
            protocol_options=client._protocol_options,
        )

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Union[str, List[MessageInput]],
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
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> ChatCompletion:
        ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Union[str, List[MessageInput]],
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
        parameters: Optional[Dict[str, Any]] = None,
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
        messages: Union[str, List[MessageInput]],
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
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ChatCompletion, "APIResponse"]:
        """Async twin of :meth:`BatchChatCompletions.create`."""
        body = _build_body(
            protocol=self._protocol,
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            enable_thinking=enable_thinking,
            response_format=response_format,
            top_k=top_k,
            tool_choice=tool_choice,
            n=n,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            thinking_budget=thinking_budget,
            enable_search=enable_search,
            search_options=search_options,
            parallel_tool_calls=parallel_tool_calls,
            parameters=parameters,
        )
        resp = await self._client.transport.request_api(
            "POST",
            _PATH,
            json=body,
            base_url=_batch_base_url(self._client.config),
            sdk_module=MODULE,
            timeout=_resolve_timeout(timeout),
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
        return _checked(self._client, resp)


class BatchChat:
    """Batch Chat resource (``client.batch_chat``)."""

    def __init__(self, client) -> None:
        self.completions = BatchChatCompletions(client)


class AsyncBatchChat:
    """Async Batch Chat resource (``client.batch_chat``)."""

    def __init__(self, client) -> None:
        self.completions = AsyncBatchChatCompletions(client)
