# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.multimodal.chat resource (design doc §2.1, M3 3.1).

The explicit escape hatch to the multimodal-generation endpoint: every
call goes there, media parts or not. ``chat.completions.create`` already
routes messages that carry media parts to this endpoint on its own, so
reach for this resource when a text-only conversation still has to hit
the multimodal service (a qwen-vl model, for one).

Compatible protocols have no separate media endpoint — they translate
parts on their own chat path — so there this resource and
``chat.completions`` are the same call.
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

from dashscope.resources._blocking import run_blocking
from dashscope.resources.chat.completions import (
    AsyncChatStream,
    ChatStream,
    _asend,
    _build_request,
    _send,
)
from dashscope.transport._options import request_options
from dashscope.types.completions import ChatCompletion
from dashscope.types.messages import MessageInput

if TYPE_CHECKING:
    from dashscope.transport._response import APIResponse

__all__ = [
    "AsyncMultimodalChat",
    "MultimodalChat",
]


class MultimodalChat:
    def __init__(self, client) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Optional[Union[str, List[MessageInput]]] = None,
        stream: Literal[True],
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
        incremental_output: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> ChatStream:
        ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Optional[Union[str, List[MessageInput]]] = None,
        stream: Literal[False] = False,
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
        incremental_output: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
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
        messages: Optional[Union[str, List[MessageInput]]] = None,
        stream: Literal[False] = False,
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
        incremental_output: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
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
        messages: Optional[Union[str, List[MessageInput]]] = None,
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
        incremental_output: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ChatCompletion, ChatStream, "APIResponse"]:
        """Run one multimodal-generation chat call.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself. It is only
        available with ``stream=False``: a streaming response body is never
        materialized, so there is nothing to hand back. Combining the two
        raises ``ValueError``.
        """
        built = _build_request(
            self._client,
            multimodal=True,
            model=model,
            messages=messages,
            stream=stream,
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
            incremental_output=incremental_output,
            parameters=parameters,
            extra_input=extra_input,
            api_key=api_key,
            workspace=workspace,
        )
        request, module, merge_incremental, server_cumulative = built
        return _send(
            self._client,
            request,
            stream=stream,
            module=module,
            merge_incremental=merge_incremental,
            server_cumulative=server_cumulative,
            api_key=api_key,
            workspace=workspace,
            timeout=timeout,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            raw=raw,
        )


class AsyncMultimodalChat:
    def __init__(self, client) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Optional[Union[str, List[MessageInput]]] = None,
        stream: Literal[True],
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
        incremental_output: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> AsyncChatStream:
        ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Optional[Union[str, List[MessageInput]]] = None,
        stream: Literal[False] = False,
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
        incremental_output: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
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
        messages: Optional[Union[str, List[MessageInput]]] = None,
        stream: Literal[False] = False,
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
        incremental_output: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
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
        messages: Optional[Union[str, List[MessageInput]]] = None,
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
        incremental_output: Optional[bool] = None,
        parameters: Optional[Dict[str, Any]] = None,
        extra_input: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ChatCompletion, AsyncChatStream, "APIResponse"]:
        """Run one multimodal-generation chat call.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself. It is only
        available with ``stream=False``: a streaming response body is never
        materialized, so there is nothing to hand back. Combining the two
        raises ``ValueError``.
        """
        # The upload path is blocking (file post + getPolicy), so the whole
        # request build runs in a worker thread -- over a transport bridged
        # back to this loop, which is what lets it serve the policy fetch.
        built = await run_blocking(
            self._client,
            _build_request,
            multimodal=True,
            model=model,
            messages=messages,
            stream=stream,
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
            incremental_output=incremental_output,
            parameters=parameters,
            extra_input=extra_input,
            api_key=api_key,
            workspace=workspace,
        )
        request, module, merge_incremental, server_cumulative = built
        return await _asend(
            self._client,
            request,
            stream=stream,
            module=module,
            merge_incremental=merge_incremental,
            server_cumulative=server_cumulative,
            api_key=api_key,
            workspace=workspace,
            timeout=timeout,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            raw=raw,
        )
