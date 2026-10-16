# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""chat.completions resource (protocol-driven, see v2/protocols).

Also the shared implementation behind ``multimodal.chat``: messages
carrying media parts route to the multimodal endpoint on their own.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, overload

from typing_extensions import Literal

from dashscope.utils.param_utils import ParamUtil
from dashscope.v2.errors import response_error
from dashscope.v2.protocols.base import ChatRequest
from dashscope.v2.resources._oss import (
    OSS_RESOLVE_HEADERS,
    upload_message_media,
)
from dashscope.v2.types.common import Usage, with_headers
from dashscope.v2.types.completions import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    SearchInfo,
    StreamChoice,
)
from dashscope.v2.transport.sse import AsyncStream, Stream
from dashscope.v2.types.messages import (
    Message,
    MessageInput,
    ToolCall,
    ToolCallFunction,
    has_media_parts,
    normalize_messages,
)

__all__ = [
    "AsyncChatStream",
    "AsyncCompletions",
    "ChatStream",
    "Completions",
]


def _endpoint(
    client,
    normalized: List[Message],
    *,
    multimodal: bool,
) -> Optional[str]:
    """Endpoint override for one call; None keeps the adapter default.

    ``multimodal`` forces the media endpoint, which is the whole of
    ``multimodal.chat``'s contract as an explicit escape hatch. Otherwise
    media parts route there on their own and text-only conversations stay
    on the text endpoint. Adapters without a separate media endpoint
    translate parts on their own chat path, so they always get None.
    """
    path = client.protocol.MULTIMODAL_PATH
    if path is None or multimodal:
        return path
    if has_media_parts(normalized):
        return path
    return None


def _build_request(
    client,
    *,
    multimodal: bool,
    model: str,
    messages: Union[str, List[MessageInput]],
    stream: bool,
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
    incremental_output: Optional[bool],
    parameters: Optional[Dict[str, Any]],
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
) -> Tuple[ChatRequest, str, bool, bool]:
    normalized = normalize_messages(messages)
    # 1.x tags the whole chat family "aigc" (MultiModalConversation lives
    # in dashscope.aigc); media routing does not change the module tag.
    module = "aigc"
    # incremental_output=False asks for cumulative snapshots. Most native
    # models only stream incrementally, so (like 1.x) the wire keeps
    # incremental_output=True and the SDK merges deltas client-side; the
    # exempt models (tts/omni/qwen-deep-research) honor the flag
    # server-side. Compatible protocols stream incremental-only, so the
    # cumulative view is always client-side there.
    merge_incremental = False
    if stream and incremental_output is not None:
        if client.protocol.name == "native":
            server_side = not ParamUtil.should_modify_incremental_output(model)
            if incremental_output is False and not server_side:
                merge_incremental = True
                # 1.x also forces the wire incremental while merging
                # client-side (Generation sets parameters[
                # "incremental_output"] = True), so an escape-hatch
                # parameters={"incremental_output": False} cannot ask the
                # server for cumulative frames the merge would double.
                parameters = {
                    **(parameters or {}),
                    "incremental_output": True,
                }
            else:
                parameters = {
                    **(parameters or {}),
                    "incremental_output": incremental_output,
                }
        elif incremental_output is False:
            merge_incremental = True
    # The merge block above decided the wire value of incremental_output
    # (native defaults it to True for streams and extra_parameters are
    # applied last), so the resource layer knows the exact frame shape.
    # A wire value of False means the server itself streams cumulative
    # snapshots (escape-hatch-only, or the named flag on exempt models);
    # the final message must then be each choice's last snapshot, not a
    # concatenation of deltas.
    server_cumulative = (
        stream
        and not merge_incremental
        and client.protocol.name == "native"
        and (parameters or {}).get("incremental_output") is False
    )
    # Local paths / file:// URIs in media parts upload to OSS first; only
    # the native protocol resolves oss:// inputs (1.x behavior). The
    # per-call key/workspace win so the upload lands in the same account
    # and region the request runs under (1.x uploads with the call-level
    # key too); the certificate is fetched from the client's own base_url.
    uploaded = False
    if client.protocol.name == "native":
        uploaded = upload_message_media(
            client,
            normalized,
            model,
            api_key,
            workspace,
        )
    request = client.protocol.build_chat_request(
        model=model,
        messages=[m.model_dump(exclude_none=True) for m in normalized],
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
        extra_parameters=parameters,
        path=_endpoint(client, normalized, multimodal=multimodal),
    )
    if uploaded:
        request = replace(
            request,
            headers={**request.headers, **OSS_RESOLVE_HEADERS},
        )
    return request, module, merge_incremental, server_cumulative


class ChatStream:
    """Typed iterator over chat chunks with text/aggregate helpers.

    With ``merge_incremental=True`` every yielded chunk is a cumulative
    snapshot (1.x ``incremental_output=False`` behavior), so
    ``text_stream`` emits the full text so far on each step.
    ``server_cumulative=True`` means the server itself streamed
    cumulative snapshots (native wire ``incremental_output=False``), so
    ``get_final_message`` takes each choice's last snapshot instead of
    concatenating deltas.
    """

    def __init__(
        self,
        inner: Stream[ChatCompletionChunk],
        *,
        merge_incremental: bool = False,
        server_cumulative: bool = False,
    ) -> None:
        self._inner = inner
        self._merge = merge_incremental
        self._server_cumulative = server_cumulative

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        if self._merge:
            return _accumulate_chunks(iter(self._inner))
        return iter(self._inner)

    @property
    def text_stream(self) -> Iterator[str]:
        for chunk in self:
            delta = chunk.delta
            if delta is not None and delta.content:
                yield delta.content

    @property
    def reasoning_stream(self) -> Iterator[str]:
        for chunk in self:
            delta = chunk.delta
            if delta is not None and delta.reasoning_content:
                yield delta.reasoning_content

    @property
    def tool_call_stream(self) -> Iterator[List[ToolCall]]:
        for chunk in self:
            delta = chunk.delta
            if delta is not None and delta.tool_calls:
                yield delta.tool_calls

    def get_final_message(self) -> ChatCompletion:
        if self._merge or self._server_cumulative:
            final = _final_from_cumulative(list(self))
        else:
            final = _aggregate(self)
        final.headers = self.headers
        return final

    @property
    def headers(self) -> Dict[str, str]:
        """Raw HTTP response headers of the stream's envelope."""
        return self._inner.headers

    def close(self) -> None:
        self._inner.close()


class AsyncChatStream:
    def __init__(
        self,
        inner: AsyncStream[ChatCompletionChunk],
        *,
        merge_incremental: bool = False,
        server_cumulative: bool = False,
    ) -> None:
        self._inner = inner
        self._merge = merge_incremental
        self._server_cumulative = server_cumulative

    def __aiter__(self):
        if self._merge:
            return self._accumulated()
        return self._inner.__aiter__()

    async def _accumulated(self):
        acc: Dict[int, Dict[str, Any]] = {}
        async for chunk in self._inner:
            yield _cumulative_chunk(acc, chunk)

    @property
    def text_stream(self):
        return self._text_stream()

    async def _text_stream(self):
        async for chunk in self:
            delta = chunk.delta
            if delta is not None and delta.content:
                yield delta.content

    @property
    def reasoning_stream(self):
        return self._reasoning_stream()

    async def _reasoning_stream(self):
        async for chunk in self:
            delta = chunk.delta
            if delta is not None and delta.reasoning_content:
                yield delta.reasoning_content

    @property
    def tool_call_stream(self):
        return self._tool_call_stream()

    async def _tool_call_stream(self):
        async for chunk in self:
            delta = chunk.delta
            if delta is not None and delta.tool_calls:
                yield delta.tool_calls

    async def get_final_message(self) -> ChatCompletion:
        chunks = [chunk async for chunk in self]
        if self._merge or self._server_cumulative:
            final = _final_from_cumulative(chunks)
        else:
            final = _aggregate(chunks)
        final.headers = self.headers
        return final

    @property
    def headers(self) -> Dict[str, str]:
        """Raw HTTP response headers of the stream's envelope."""
        return self._inner.headers

    async def aclose(self) -> None:
        await self._inner.aclose()


def _concat_logprobs(existing, new):
    """Concatenate streaming logprobs fragments.

    Each chunk's choice-level logprobs carries the entries for that
    chunk's tokens, so the aggregated value is the ordered
    concatenation of the list-valued keys (OpenAI shape:
    ``{"content": [...], "refusal": [...]}``), aligning with the
    concatenated content. Unknown shapes fall back to the latest
    fragment rather than risking corruption.
    """
    if existing is None:
        return new
    if new is None:
        return existing
    if isinstance(existing, dict) and isinstance(new, dict):
        merged = dict(existing)
        for key, value in new.items():
            if isinstance(value, list) and isinstance(merged.get(key), list):
                merged[key] = merged[key] + value
            else:
                merged[key] = value
        return merged
    return new


def _merge_tool_calls(
    existing: Optional[List[ToolCall]],
    delta_calls: List[ToolCall],
) -> List[ToolCall]:
    # OpenAI streams tool_calls as non-cumulative fragments addressed by
    # ``index``; native/anthropic chunks carry cumulative snapshots.
    if all(call.index is None for call in delta_calls):
        return [call.model_copy(deep=True) for call in delta_calls]
    merged = [call.model_copy(deep=True) for call in existing or []]
    for call in delta_calls:
        index = call.index or 0
        while len(merged) <= index:
            merged.append(ToolCall(function=ToolCallFunction()))
        target = merged[index]
        if call.id:
            target.id = call.id
        if call.type:
            target.type = call.type
        if call.function.name:
            target.function.name = (
                target.function.name or ""
            ) + call.function.name
        if call.function.arguments:
            target.function.arguments += call.function.arguments
    return merged


def _cumulative_chunk(
    acc: Dict[int, Dict[str, Any]],
    chunk: ChatCompletionChunk,
) -> ChatCompletionChunk:
    """Fold one incremental chunk into per-choice cumulative snapshots."""
    choices = []
    for choice in chunk.choices:
        slot = acc.setdefault(
            choice.index,
            {
                "role": None,
                "content": "",
                "reasoning": "",
                "signature": "",
                "tool_calls": None,
                "logprobs": None,
            },
        )
        delta = choice.delta
        if delta.role:
            slot["role"] = delta.role
        if delta.content:
            slot["content"] += delta.content
        if delta.reasoning_content:
            slot["reasoning"] += delta.reasoning_content
        if delta.reasoning_signature:
            slot["signature"] += delta.reasoning_signature
        if delta.tool_calls:
            slot["tool_calls"] = _merge_tool_calls(
                slot["tool_calls"],
                delta.tool_calls,
            )
        # Choice-level logprobs stream as per-frame fragments; the
        # cumulative snapshot carries their ordered concatenation,
        # mirroring the accumulated content.
        slot["logprobs"] = _concat_logprobs(
            slot["logprobs"],
            (choice.model_extra or {}).get("logprobs"),
        )
        # model_copy keeps provider extras the rebuild would otherwise
        # drop (delta-level omni audio); accumulated logprobs override
        # the current frame's fragment.
        update: Dict[str, Any] = {
            "delta": choice.delta.model_copy(
                update={
                    "role": slot["role"],
                    "content": slot["content"] or None,
                    "reasoning_content": slot["reasoning"] or None,
                    "reasoning_signature": slot["signature"] or None,
                    "tool_calls": slot["tool_calls"],
                },
            ),
        }
        if slot["logprobs"] is not None:
            update["logprobs"] = slot["logprobs"]
        choices.append(choice.model_copy(update=update))
    return chunk.model_copy(update={"choices": choices})


def _accumulate_chunks(
    chunks: Iterator[ChatCompletionChunk],
) -> Iterator[ChatCompletionChunk]:
    acc: Dict[int, Dict[str, Any]] = {}
    for chunk in chunks:
        yield _cumulative_chunk(acc, chunk)


def _final_from_cumulative(
    chunks: List[ChatCompletionChunk],
) -> ChatCompletion:
    """Final message for a cumulative stream: per choice, its last snapshot.

    The trailing frame may carry usage with an empty choices list (the
    openai adapter always sets ``stream_options.include_usage``), and with
    ``n > 1`` each choice's finish frame arrives separately, so the final
    message cannot be read off the last chunk alone.
    """
    if not chunks:
        return ChatCompletion()
    request_id = model = None
    usage: Optional[Usage] = None
    search_info: Optional[SearchInfo] = None
    last_by_index: Dict[int, StreamChoice] = {}
    for chunk in chunks:
        request_id = chunk.request_id or request_id
        model = chunk.model or model
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.search_info is not None:
            search_info = chunk.search_info
        for choice in chunk.choices:
            last_by_index[choice.index] = choice
    choices = []
    for index in sorted(last_by_index):
        choice = last_by_index[index]
        message = ChatCompletionMessage(
            role=choice.delta.role or "assistant",
            content=choice.delta.content,
            reasoning_content=choice.delta.reasoning_content,
            reasoning_signature=choice.delta.reasoning_signature,
            tool_calls=choice.delta.tool_calls,
            **(choice.delta.model_extra or {}),
        )
        choices.append(
            Choice(
                index=index,
                finish_reason=choice.finish_reason,
                message=message,
                **(choice.model_extra or {}),
            ),
        )
    return ChatCompletion(
        request_id=request_id,
        model=model,
        choices=choices,
        usage=usage,
        search_info=search_info,
    )


def _aggregate(chunks) -> ChatCompletion:  # pylint: disable=too-many-branches
    acc: Dict[int, Dict[str, Any]] = {}
    request_id = model = None
    usage: Optional[Usage] = None
    search_info: Optional[SearchInfo] = None
    for chunk in chunks:
        request_id = chunk.request_id or request_id
        model = chunk.model or model
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.search_info is not None:
            search_info = chunk.search_info
        for choice in chunk.choices:
            slot = acc.setdefault(
                choice.index,
                {
                    "content": [],
                    "reasoning": [],
                    "signature": [],
                    "message": {},
                    "logprobs": None,
                },
            )
            delta = choice.delta
            if delta.role:
                slot["message"]["role"] = delta.role
            if delta.content:
                slot["content"].append(delta.content)
            if delta.reasoning_content:
                slot["reasoning"].append(delta.reasoning_content)
            if delta.reasoning_signature:
                slot["signature"].append(delta.reasoning_signature)
            if delta.tool_calls:
                slot["message"]["tool_calls"] = _merge_tool_calls(
                    slot["message"].get("tool_calls"),
                    delta.tool_calls,
                )
            logprobs = (choice.model_extra or {}).get("logprobs")
            if logprobs is not None:
                slot["logprobs"] = _concat_logprobs(
                    slot["logprobs"],
                    logprobs,
                )
            if choice.finish_reason:
                slot["message"]["finish_reason"] = choice.finish_reason
    choices = []
    for index in sorted(acc):
        slot = acc[index]
        message = ChatCompletionMessage(
            role=slot["message"].get("role", "assistant"),
            content="".join(slot["content"]) or None,
            reasoning_content="".join(slot["reasoning"]) or None,
            reasoning_signature="".join(slot["signature"]) or None,
            tool_calls=slot["message"].get("tool_calls"),
        )
        extras: Dict[str, Any] = {}
        if slot["logprobs"] is not None:
            extras["logprobs"] = slot["logprobs"]
        choices.append(
            Choice(
                index=index,
                finish_reason=slot["message"].get("finish_reason"),
                message=message,
                **extras,
            ),
        )
    return ChatCompletion(
        request_id=request_id,
        model=model,
        choices=choices,
        usage=usage,
        search_info=search_info,
    )


def _checked_completion(
    client,
    response,
) -> Optional[ChatCompletion]:
    """Raise on failure, or return an error envelope in compat mode."""
    error = response_error(response)
    if error is None:
        return None
    if client.raise_on_error:
        raise error
    return ChatCompletion(
        request_id=error.request_id,
        status_code=error.status_code,
        code=error.code,
        message=error.message,
        is_error=True,
        headers=dict(response.headers),
    )


def _send(
    client,
    request: ChatRequest,
    *,
    stream: bool,
    module: str,
    merge_incremental: bool,
    server_cumulative: bool,
    api_key: Optional[str],
    workspace: Optional[str],
    timeout: Optional[float],
) -> Union[ChatCompletion, ChatStream]:
    transport = client.transport
    base_url = client.config.protocol_base_url
    if stream:
        response = transport.open_stream(
            "POST",
            request.path,
            json=request.body,
            headers=request.headers or None,
            base_url=base_url,
            api_key=api_key,
            workspace=workspace,
            sdk_module=module,
            timeout=timeout,
        )
        return ChatStream(
            Stream(response, client.protocol.make_chunk_parser()),
            merge_incremental=merge_incremental,
            server_cumulative=server_cumulative,
        )
    response = transport.request(
        "POST",
        request.path,
        json=request.body,
        headers=request.headers or None,
        base_url=base_url,
        api_key=api_key,
        workspace=workspace,
        sdk_module=module,
        timeout=timeout,
    )
    error_result = _checked_completion(client, response)
    if error_result is not None:
        return error_result
    return with_headers(
        client.protocol.parse_completion(response.json()),
        response,
    )


async def _asend(
    client,
    request: ChatRequest,
    *,
    stream: bool,
    module: str,
    merge_incremental: bool,
    server_cumulative: bool,
    api_key: Optional[str],
    workspace: Optional[str],
    timeout: Optional[float],
) -> Union[ChatCompletion, AsyncChatStream]:
    transport = client.transport
    base_url = client.config.protocol_base_url
    if stream:
        response = await transport.open_stream(
            "POST",
            request.path,
            json=request.body,
            headers=request.headers or None,
            base_url=base_url,
            api_key=api_key,
            workspace=workspace,
            sdk_module=module,
            timeout=timeout,
        )
        return AsyncChatStream(
            AsyncStream(response, client.protocol.make_chunk_parser()),
            merge_incremental=merge_incremental,
            server_cumulative=server_cumulative,
        )
    response = await transport.request(
        "POST",
        request.path,
        json=request.body,
        headers=request.headers or None,
        base_url=base_url,
        api_key=api_key,
        workspace=workspace,
        sdk_module=module,
        timeout=timeout,
    )
    error_result = _checked_completion(client, response)
    if error_result is not None:
        return error_result
    return with_headers(
        client.protocol.parse_completion(response.json()),
        response,
    )


class Completions:
    def __init__(self, client) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Union[str, List[MessageInput]],
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
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> ChatStream:
        ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Union[str, List[MessageInput]],
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
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> ChatCompletion:
        ...

    def create(
        self,
        *,
        model: str,
        messages: Union[str, List[MessageInput]],
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
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Union[ChatCompletion, ChatStream]:
        built = _build_request(
            self._client,
            multimodal=False,
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
        )


class AsyncCompletions:
    def __init__(self, client) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Union[str, List[MessageInput]],
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
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> AsyncChatStream:
        ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Union[str, List[MessageInput]],
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
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> ChatCompletion:
        ...

    async def create(
        self,
        *,
        model: str,
        messages: Union[str, List[MessageInput]],
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
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncChatStream]:
        # The upload path is blocking (requests + getPolicy); run the whole
        # request build off the event loop so it never stalls other tasks.
        built = await asyncio.to_thread(
            _build_request,
            self._client,
            multimodal=False,
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
        )
