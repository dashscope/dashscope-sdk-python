# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.apps resource (design doc M3 3.8).

Bailian application completion on the v2 transport:
``POST /apps/{app_id}/completion``; streaming uses the DashScope SSE
convention (``X-DashScope-SSE: enable`` header + ``data:`` frames).
"""

from __future__ import annotations

import copy
import json
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from pydantic import BaseModel, ConfigDict

from dashscope.v2.errors import (
    check_response,
    error_from_business_code,
)
from dashscope.v2.types.common import (
    ResponseBase,
    Usage,
    flatten_envelope,
    with_headers,
)

__all__ = [
    "AppCompletion",
    "AppCompletions",
    "AppModelUsage",
    "AppStream",
    "AppUsage",
    "Apps",
    "AsyncAppCompletions",
    "AsyncAppStream",
    "AsyncApps",
    "DocReference",
    "Thought",
    "WorkflowMessage",
]

_SSE_HEADER = {"X-DashScope-SSE": "enable"}

MODULE = "app"


class Thought(BaseModel):
    """One planning or tool-call step; set ``has_thoughts`` to get them."""

    model_config = ConfigDict(extra="allow")

    thought: Optional[str] = None
    action_type: Optional[str] = None
    action_name: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    action_input_stream: Optional[str] = None
    response: Optional[str] = None
    observation: Optional[str] = None


class DocReference(BaseModel):
    """One retrieved document behind the answer (RAG citation)."""

    model_config = ConfigDict(extra="allow")

    index_id: Optional[str] = None
    doc_id: Optional[str] = None
    doc_name: Optional[str] = None
    doc_url: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None
    biz_id: Optional[str] = None
    images: List[str] = []
    page_number: List[int] = []


class WorkflowMessage(BaseModel):
    """Workflow node progress, from ``output.workflow_message``."""

    model_config = ConfigDict(extra="allow")

    class Message(BaseModel):
        model_config = ConfigDict(extra="allow")

        role: Optional[str] = None
        content: Optional[str] = None

    node_id: Optional[str] = None
    node_name: Optional[str] = None
    node_type: Optional[str] = None
    node_status: Optional[str] = None
    # The wire spells the completion flag as a string, not a bool.
    node_is_completed: Optional[str] = None
    node_msg_seq_id: Optional[int] = None
    message: Optional[Message] = None


class AppModelUsage(BaseModel):
    """Per-model token detail of one entry in ``usage.models``."""

    model_config = ConfigDict(extra="allow")

    model_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0


class AppUsage(Usage):
    """App usage: the shared counters plus per-model detail."""

    models: List[AppModelUsage] = []


class AppCompletion(ResponseBase):
    text: Optional[str] = None
    finish_reason: Optional[str] = None
    session_id: Optional[str] = None
    thoughts: Optional[List[Thought]] = None
    doc_references: Optional[List[DocReference]] = None
    workflow_message: Optional[WorkflowMessage] = None
    usage: Optional[AppUsage] = None


def _parse(body: Dict[str, Any]) -> AppCompletion:
    # An SSE frame is not guaranteed to be JSON; _try_json hands back the raw
    # string in that case.
    payload = flatten_envelope(body) if isinstance(body, dict) else {}
    return AppCompletion(**payload)


def _try_json(data: str) -> Any:
    try:
        return json.loads(data)
    except (TypeError, ValueError):
        return data


def _build_input(
    prompt: Optional[str],
    messages: Optional[List[Dict[str, Any]]],
    session_id: Optional[str],
    doc_reference_type: Optional[str],
    doc_tag_codes: Optional[List[str]],
    memory_id: Optional[str],
    biz_params: Optional[Dict[str, Any]],
    image_list: Optional[List[str]],
    file_list: Optional[List[str]],
) -> Dict[str, Any]:
    if messages is None:
        input_param: Dict[str, Any] = {"prompt": prompt}
    else:
        msgs = copy.deepcopy(messages)
        if prompt:
            msgs.append({"role": "user", "content": prompt})
        input_param = {"messages": msgs}
    if session_id:
        input_param["session_id"] = session_id
    if doc_reference_type:
        input_param["doc_reference_type"] = doc_reference_type
    if doc_tag_codes is not None:
        if not (
            isinstance(doc_tag_codes, list)
            and all(isinstance(item, str) for item in doc_tag_codes)
        ):
            raise ValueError("doc_tag_codes must be a list of strings")
        input_param["doc_tag_codes"] = doc_tag_codes
    if memory_id:
        input_param["memory_id"] = memory_id
    if biz_params:
        input_param["biz_params"] = biz_params
    if image_list:
        input_param["image_list"] = image_list
    if file_list:
        input_param["file_list"] = file_list
    return input_param


def _build_parameters(
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    has_thoughts: Optional[bool],
    rag_options: Optional[Dict[str, Any]],
    incremental_output: Optional[bool],
    parameters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if temperature is not None:
        merged["temperature"] = temperature
    if top_p is not None:
        merged["top_p"] = top_p
    if top_k is not None:
        merged["top_k"] = top_k
    if seed is not None:
        merged["seed"] = seed
    if has_thoughts is not None:
        merged["has_thoughts"] = has_thoughts
    if rag_options is not None:
        merged["rag_options"] = rag_options
    if incremental_output is not None:
        merged["incremental_output"] = incremental_output
    if parameters:
        merged.update(parameters)
    return merged


def _build_body(
    prompt: Optional[str],
    messages: Optional[List[Dict[str, Any]]],
    session_id: Optional[str],
    doc_reference_type: Optional[str],
    doc_tag_codes: Optional[List[str]],
    memory_id: Optional[str],
    biz_params: Optional[Dict[str, Any]],
    image_list: Optional[List[str]],
    file_list: Optional[List[str]],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    has_thoughts: Optional[bool],
    rag_options: Optional[Dict[str, Any]],
    incremental_output: Optional[bool],
    parameters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "input": _build_input(
            prompt,
            messages,
            session_id,
            doc_reference_type,
            doc_tag_codes,
            memory_id,
            biz_params,
            image_list,
            file_list,
        ),
        "parameters": _build_parameters(
            temperature,
            top_p,
            top_k,
            seed,
            has_thoughts,
            rag_options,
            incremental_output,
            parameters,
        ),
    }


def _validate(app_id: Optional[str], prompt, messages) -> None:
    if not app_id:
        raise ValueError("app_id is required")
    if not prompt and not messages:
        raise ValueError("prompt or messages is required")


def _sse_frames(response) -> Iterator[Any]:
    is_error = False
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith("event:error"):
            is_error = True
        elif line.startswith("data:"):
            yield is_error, line[len("data:") :].strip()
            if is_error:
                break


def _raise_frame_error(data: str) -> None:
    body = _try_json(data)
    payload = body if isinstance(body, dict) else {}
    raise error_from_business_code(
        message=payload.get("message") or data,
        code=payload.get("code"),
        body=body,
    )


def _error_event(data: str) -> AppCompletion:
    # Non-raising mode: surface the stream error as a terminal event
    # instead of silently truncating the stream.
    body = _try_json(data)
    payload = body if isinstance(body, dict) else {}
    return AppCompletion(
        request_id=payload.get("request_id"),
        code=payload.get("code"),
        message=payload.get("message") or (None if payload else str(body)),
        is_error=True,
    )


class AppStream:
    """Typed iterator over app completion events, mirroring ChatStream."""

    def __init__(self, inner: Iterator[AppCompletion], response: Any) -> None:
        self._inner = inner
        self._response = response

    def __iter__(self) -> Iterator[AppCompletion]:
        return iter(self._inner)

    @property
    def headers(self) -> Dict[str, str]:
        """Raw HTTP response headers of the stream's envelope."""
        return dict(self._response.headers)

    @property
    def text_stream(self) -> Iterator[str]:
        for event in self._inner:
            if event.text:
                yield event.text

    def close(self) -> None:
        self._inner.close()


class AsyncAppStream:
    def __init__(
        self,
        inner: AsyncIterator[AppCompletion],
        response: Any,
    ) -> None:
        self._inner = inner
        self._response = response

    def __aiter__(self):
        return self._inner.__aiter__()

    @property
    def headers(self) -> Dict[str, str]:
        """Raw HTTP response headers of the stream's envelope."""
        return dict(self._response.headers)

    @property
    def text_stream(self):
        return self._text_stream()

    async def _text_stream(self):
        async for event in self._inner:
            if event.text:
                yield event.text

    async def aclose(self) -> None:
        await self._inner.aclose()


class AppCompletions:
    """One Bailian application call, at ``client.apps.completions``."""

    def __init__(self, client) -> None:
        self._client = client

    def create(  # pylint: disable=too-many-locals
        self,
        *,
        app_id: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        session_id: Optional[str] = None,
        biz_params: Optional[Dict[str, Any]] = None,
        has_thoughts: Optional[bool] = None,
        doc_tag_codes: Optional[List[str]] = None,
        doc_reference_type: Optional[str] = None,
        memory_id: Optional[str] = None,
        image_list: Optional[List[str]] = None,
        file_list: Optional[List[str]] = None,
        rag_options: Optional[Dict[str, Any]] = None,
        incremental_output: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Union[AppCompletion, AppStream]:
        """Call the application; ``stream=True`` returns an AppStream.

        Send ``prompt`` for one turn, ``messages`` for a transcript the
        caller maintains, or both — ``prompt`` is then appended as the
        latest user turn. ``session_id`` and ``memory_id`` ask the
        service to keep the conversation state instead.

        ``biz_params`` carries the application's own flow and plugin
        inputs; ``parameters`` holds model knobs and anything this
        signature does not name.
        """
        _validate(app_id, prompt, messages)
        body = _build_body(
            prompt,
            messages,
            session_id,
            doc_reference_type,
            doc_tag_codes,
            memory_id,
            biz_params,
            image_list,
            file_list,
            temperature,
            top_p,
            top_k,
            seed,
            has_thoughts,
            rag_options,
            incremental_output,
            parameters,
        )
        path = f"/apps/{app_id}/completion"
        transport = self._client.transport
        if stream:
            response = transport.open_stream(
                "POST",
                path,
                json=body,
                headers=_SSE_HEADER,
                sdk_module=MODULE,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
            )
            return AppStream(self._events(response), response)
        response = transport.request(
            "POST",
            path,
            json=body,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse(response.json()), response)

    def _events(self, response) -> Iterator[AppCompletion]:
        try:
            for is_error, data in _sse_frames(response):
                if is_error:
                    if self._client.raise_on_error:
                        _raise_frame_error(data)
                    yield with_headers(_error_event(data), response)
                    return
                yield _parse(_try_json(data))
        finally:
            response.close()


class Apps:
    """``client.apps``: Bailian application resources."""

    def __init__(self, client) -> None:
        self.completions = AppCompletions(client)


class AsyncAppCompletions:
    """Async ``client.apps.completions``; same contract as the sync one."""

    def __init__(self, client) -> None:
        self._client = client

    async def create(  # pylint: disable=too-many-locals
        self,
        *,
        app_id: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        session_id: Optional[str] = None,
        biz_params: Optional[Dict[str, Any]] = None,
        has_thoughts: Optional[bool] = None,
        doc_tag_codes: Optional[List[str]] = None,
        doc_reference_type: Optional[str] = None,
        memory_id: Optional[str] = None,
        image_list: Optional[List[str]] = None,
        file_list: Optional[List[str]] = None,
        rag_options: Optional[Dict[str, Any]] = None,
        incremental_output: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Union[AppCompletion, AsyncAppStream]:
        _validate(app_id, prompt, messages)
        body = _build_body(
            prompt,
            messages,
            session_id,
            doc_reference_type,
            doc_tag_codes,
            memory_id,
            biz_params,
            image_list,
            file_list,
            temperature,
            top_p,
            top_k,
            seed,
            has_thoughts,
            rag_options,
            incremental_output,
            parameters,
        )
        path = f"/apps/{app_id}/completion"
        transport = self._client.transport
        if stream:
            response = await transport.open_stream(
                "POST",
                path,
                json=body,
                headers=_SSE_HEADER,
                sdk_module=MODULE,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
            )
            return AsyncAppStream(self._events(response), response)
        response = await transport.request(
            "POST",
            path,
            json=body,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse(response.json()), response)

    async def _events(self, response) -> AsyncIterator[AppCompletion]:
        try:
            is_error = False
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("event:error"):
                    is_error = True
                elif line.startswith("data:"):
                    data = line[len("data:") :].strip()
                    if is_error:
                        if self._client.raise_on_error:
                            _raise_frame_error(data)
                        yield with_headers(_error_event(data), response)
                        return
                    yield _parse(_try_json(data))
        finally:
            await response.aclose()


class AsyncApps:
    """Async ``client.apps``."""

    def __init__(self, client) -> None:
        self.completions = AsyncAppCompletions(client)
