# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.tokenization resource (token counting)."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    overload,
)

from dashscope.transport._options import request_options
from dashscope.types._parse import parse_model
from dashscope.types.common import (
    ResponseBase,
    Usage,
    response_envelope,
    with_headers,
)
from dashscope.types.messages import MessageInput, normalize_messages

if TYPE_CHECKING:
    import httpx

    from dashscope.transport._response import APIResponse

_PATH = "/tokenizer"
# Functional-area tag for x-dashscope-sdk-client (matches 1.x ``tokenizers``).
MODULE = "tokenizers"

__all__ = [
    "AsyncTokenization",
    "TokenCount",
    "Tokenization",
]


class TokenCount(ResponseBase):
    usage: Usage = Usage()
    # Server tokenizer output (1.x ``output.token_ids`` / ``output.tokens``).
    token_ids: Optional[List[int]] = None
    tokens: Optional[List[str]] = None

    @property
    def input_tokens(self) -> int:
        return self.usage.input_tokens


def _parse(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> TokenCount:
    output = body.get("output")
    if not isinstance(output, dict):
        # ``output`` drifted to a scalar; ``.get`` on it would escape as a
        # bare AttributeError before parse_model ever saw the body.
        output = {}
    return parse_model(
        TokenCount,
        {
            "usage": body.get("usage") or {},
            "token_ids": output.get("token_ids"),
            "tokens": output.get("tokens"),
            **response_envelope(body),
        },
        strict=strict,
        response=response,
    )


def _build_body(
    *,
    model: str,
    prompt: Optional[str],
    messages: Union[str, List[MessageInput], None],
    parameters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    input_body: Dict[str, Any]
    if messages is not None:
        dumped = [
            m.model_dump(exclude_none=True)
            for m in normalize_messages(messages)
        ]
        if prompt:
            dumped.append({"role": "user", "content": prompt})
        input_body = {"messages": dumped}
    elif prompt:
        input_body = {"prompt": prompt}
    else:
        raise ValueError("prompt or messages is required")
    return {
        "model": model,
        "input": input_body,
        "parameters": dict(parameters or {}),
    }


class Tokenization:
    def __init__(self, client) -> None:
        self._client = client

    @overload
    def count(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        messages: Union[str, List[MessageInput], None] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenCount:
        ...

    @overload
    def count(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        messages: Union[str, List[MessageInput], None] = None,
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

    def count(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        messages: Union[str, List[MessageInput], None] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenCount, "APIResponse"]:
        """Count tokens for ``prompt`` / ``messages``.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        body = _build_body(
            model=model,
            prompt=prompt,
            messages=messages,
            parameters=parameters,
        )
        resp = self._client.transport.request_api(
            "POST",
            _PATH,
            json=body,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            sdk_module=MODULE,
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
        resp.raise_for_status()
        return with_headers(
            _parse(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )


class AsyncTokenization:
    def __init__(self, client) -> None:
        self._client = client

    @overload
    async def count(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        messages: Union[str, List[MessageInput], None] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenCount:
        ...

    @overload
    async def count(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        messages: Union[str, List[MessageInput], None] = None,
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

    async def count(
        self,
        *,
        model: str,
        prompt: Optional[str] = None,
        messages: Union[str, List[MessageInput], None] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenCount, "APIResponse"]:
        """Count tokens for ``prompt`` / ``messages``.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        body = _build_body(
            model=model,
            prompt=prompt,
            messages=messages,
            parameters=parameters,
        )
        resp = await self._client.transport.request_api(
            "POST",
            _PATH,
            json=body,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            sdk_module=MODULE,
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
        resp.raise_for_status()
        return with_headers(
            _parse(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )
