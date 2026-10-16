# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.tokenization resource (token counting)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from dashscope.v2.errors import check_response
from dashscope.v2.types.common import (
    ResponseBase,
    Usage,
    response_envelope,
    with_headers,
)
from dashscope.v2.types.messages import MessageInput, normalize_messages

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


def _parse(body: Dict[str, Any]) -> TokenCount:
    output = body.get("output") or {}
    return TokenCount(
        usage=body.get("usage") or {},
        token_ids=output.get("token_ids"),
        tokens=output.get("tokens"),
        **response_envelope(body),
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
    ) -> TokenCount:
        body = _build_body(
            model=model,
            prompt=prompt,
            messages=messages,
            parameters=parameters,
        )
        response = self._client.transport.request(
            "POST",
            _PATH,
            json=body,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            sdk_module=MODULE,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse(response.json()), response)


class AsyncTokenization:
    def __init__(self, client) -> None:
        self._client = client

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
    ) -> TokenCount:
        body = _build_body(
            model=model,
            prompt=prompt,
            messages=messages,
            parameters=parameters,
        )
        response = await self._client.transport.request(
            "POST",
            _PATH,
            json=body,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            sdk_module=MODULE,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse(response.json()), response)
