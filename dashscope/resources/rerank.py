# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.rerank resource (native text-rerank)."""

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

from pydantic import BaseModel, ConfigDict

from dashscope.transport._options import request_options
from dashscope.types._parse import parse_model
from dashscope.types.common import (
    ResponseBase,
    Usage,
    response_envelope,
    with_headers,
)

if TYPE_CHECKING:
    import httpx

    from dashscope.transport._response import APIResponse

_PATH = "/services/rerank/text-rerank/text-rerank"
# Functional-area tag for x-dashscope-sdk-client (matches 1.x ``rerank``).
MODULE = "rerank"

__all__ = [
    "AsyncRerank",
    "Rerank",
    "RerankDocument",
    "RerankResponse",
    "RerankResult",
]


class RerankDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: Optional[str] = None


class RerankResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    relevance_score: float
    document: Optional[RerankDocument] = None


class RerankResponse(ResponseBase):
    results: List[RerankResult] = []
    usage: Optional[Usage] = None


def _parse(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> RerankResponse:
    output = body.get("output")
    if not isinstance(output, dict):
        # ``output`` drifted to a scalar; ``.get`` on it would escape as a
        # bare AttributeError before parse_model ever saw the body.
        output = {}
    return parse_model(
        RerankResponse,
        {
            "results": output.get("results") or [],
            "usage": body.get("usage"),
            **response_envelope(body),
        },
        strict=strict,
        response=response,
    )


def _build_body(
    *,
    model: str,
    query: str,
    documents: List[str],
    top_n: Optional[int],
    return_documents: Optional[bool],
    instruct: Optional[str],
    parameters: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not query:
        raise ValueError("query must not be empty")
    if not documents:
        raise ValueError("documents must not be empty")
    merged: Dict[str, Any] = {}
    for key, value in (
        ("top_n", top_n),
        ("return_documents", return_documents),
        ("instruct", instruct),
    ):
        if value is not None:
            merged[key] = value
    if parameters:
        merged.update(parameters)
    return {
        "model": model,
        "input": {"query": query, "documents": documents},
        "parameters": merged,
    }


class Rerank:
    def __init__(self, client) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = None,
        instruct: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> RerankResponse:
        ...

    @overload
    def create(
        self,
        *,
        model: str,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = None,
        instruct: Optional[str] = None,
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
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = None,
        instruct: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[RerankResponse, "APIResponse"]:
        """Rerank ``documents`` against ``query``.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        body = _build_body(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
            instruct=instruct,
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


class AsyncRerank:
    def __init__(self, client) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = None,
        instruct: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> RerankResponse:
        ...

    @overload
    async def create(
        self,
        *,
        model: str,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = None,
        instruct: Optional[str] = None,
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
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = None,
        instruct: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[RerankResponse, "APIResponse"]:
        """Rerank ``documents`` against ``query``.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        body = _build_body(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
            instruct=instruct,
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
