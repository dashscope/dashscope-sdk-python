# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.rerank resource (native text-rerank)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from dashscope.v2.errors import check_response
from dashscope.v2.types.common import (
    ResponseBase,
    Usage,
    response_envelope,
    with_headers,
)

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


def _parse(body: Dict[str, Any]) -> RerankResponse:
    output = body.get("output") or {}
    return RerankResponse(
        results=output.get("results") or [],
        usage=body.get("usage"),
        **response_envelope(body),
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
    ) -> RerankResponse:
        body = _build_body(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
            instruct=instruct,
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


class AsyncRerank:
    def __init__(self, client) -> None:
        self._client = client

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
    ) -> RerankResponse:
        body = _build_body(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=return_documents,
            instruct=instruct,
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
