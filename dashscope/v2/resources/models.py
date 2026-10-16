# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.models resource (list/get)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, Field

from dashscope.v2.errors import (
    ErrorCode,
    check_response,
    error_from_business_code,
)
from dashscope.v2.types.common import (
    ResponseBase,
    response_envelope,
    with_headers,
)

_BASE_PATH = "/models"

MODULE = "models"

__all__ = ["AsyncModels", "Model", "ModelList", "Models"]


class Model(ResponseBase):
    """One model entry.

    Returned on its own by ``get`` and nested in ``ModelList.models``; both
    roles share one shape. The envelope is filled in on the standalone result
    and stays empty on the nested copies, because the server reports it once
    per response rather than once per item.
    """

    model: Optional[str] = None
    source: Optional[str] = None
    provider: Optional[str] = None
    # The wire field is "opensource"; the public name is spelled properly.
    open_source: Optional[bool] = Field(
        None,
        validation_alias=AliasChoices("open_source", "opensource"),
    )


class ModelList(ResponseBase):
    page_no: int = 1
    page_size: int = 10
    total_count: int = 0
    models: List[Model] = []


def _parse_list(body: Dict[str, Any]) -> ModelList:
    # ``get`` is this same endpoint filtered by query params, and the
    # native envelope nests its page under ``output``; ``data`` stays
    # accepted for gateways that re-wrap the page OpenAI-style.
    data = body.get("output") or body.get("data") or {}
    if isinstance(data, dict):
        models = data.get("models") or []
        page_no = data.get("page_no", 1)
        page_size = data.get("page_size", 10)
        total = data.get("total", len(models))
    else:
        models, page_no, page_size, total = data, 1, 10, len(data)
    return ModelList(
        page_no=page_no,
        page_size=page_size,
        total_count=total,
        models=models,
        **response_envelope(body),
    )


def _parse_get_models(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = body.get("output") or body.get("data") or {}
    if isinstance(output, dict):
        return output.get("models") or []
    return []


def _match_provider(model: Model, provider: str) -> bool:
    wanted = provider.lower()
    return wanted in (
        (model.provider or "").lower(),
        (model.source or "").lower(),
    )


class Models:
    def __init__(self, client) -> None:
        self._client = client

    def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        provider: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> ModelList:
        """List models; ``provider`` filters client-side per page.

        The server has no provider filter, so ``total_count`` is always
        the pre-filter total of the whole catalog, not of the filtered
        ``models`` slice.
        """
        response = self._client.transport.request(
            "GET",
            _BASE_PATH,
            params={"page_no": page_no, "page_size": page_size},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        result = _parse_list(response.json())
        if provider:
            result.models = [
                m for m in result.models if _match_provider(m, provider)
            ]
        return with_headers(result, response)

    def get(
        self,
        model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Model:
        # Server filters by model name via query params.
        response = self._client.transport.request(
            "GET",
            _BASE_PATH,
            params={"model": model, "page_no": 1, "page_size": 1},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        body = response.json()
        models = _parse_get_models(body)
        if not models:
            if self._client.raise_on_error:
                raise error_from_business_code(
                    message=f"Model '{model}' not found",
                    code=ErrorCode.MODEL_NOT_EXIST.value,
                    body=body,
                    headers=response.headers,
                )
            return with_headers(Model(**response_envelope(body)), response)
        return with_headers(
            Model(**models[0], **response_envelope(body)),
            response,
        )


class AsyncModels:
    def __init__(self, client) -> None:
        self._client = client

    async def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        provider: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> ModelList:
        response = await self._client.transport.request(
            "GET",
            _BASE_PATH,
            params={"page_no": page_no, "page_size": page_size},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        result = _parse_list(response.json())
        if provider:
            result.models = [
                m for m in result.models if _match_provider(m, provider)
            ]
        return with_headers(result, response)

    async def get(
        self,
        model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> Model:
        # Server filters by model name via query params.
        response = await self._client.transport.request(
            "GET",
            _BASE_PATH,
            params={"model": model, "page_no": 1, "page_size": 1},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        body = response.json()
        models = _parse_get_models(body)
        if not models:
            if self._client.raise_on_error:
                raise error_from_business_code(
                    message=f"Model '{model}' not found",
                    code=ErrorCode.MODEL_NOT_EXIST.value,
                    body=body,
                    headers=response.headers,
                )
            return with_headers(Model(**response_envelope(body)), response)
        return with_headers(
            Model(**models[0], **response_envelope(body)),
            response,
        )
