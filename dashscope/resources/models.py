# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.models resource (list/get)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from pydantic import AliasChoices, Field

from dashscope.errors import (
    ErrorCode,
    check_response,
    error_from_business_code,
    response_json,
)
from dashscope.transport._options import request_options
from dashscope.types._parse import parse_model
from dashscope.types.common import (
    ResponseBase,
    response_envelope,
    with_headers,
)
from dashscope.types.pagination import BasePage

if TYPE_CHECKING:
    import httpx

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


class ModelList(BasePage[Model]):
    """One page of the model catalog, from ``models.list``.

    ``page_no`` / ``page_size`` / ``total_count`` come from
    :class:`~dashscope.types.pagination.BasePage`; iterate the page directly
    (``for m in page``) or cross pages with ``iter_pages()`` /
    ``next_page()``.
    """

    models: List[Model] = []

    @property
    def items(self) -> List[Model]:
        return self.models


def _parse_list(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional[httpx.Response] = None,
) -> ModelList:
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
    return parse_model(
        ModelList,
        {
            "page_no": page_no,
            "page_size": page_size,
            "total_count": total,
            "models": models,
            **response_envelope(body),
        },
        strict=strict,
        response=response,
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
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
    ) -> ModelList:
        """List models; ``provider`` filters client-side per page.

        The server has no provider filter, so ``total_count`` is always
        the pre-filter total of the whole catalog, not of the filtered
        ``models`` slice. Pagination therefore walks the unfiltered catalog:
        ``has_next_page()`` / ``iter_pages()`` follow ``total_count``, and the
        ``provider`` filter is applied to each fetched page independently.
        """
        response = self._client.transport.request(
            "GET",
            _BASE_PATH,
            params={"page_no": page_no, "page_size": page_size},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
            ),
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        result = _parse_list(
            response_json(response),
            strict=self._client.config.strict_validation,
            response=response,
        )
        if provider:
            result.models = [
                m for m in result.models if _match_provider(m, provider)
            ]
        result.bind_pagination(
            lambda next_no, next_size: self.list(
                page_no=next_no,
                page_size=next_size,
                provider=provider,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
            ),
        )
        return with_headers(result, response)

    def get(
        self,
        model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
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
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
            ),
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        body = response_json(response)
        models = _parse_get_models(body)
        if not models:
            if self._client.raise_on_error:
                raise error_from_business_code(
                    message=f"Model '{model}' not found",
                    code=ErrorCode.MODEL_NOT_EXIST.value,
                    body=body,
                    headers=response.headers,
                )
            return with_headers(
                parse_model(
                    Model,
                    response_envelope(body),
                    strict=self._client.config.strict_validation,
                    response=response,
                ),
                response,
            )
        return with_headers(
            parse_model(
                Model,
                # Dict merge, not ``**`` twice: a wire entry carrying an
                # envelope key of its own would otherwise raise a
                # duplicate-keyword TypeError instead of being parsed.
                {**models[0], **response_envelope(body)},
                strict=self._client.config.strict_validation,
                response=response,
            ),
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
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
    ) -> ModelList:
        response = await self._client.transport.request(
            "GET",
            _BASE_PATH,
            params={"page_no": page_no, "page_size": page_size},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
            ),
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        result = _parse_list(
            response_json(response),
            strict=self._client.config.strict_validation,
            response=response,
        )
        if provider:
            result.models = [
                m for m in result.models if _match_provider(m, provider)
            ]
        result.bind_pagination(
            lambda next_no, next_size: self.list(
                page_no=next_no,
                page_size=next_size,
                provider=provider,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
            ),
        )
        return with_headers(result, response)

    async def get(
        self,
        model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
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
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
            ),
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        body = response_json(response)
        models = _parse_get_models(body)
        if not models:
            if self._client.raise_on_error:
                raise error_from_business_code(
                    message=f"Model '{model}' not found",
                    code=ErrorCode.MODEL_NOT_EXIST.value,
                    body=body,
                    headers=response.headers,
                )
            return with_headers(
                parse_model(
                    Model,
                    response_envelope(body),
                    strict=self._client.config.strict_validation,
                    response=response,
                ),
                response,
            )
        return with_headers(
            parse_model(
                Model,
                # Dict merge, not ``**`` twice: a wire entry carrying an
                # envelope key of its own would otherwise raise a
                # duplicate-keyword TypeError instead of being parsed.
                {**models[0], **response_envelope(body)},
                strict=self._client.config.strict_validation,
                response=response,
            ),
            response,
        )
