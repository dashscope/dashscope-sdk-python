# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.models resource (list/get)."""

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

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from dashscope.errors import (
    ErrorCode,
    error_from_business_code,
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

    from dashscope.transport._response import APIResponse

_BASE_PATH = "/models"

MODULE = "models"

__all__ = [
    "AsyncModels",
    "InferenceMetadata",
    "Model",
    "ModelInfo",
    "ModelList",
    "ModelPrice",
    "ModelPriceRange",
    "Models",
]


class ModelPrice(BaseModel):
    """One billing item inside a price range (``type``/``price``/unit/name)."""

    model_config = ConfigDict(extra="allow")

    type: Optional[str] = None
    # The wire sends price as a string ("2", "0.0008"); kept verbatim so no
    # precision is lost coercing it to a float.
    price: Optional[str] = None
    price_unit: Optional[str] = None
    price_name: Optional[str] = None


class ModelPriceRange(BaseModel):
    """One tiered billing range; ``range_name`` is ``Default`` when untiered."""

    model_config = ConfigDict(extra="allow")

    prices: List[ModelPrice] = []
    range_name: Optional[str] = None


class InferenceMetadata(BaseModel):
    """Input/output modalities (``Text``/``Image``/``Audio``/``Video``)."""

    model_config = ConfigDict(extra="allow")

    request_modality: List[str] = []
    response_modality: List[str] = []


class ModelInfo(BaseModel):
    """Context-length limits; ``None`` means unlimited or not applicable."""

    model_config = ConfigDict(extra="allow")

    context_window: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    max_reasoning_tokens: Optional[int] = None
    reasoning_max_input_tokens: Optional[int] = None
    reasoning_max_output_tokens: Optional[int] = None


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
    name: Optional[str] = None
    description: Optional[str] = None
    capabilities: List[str] = []
    inference_provider: Optional[str] = None
    features: List[str] = []
    prices: List[ModelPriceRange] = []
    published_time: Optional[str] = None
    inference_metadata: Optional[InferenceMetadata] = None
    model_info: Optional[ModelInfo] = None
    equivalent_snapshot: Optional[str] = None


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
    """Build a page from the native envelope.

    ``get`` is this same endpoint filtered by query params, and the native
    gateway nests its page under ``output``; ``data`` stays accepted for
    gateways that re-wrap the page OpenAI-style.

    A body carrying the page as a bare list has no ``page_no`` / ``total`` to
    read, so the list is taken as one whole page: the generic envelope
    lifter only unwraps a *dict*, and letting this shape through it would
    parse as an empty catalog rather than as the malformed one it is. The
    ``total`` fallback to ``len(models)`` keeps ``total_count`` meaningful
    for gateways that omit it, matching ``files.list``.
    """
    data = body.get("output") or body.get("data") or {}
    if isinstance(data, dict):
        models = data.get("models")
        # A retyped ``models`` would both poison the ``len()`` below (a
        # string's length is not a page total) and blow up in the model
        # layer's repair of a list field. Empty is the honest reading of
        # "not a list", matching ``JobList.from_payload``.
        if not isinstance(models, list):
            models = []
        page_no = data.get("page_no", 1)
        page_size = data.get("page_size", 10)
        total = data.get("total", len(models))
    else:
        # Only a genuine list can be read as a whole page; ``len()`` on a
        # drifted scalar would escape as a bare TypeError from here.
        models = data if isinstance(data, list) else []
        page_no, page_size, total = 1, 10, len(models)
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


def _build_list_params(
    page_no: int,
    page_size: int,
    *,
    name: Optional[str],
    model: Optional[str],
    language: Optional[str],
    provider: Optional[str],
    providers: Optional[List[str]],
    inference_providers: Optional[List[str]],
    capabilities: Optional[List[str]],
    features: Optional[List[str]],
    context_window: Optional[int],
    service_site: Optional[str],
    supports: Optional[List[str]],
    deployment_methods: Optional[List[str]],
    deployment_ptu_service_tiers: Optional[List[str]],
) -> Dict[str, Any]:
    """Translate the public filters into the server's query contract.

    Array filters are sent as repeated parameter names (``capabilities=TG&
    capabilities=Reasoning``), which httpx produces from a list value. The
    singular ``provider`` convenience folds into the server-side ``providers``
    filter so the catalog total reflects the filtered set, rather than the
    old client-side per-page slice whose ``total_count`` stayed pre-filter.
    """
    params: Dict[str, Any] = {"page_no": page_no, "page_size": page_size}
    effective_providers = list(providers or [])
    if provider and provider not in effective_providers:
        effective_providers.append(provider)
    if effective_providers:
        params["providers"] = effective_providers
    if name is not None:
        params["name"] = name
    if model is not None:
        params["model"] = model
    if language is not None:
        params["language"] = language
    if inference_providers:
        params["inference_providers"] = inference_providers
    if capabilities:
        params["capabilities"] = capabilities
    if features:
        params["features"] = features
    if context_window is not None:
        params["context_window"] = context_window
    if service_site is not None:
        params["service_site"] = service_site
    if supports:
        params["supports"] = supports
    if deployment_methods:
        params["deployment_methods"] = deployment_methods
    if deployment_ptu_service_tiers:
        params["deployment_ptu_service_tiers"] = deployment_ptu_service_tiers
    return params


class Models:
    def __init__(self, client) -> None:
        self._client = client

    @overload
    def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        provider: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        providers: Optional[List[str]] = None,
        inference_providers: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        context_window: Optional[int] = None,
        service_site: Optional[str] = None,
        supports: Optional[List[str]] = None,
        deployment_methods: Optional[List[str]] = None,
        deployment_ptu_service_tiers: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> ModelList:
        ...

    @overload
    def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        provider: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        providers: Optional[List[str]] = None,
        inference_providers: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        context_window: Optional[int] = None,
        service_site: Optional[str] = None,
        supports: Optional[List[str]] = None,
        deployment_methods: Optional[List[str]] = None,
        deployment_ptu_service_tiers: Optional[List[str]] = None,
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

    def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        provider: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        providers: Optional[List[str]] = None,
        inference_providers: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        context_window: Optional[int] = None,
        service_site: Optional[str] = None,
        supports: Optional[List[str]] = None,
        deployment_methods: Optional[List[str]] = None,
        deployment_ptu_service_tiers: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ModelList, "APIResponse"]:
        """List models with the server's full filter contract.

        All filters are applied server-side, so ``total_count`` is the total
        of the *filtered* catalog and pagination walks that filtered set.
        Array filters (``providers``/``capabilities``/``features``/...) are
        sent as repeated query parameters. The singular ``provider`` is a
        convenience that folds into ``providers``; it no longer filters the
        current page client-side (which left ``total_count`` pre-filter and
        hid matches on other pages).

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        instead of a ``ModelList``, before any error check, so callers can
        read the raw response (status, headers, ``elapsed``) or re-parse it.
        Note that ``resp.parse(ModelList)`` is the generic envelope mapping
        and does not replay this method's own reshaping: a page delivered as
        a bare list, or one whose ``total`` the gateway omitted, parses
        through it as an empty catalog. Use ``resp.json()`` and
        ``_parse_list`` when the raw path has to agree with this one.
        """
        filter_kwargs = dict(
            name=name,
            model=model,
            language=language,
            provider=provider,
            providers=providers,
            inference_providers=inference_providers,
            capabilities=capabilities,
            features=features,
            context_window=context_window,
            service_site=service_site,
            supports=supports,
            deployment_methods=deployment_methods,
            deployment_ptu_service_tiers=deployment_ptu_service_tiers,
        )
        resp = self._client.transport.request_api(
            "GET",
            _BASE_PATH,
            params=_build_list_params(page_no, page_size, **filter_kwargs),
            sdk_module=MODULE,
            timeout=timeout,
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
        resp.raise_for_status()
        result = _parse_list(
            resp.json(),
            strict=self._client.config.strict_validation,
            response=resp.http_response,
        )
        result.bind_pagination(
            lambda next_no, next_size: self.list(
                page_no=next_no,
                page_size=next_size,
                **filter_kwargs,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            ),
            requested_page_no=page_no,
            requested_page_size=page_size,
        )
        return with_headers(result, resp.http_response)

    @overload
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
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Model:
        ...

    @overload
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
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

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
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Model, "APIResponse"]:
        # Server filters by model name via query params.
        resp = self._client.transport.request_api(
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
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        body = resp.json()
        models = _parse_get_models(body)
        if not models:
            if self._client.raise_on_error:
                raise error_from_business_code(
                    message=f"Model '{model}' not found",
                    code=ErrorCode.MODEL_NOT_EXIST.value,
                    body=body,
                    headers=resp.http_response.headers,
                )
            return with_headers(
                parse_model(
                    Model,
                    response_envelope(body),
                    strict=self._client.config.strict_validation,
                    response=resp.http_response,
                ),
                resp.http_response,
            )
        return with_headers(
            parse_model(
                Model,
                # Dict merge, not ``**`` twice: a wire entry carrying an
                # envelope key of its own would otherwise raise a
                # duplicate-keyword TypeError instead of being parsed.
                {**models[0], **response_envelope(body)},
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )


class AsyncModels:
    def __init__(self, client) -> None:
        self._client = client

    @overload
    async def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        provider: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        providers: Optional[List[str]] = None,
        inference_providers: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        context_window: Optional[int] = None,
        service_site: Optional[str] = None,
        supports: Optional[List[str]] = None,
        deployment_methods: Optional[List[str]] = None,
        deployment_ptu_service_tiers: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> ModelList:
        ...

    @overload
    async def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        provider: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        providers: Optional[List[str]] = None,
        inference_providers: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        context_window: Optional[int] = None,
        service_site: Optional[str] = None,
        supports: Optional[List[str]] = None,
        deployment_methods: Optional[List[str]] = None,
        deployment_ptu_service_tiers: Optional[List[str]] = None,
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

    async def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        provider: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        providers: Optional[List[str]] = None,
        inference_providers: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        context_window: Optional[int] = None,
        service_site: Optional[str] = None,
        supports: Optional[List[str]] = None,
        deployment_methods: Optional[List[str]] = None,
        deployment_ptu_service_tiers: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ModelList, "APIResponse"]:
        filter_kwargs = dict(
            name=name,
            model=model,
            language=language,
            provider=provider,
            providers=providers,
            inference_providers=inference_providers,
            capabilities=capabilities,
            features=features,
            context_window=context_window,
            service_site=service_site,
            supports=supports,
            deployment_methods=deployment_methods,
            deployment_ptu_service_tiers=deployment_ptu_service_tiers,
        )
        resp = await self._client.transport.request_api(
            "GET",
            _BASE_PATH,
            params=_build_list_params(page_no, page_size, **filter_kwargs),
            sdk_module=MODULE,
            timeout=timeout,
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
        resp.raise_for_status()
        result = _parse_list(
            resp.json(),
            strict=self._client.config.strict_validation,
            response=resp.http_response,
        )
        result.bind_pagination(
            lambda next_no, next_size: self.list(
                page_no=next_no,
                page_size=next_size,
                **filter_kwargs,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            ),
            requested_page_no=page_no,
            requested_page_size=page_size,
        )
        return with_headers(result, resp.http_response)

    @overload
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
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Model:
        ...

    @overload
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
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

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
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Model, "APIResponse"]:
        # Server filters by model name via query params.
        resp = await self._client.transport.request_api(
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
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        body = resp.json()
        models = _parse_get_models(body)
        if not models:
            if self._client.raise_on_error:
                raise error_from_business_code(
                    message=f"Model '{model}' not found",
                    code=ErrorCode.MODEL_NOT_EXIST.value,
                    body=body,
                    headers=resp.http_response.headers,
                )
            return with_headers(
                parse_model(
                    Model,
                    response_envelope(body),
                    strict=self._client.config.strict_validation,
                    response=resp.http_response,
                ),
                resp.http_response,
            )
        return with_headers(
            parse_model(
                Model,
                # Dict merge, not ``**`` twice: a wire entry carrying an
                # envelope key of its own would otherwise raise a
                # duplicate-keyword TypeError instead of being parsed.
                {**models[0], **response_envelope(body)},
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )