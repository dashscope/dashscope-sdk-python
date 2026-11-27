# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.models resource (list/get)."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
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
    response_error,
)
from dashscope.resources._model_cache import (
    _CatalogCache,
    catalog_key,
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
    "AsyncModelLimits",
    "AsyncModelPermissions",
    "AsyncModels",
    "InferenceMetadata",
    "Model",
    "ModelInfo",
    "ModelLimits",
    "ModelList",
    "ModelPermission",
    "ModelPermissions",
    "ModelPrice",
    "ModelPriceRange",
    "ModelQuota",
    "Models",
    "PermissionDetail",
    "PermissionList",
    "QuotaList",
    "RateLimit",
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
    # SDK-injected query context, not per-model wire fields: the catalog
    # endpoint's region and the effective workspace. ``service_site`` echoes
    # the query's deployment-site filter (None when unfiltered, i.e. the page
    # may span sites); it is a different dimension from ``served_region`` and
    # the two are never conflated. A gateway that does return ``service_site``
    # per model keeps its own value.
    served_region: Optional[str] = None
    workspace: Optional[str] = None
    service_site: Optional[str] = None


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


class RateLimit(BaseModel):
    """One rate-limit tier (account- or workspace-level).

    ``request_limit_period`` is in seconds: 60 means RPM, 1 means QPS.
    ``usage_limit`` of ``None`` means no usage cap. Every field is optional
    because the gateway omits the dimensions a model does not constrain.
    """

    model_config = ConfigDict(extra="allow")

    request_limit: Optional[int] = None
    request_limit_period: Optional[int] = None
    usage_limit: Optional[int] = None
    usage_limit_field: Optional[str] = None
    usage_limit_period: Optional[int] = None
    async_user_queue_limit: Optional[int] = None
    async_user_concurrency_limit: Optional[int] = None


class ModelQuota(ResponseBase):
    """One model's rate-limit quotas, from ``models.limits``.

    ``model_limit`` is the account-level ceiling (``None`` when unset);
    ``workspace_limit`` is the per-workspace cap (``None`` when the workspace
    inherits the account limit). The envelope is filled on the standalone
    ``get`` result and stays empty on the nested list copies.
    """

    model: Optional[str] = None
    workspace_id: Optional[str] = None
    model_limit: Optional[RateLimit] = None
    workspace_limit: Optional[RateLimit] = None


class QuotaList(BasePage[ModelQuota]):
    """One page of model quotas, from ``models.limits.list``.

    ``page_no`` / ``page_size`` / ``total_count`` come from
    :class:`~dashscope.types.pagination.BasePage`; iterate the page directly
    or cross pages with ``iter_pages()`` / ``next_page()``.
    """

    quotas: List[ModelQuota] = []

    @property
    def items(self) -> List[ModelQuota]:
        return self.quotas


class PermissionDetail(BaseModel):
    """Per-action authorization flags for one model in a workspace."""

    model_config = ConfigDict(extra="allow")

    inference: Optional[bool] = None
    fine_tune: Optional[bool] = None
    deploy: Optional[bool] = None


class ModelPermission(ResponseBase):
    """One model's authorization state, from ``models.permissions``.

    ``permissions`` carries the per-action flags (inference / fine_tune /
    deploy). Whether a model appears at all depends on the query's
    ``authorization_scope`` (AUTHORIZABLE vs AUTHORIZED).
    """

    model: Optional[str] = None
    name: Optional[str] = None
    permissions: Optional[PermissionDetail] = None


class PermissionList(BasePage[ModelPermission]):
    """One page of model permissions, from ``models.permissions.list``.

    ``page_no`` / ``page_size`` / ``total_count`` come from
    :class:`~dashscope.types.pagination.BasePage`; iterate the page directly
    or cross pages with ``iter_pages()`` / ``next_page()``.
    """

    permissions: List[ModelPermission] = []

    @property
    def items(self) -> List[ModelPermission]:
        return self.permissions


def _with_context(
    item: Dict[str, Any],
    *,
    region: Optional[str],
    workspace: Optional[str],
    service_site: Optional[str],
) -> Dict[str, Any]:
    """Stamp the SDK-injected query context onto one wire model dict.

    ``served_region`` / ``workspace`` describe the endpoint the catalog was
    read from, so the SDK's values are authoritative and override any a
    gateway might echo. ``service_site`` is only filled from the query filter
    when the wire item does not carry its own, so a real per-model value is
    never erased by an unfiltered (None) query.
    """
    merged = dict(item)
    merged["served_region"] = region
    merged["workspace"] = workspace
    merged.setdefault("service_site", service_site)
    return merged


def _parse_list(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional[httpx.Response] = None,
    region: Optional[str] = None,
    workspace: Optional[str] = None,
    service_site: Optional[str] = None,
) -> ModelList:
    """Build a page from the native envelope.

    ``get`` is this same endpoint filtered by query params, and the native
    gateway nests its page under ``output``; ``data`` stays accepted for
    gateways that re-wrap the page OpenAI-style.

    A body carrying the page as a bare list has no ``page_no`` / ``total`` to
    read, so the list is taken as one whole page: the generic envelope
    lifter only unwraps a *dict*, and letting this shape through it would
    parse as an empty catalog rather than as the malformed one it is. When
    a dict page omits ``total``, ``total_count`` falls back to
    ``len(models)`` for display, but the page is marked ``_total_unknown``
    so traversal continues while pages come back full instead of stopping
    after page 1.

    ``region`` / ``workspace`` / ``service_site`` are stamped onto every model
    as SDK-injected query context (see :func:`_with_context`).
    """
    data = body.get("output") or body.get("data") or {}
    total_unknown = False
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
        total = data.get("total")
        if total is None:
            # No total to do offset arithmetic against: traversal falls
            # back to full-page continuation instead of stopping at page 1.
            total_unknown = True
            total = len(models)
    else:
        # Only a genuine list can be read as a whole page; ``len()`` on a
        # drifted scalar would escape as a bare TypeError from here.
        models = data if isinstance(data, list) else []
        page_no, page_size, total = 1, 10, len(models)
    models = [
        _with_context(
            m,
            region=region,
            workspace=workspace,
            service_site=service_site,
        )
        if isinstance(m, dict)
        else m
        for m in models
    ]
    result = parse_model(
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
    result._total_unknown = total_unknown  # pylint: disable=protected-access
    return result


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
    scalar_filters = {
        "name": name,
        "model": model,
        "language": language,
        "context_window": context_window,
        "service_site": service_site,
    }
    params.update({k: v for k, v in scalar_filters.items() if v is not None})
    list_filters = {
        "inference_providers": inference_providers,
        "capabilities": capabilities,
        "features": features,
        "supports": supports,
        "deployment_methods": deployment_methods,
        "deployment_ptu_service_tiers": deployment_ptu_service_tiers,
    }
    params.update({k: v for k, v in list_filters.items() if v})
    return params


def _parse_meta_page(
    cls: type,
    body: Dict[str, Any],
    *,
    items_key: str,
    strict: bool = False,
    response: Optional[httpx.Response] = None,
) -> Any:
    """Build a quota/permission page from the native ``output`` envelope.

    ``models.limits`` and ``models.permissions`` answer with the same shape as
    the catalog -- ``output.total`` / ``page_no`` / ``page_size`` plus an item
    array (``quotas`` / ``permissions``) -- so they share this lifter and the
    :class:`~dashscope.types.pagination.BasePage` traversal. ``data`` stays
    accepted for gateways that re-wrap the page OpenAI-style. A non-list item
    field reads as empty rather than poisoning ``len()``; when the gateway
    omits ``total``, ``total_count`` falls back to the item count for display
    and the page is marked ``_total_unknown`` so traversal continues while
    pages come back full.
    """
    data = body.get("output") or body.get("data") or {}
    total_unknown = False
    if isinstance(data, dict):
        items = data.get(items_key)
        if not isinstance(items, list):
            items = []
        page_no = data.get("page_no", 1)
        page_size = data.get("page_size", 20)
        total = data.get("total")
        if total is None:
            total_unknown = True
            total = len(items)
    else:
        items = data if isinstance(data, list) else []
        page_no, page_size, total = 1, 20, len(items)
    items = [item for item in items if isinstance(item, dict)]
    result = parse_model(
        cls,
        {
            "page_no": page_no,
            "page_size": page_size,
            "total_count": total,
            items_key: items,
            **response_envelope(body),
        },
        strict=strict,
        response=response,
    )
    result._total_unknown = total_unknown  # pylint: disable=protected-access
    return result


def _meta_items(body: Dict[str, Any], items_key: str) -> List[Dict[str, Any]]:
    """Raw item dicts of a quota/permission page, for the standalone ``get``."""
    data = body.get("output") or body.get("data") or {}
    if isinstance(data, dict):
        items = data.get(items_key) or []
        return [item for item in items if isinstance(item, dict)]
    return []


def _require_model_id(model: str) -> None:
    # An empty model filter matches the whole catalog, so a standalone
    # ``get("")`` would silently answer with the first entry instead of
    # failing. Same guard as ``is_available``.
    if not model or not model.strip():
        raise ValueError("model must be a non-empty model id")


def _build_limits_params(
    page_no: int,
    page_size: int,
    *,
    name: Optional[str],
    model: Optional[str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"page_no": page_no, "page_size": page_size}
    if name is not None:
        params["name"] = name
    if model is not None:
        params["model"] = model
    return params


def _build_permissions_params(
    page_no: int,
    page_size: int,
    *,
    name: Optional[str],
    model: Optional[str],
    authorization_scope: Optional[str],
    action: Optional[str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"page_no": page_no, "page_size": page_size}
    if name is not None:
        params["name"] = name
    if model is not None:
        params["model"] = model
    if authorization_scope is not None:
        params["authorization_scope"] = authorization_scope
    if action is not None:
        params["action"] = action
    return params


def _parse_catalog_model(
    resp: "APIResponse",
    entry: Dict[str, Any],
    *,
    region: Optional[str],
    workspace: Optional[str],
    strict: bool,
) -> Model:
    return with_headers(
        parse_model(
            Model,
            {
                **_with_context(
                    entry,
                    region=region,
                    workspace=workspace,
                    service_site=None,
                ),
                **response_envelope(resp.json()),
            },
            strict=strict,
            response=resp.http_response,
        ),
        resp.http_response,
    )


def _check_catalog_probe(resp: "APIResponse", raise_on_error: bool) -> None:
    # Boolean/optional probes have no error envelope. Even when other methods
    # return errors as data, a probe must distinguish failure from no match.
    if not raise_on_error:
        error = response_error(resp.http_response)
        if error is not None:
            raise error


class Models:
    def __init__(self, client) -> None:
        self._client = client
        self.limits = ModelLimits(client)
        self.permissions = ModelPermissions(client)
        self._catalog_cache = _CatalogCache()

    def _models_get(
        self,
        params: Mapping[str, Any],
        *,
        refresh: bool,
        raw: bool,
        timeout: Optional[float],
        api_key: Optional[str],
        workspace: Optional[str],
        options: Any,
    ) -> "APIResponse":
        """Single cached read seam for the catalog (list/get/is_available).

        ``raw=True`` bypasses the cache and the error check, returning the
        live response. Otherwise a fresh, unexpired entry is reused; on a
        miss the response is fetched, error-checked, and cached only when it
        is genuinely successful, so errors are never cached even in
        non-raising mode. ``refresh=True`` skips the lookup but still updates
        the entry. Options carrying ``extra_query`` / ``extra_headers`` /
        ``extra_body`` bypass the cache entirely: they change the effective
        query, can override the auth/workspace headers, or add a JSON body
        (even on GET), none of which the key captures. The cached response
        is re-parsed by each caller, so every call hands back fresh model
        objects rather than a shared instance.
        """
        if raw:
            return self._client.transport.request_api(
                "GET",
                _BASE_PATH,
                params=params,
                sdk_module=MODULE,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                options=options,
                client=self._client,
            )
        # An injected client's mutable defaults, auth or hooks can change
        # the actual request independently of the SDK's catalog key.
        cacheable = self._client.transport.owns_http_client and (
            options is None
            or not (
                options.extra_query
                or options.extra_headers
                or options.extra_body
            )
        )
        key = catalog_key(
            self._client.config,
            params,
            api_key=api_key,
            workspace=workspace,
        )
        if cacheable and not refresh:
            cached = self._catalog_cache.get(key)
            if cached is not None:
                return cached
        resp = self._client.transport.request_api(
            "GET",
            _BASE_PATH,
            params=params,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=options,
            client=self._client,
        )
        resp.raise_for_status()
        if cacheable and response_error(resp.http_response) is None:
            self._catalog_cache.set(key, resp)
        return resp

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
        refresh: bool = False,
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
        refresh: bool = False,
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
        refresh: bool = False,
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

        Reads are served from a short-TTL (60s) per-client cache keyed by
        Region, workspace, credential and the exact query, so repeated
        identical reads avoid the gateway. ``refresh=True`` forces a fresh
        fetch (and updates the cache); errors and ``raw=True`` are never
        cached. The cache never crosses tenants or Regions.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        instead of a ``ModelList``, before any error check, so callers can
        read the raw response (status, headers, ``elapsed``) or re-parse it.
        Note that ``resp.parse(ModelList)`` is the generic envelope mapping
        and does not replay this method's own reshaping: a page delivered as
        a bare list, or one whose ``total`` the gateway omitted, parses
        through it as an empty catalog. Use ``resp.json()`` and
        ``_parse_list`` when the raw path has to agree with this one.
        """
        filter_kwargs: Dict[str, Any] = {
            "name": name,
            "model": model,
            "language": language,
            "provider": provider,
            "providers": providers,
            "inference_providers": inference_providers,
            "capabilities": capabilities,
            "features": features,
            "context_window": context_window,
            "service_site": service_site,
            "supports": supports,
            "deployment_methods": deployment_methods,
            "deployment_ptu_service_tiers": deployment_ptu_service_tiers,
        }
        resp = self._models_get(
            _build_list_params(page_no, page_size, **filter_kwargs),
            refresh=refresh,
            raw=raw,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
        )
        if raw:
            return resp
        result = _parse_list(
            resp.json(),
            strict=self._client.config.strict_validation,
            response=resp.http_response,
            region=self._client.config.region,
            workspace=(
                workspace
                if workspace is not None
                else self._client.config.workspace
            ),
            service_site=service_site,
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
                refresh=refresh,
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
        refresh: bool = False,
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
        refresh: bool = False,
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
        refresh: bool = False,
        raw: bool = False,
    ) -> Union[Model, "APIResponse"]:
        # Server filters by model name via query params. Shares the catalog
        # cache with list/is_available: the same (region, workspace,
        # credential, query) reads one response. ``refresh=True`` forces a
        # fresh fetch; errors and ``raw=True`` are never cached.
        _require_model_id(model)
        resp = self._models_get(
            {"model": model, "page_no": 1, "page_size": 1},
            refresh=refresh,
            raw=raw,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
        )
        if raw:
            return resp
        body = resp.json()
        models = _parse_get_models(body)
        region = self._client.config.region
        effective_workspace = (
            workspace
            if workspace is not None
            else self._client.config.workspace
        )
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
                    {
                        **response_envelope(body),
                        "served_region": region,
                        "workspace": effective_workspace,
                    },
                    strict=self._client.config.strict_validation,
                    response=resp.http_response,
                ),
                resp.http_response,
            )
        return _parse_catalog_model(
            resp,
            models[0],
            region=region,
            workspace=effective_workspace,
            strict=self._client.config.strict_validation,
        )

    def iter(
        self,
        *,
        page_size: int = 100,
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
        refresh: bool = False,
    ) -> Iterator[Model]:
        """Iterate every model across all pages, lazily.

        Takes the same server-side filters as :meth:`list` (minus ``page_no``
        / ``raw``) and walks the filtered catalog from page 1 to the end, so
        callers never maintain ``page_no`` / ``total_count`` by hand.
        ``page_size`` defaults to 100 to cut round-trips. Each yielded
        :class:`Model` carries the injected ``served_region`` / ``workspace``
        / ``service_site`` context. Only the current Client Region is read.
        Pages share the catalog cache; ``refresh=True`` forces every page to
        be fetched fresh.
        """
        page = self.list(
            page_no=1,
            page_size=page_size,
            provider=provider,
            name=name,
            model=model,
            language=language,
            providers=providers,
            inference_providers=inference_providers,
            capabilities=capabilities,
            features=features,
            context_window=context_window,
            service_site=service_site,
            supports=supports,
            deployment_methods=deployment_methods,
            deployment_ptu_service_tiers=deployment_ptu_service_tiers,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
            refresh=refresh,
        )
        for following in page.iter_pages():
            yield from following

    def list_all(
        self,
        *,
        page_size: int = 100,
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
        refresh: bool = False,
    ) -> List[Model]:
        """Collect every model across all pages into one list.

        Eager counterpart of :meth:`iter` with the same filters; walks the
        whole filtered catalog of the current Client Region. Prefer
        :meth:`iter` when the catalog is large or only a prefix is needed.
        """
        return list(
            self.iter(
                page_size=page_size,
                provider=provider,
                name=name,
                model=model,
                language=language,
                providers=providers,
                inference_providers=inference_providers,
                capabilities=capabilities,
                features=features,
                context_window=context_window,
                service_site=service_site,
                supports=supports,
                deployment_methods=deployment_methods,
                deployment_ptu_service_tiers=deployment_ptu_service_tiers,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
                refresh=refresh,
            ),
        )

    def is_available(
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
        refresh: bool = False,
    ) -> bool:
        """Whether ``model`` is in the current Client Region's catalog.

        Answers only for this Client's Region/endpoint and never queries
        other Regions implicitly (use a per-Region Client for that). Returns
        ``True`` when the catalog query returns a matching entry, ``False``
        when it returns none. Transport/HTTP errors propagate rather than
        being read as "unavailable". ``model`` must be a non-empty id, so an
        empty filter cannot masquerade as the whole catalog being a match.
        Shares the catalog cache with ``get``; ``refresh=True`` forces a
        fresh fetch.
        """
        if not model or not model.strip():
            raise ValueError("model must be a non-empty model id")
        resp = self._models_get(
            {"model": model, "page_no": 1, "page_size": 1},
            refresh=refresh,
            raw=False,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
        )
        _check_catalog_probe(resp, self._client.raise_on_error)
        return bool(_parse_get_models(resp.json()))

    def _find_catalog_model(
        self,
        model: str,
        *,
        timeout: Optional[float],
        refresh: bool,
    ) -> Optional[Model]:
        """Probe and parse the RegionRouter entry with one catalog read."""
        resp = self._models_get(
            {"model": model, "page_no": 1, "page_size": 1},
            refresh=refresh,
            raw=False,
            timeout=timeout,
            api_key=None,
            workspace=None,
            options=None,
        )
        _check_catalog_probe(resp, self._client.raise_on_error)
        entries = _parse_get_models(resp.json())
        if not entries:
            return None
        return _parse_catalog_model(
            resp,
            entries[0],
            region=self._client.config.region,
            workspace=self._client.config.workspace,
            strict=self._client.config.strict_validation,
        )


class AsyncModels:
    def __init__(self, client) -> None:
        self._client = client
        self.limits = AsyncModelLimits(client)
        self.permissions = AsyncModelPermissions(client)
        self._catalog_cache = _CatalogCache()

    async def _models_get(
        self,
        params: Mapping[str, Any],
        *,
        refresh: bool,
        raw: bool,
        timeout: Optional[float],
        api_key: Optional[str],
        workspace: Optional[str],
        options: Any,
    ) -> "APIResponse":
        """Async :meth:`Models._models_get`; same cache, awaited fetch.

        The cache lookup/store are synchronous and never await inside the
        lock, so sharing one ``_CatalogCache`` on a single event loop is
        safe. Like the sync seam, options carrying ``extra_query`` /
        ``extra_headers`` / ``extra_body`` bypass the cache.
        """
        if raw:
            return await self._client.transport.request_api(
                "GET",
                _BASE_PATH,
                params=params,
                sdk_module=MODULE,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                options=options,
                client=self._client,
            )
        # An injected client's mutable defaults, auth or hooks can change
        # the actual request independently of the SDK's catalog key.
        cacheable = self._client.transport.owns_http_client and (
            options is None
            or not (
                options.extra_query
                or options.extra_headers
                or options.extra_body
            )
        )
        key = catalog_key(
            self._client.config,
            params,
            api_key=api_key,
            workspace=workspace,
        )
        if cacheable and not refresh:
            cached = self._catalog_cache.get(key)
            if cached is not None:
                return cached
        resp = await self._client.transport.request_api(
            "GET",
            _BASE_PATH,
            params=params,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=options,
            client=self._client,
        )
        resp.raise_for_status()
        if cacheable and response_error(resp.http_response) is None:
            self._catalog_cache.set(key, resp)
        return resp

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
        refresh: bool = False,
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
        refresh: bool = False,
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
        refresh: bool = False,
        raw: bool = False,
    ) -> Union[ModelList, "APIResponse"]:
        filter_kwargs: Dict[str, Any] = {
            "name": name,
            "model": model,
            "language": language,
            "provider": provider,
            "providers": providers,
            "inference_providers": inference_providers,
            "capabilities": capabilities,
            "features": features,
            "context_window": context_window,
            "service_site": service_site,
            "supports": supports,
            "deployment_methods": deployment_methods,
            "deployment_ptu_service_tiers": deployment_ptu_service_tiers,
        }
        resp = await self._models_get(
            _build_list_params(page_no, page_size, **filter_kwargs),
            refresh=refresh,
            raw=raw,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
        )
        if raw:
            return resp
        result = _parse_list(
            resp.json(),
            strict=self._client.config.strict_validation,
            response=resp.http_response,
            region=self._client.config.region,
            workspace=(
                workspace
                if workspace is not None
                else self._client.config.workspace
            ),
            service_site=service_site,
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
                refresh=refresh,
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
        refresh: bool = False,
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
        refresh: bool = False,
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
        refresh: bool = False,
        raw: bool = False,
    ) -> Union[Model, "APIResponse"]:
        # Server filters by model name via query params. Shares the catalog
        # cache with list/is_available; ``refresh=True`` forces a fresh
        # fetch, errors and ``raw=True`` are never cached.
        _require_model_id(model)
        resp = await self._models_get(
            {"model": model, "page_no": 1, "page_size": 1},
            refresh=refresh,
            raw=raw,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
        )
        if raw:
            return resp
        body = resp.json()
        models = _parse_get_models(body)
        region = self._client.config.region
        effective_workspace = (
            workspace
            if workspace is not None
            else self._client.config.workspace
        )
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
                    {
                        **response_envelope(body),
                        "served_region": region,
                        "workspace": effective_workspace,
                    },
                    strict=self._client.config.strict_validation,
                    response=resp.http_response,
                ),
                resp.http_response,
            )
        return _parse_catalog_model(
            resp,
            models[0],
            region=region,
            workspace=effective_workspace,
            strict=self._client.config.strict_validation,
        )

    async def iter(
        self,
        *,
        page_size: int = 100,
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
        refresh: bool = False,
    ) -> AsyncIterator[Model]:
        """Async :meth:`Models.iter`: ``async for m in client.models.iter(...)``.

        Walks every page of the current Client Region's filtered catalog,
        fetching the next page lazily only as iteration advances. Each model
        carries the injected ``served_region`` / ``workspace`` /
        ``service_site`` context. Pages share the catalog cache;
        ``refresh=True`` forces every page to be fetched fresh.
        """
        page = await self.list(
            page_no=1,
            page_size=page_size,
            provider=provider,
            name=name,
            model=model,
            language=language,
            providers=providers,
            inference_providers=inference_providers,
            capabilities=capabilities,
            features=features,
            context_window=context_window,
            service_site=service_site,
            supports=supports,
            deployment_methods=deployment_methods,
            deployment_ptu_service_tiers=deployment_ptu_service_tiers,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
            refresh=refresh,
        )
        async for following in page.iter_pages_async():
            for item in following.items:
                yield item

    async def list_all(
        self,
        *,
        page_size: int = 100,
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
        refresh: bool = False,
    ) -> List[Model]:
        """Async :meth:`Models.list_all`; collects the whole filtered catalog."""
        return [
            m
            async for m in self.iter(
                page_size=page_size,
                provider=provider,
                name=name,
                model=model,
                language=language,
                providers=providers,
                inference_providers=inference_providers,
                capabilities=capabilities,
                features=features,
                context_window=context_window,
                service_site=service_site,
                supports=supports,
                deployment_methods=deployment_methods,
                deployment_ptu_service_tiers=deployment_ptu_service_tiers,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
                refresh=refresh,
            )
        ]

    async def is_available(
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
        refresh: bool = False,
    ) -> bool:
        """Async :meth:`Models.is_available`; current Client Region only.

        Shares the catalog cache with ``get``; ``refresh=True`` forces a
        fresh fetch.
        """
        if not model or not model.strip():
            raise ValueError("model must be a non-empty model id")
        resp = await self._models_get(
            {"model": model, "page_no": 1, "page_size": 1},
            refresh=refresh,
            raw=False,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
        )
        _check_catalog_probe(resp, self._client.raise_on_error)
        return bool(_parse_get_models(resp.json()))

    async def _find_catalog_model(
        self,
        model: str,
        *,
        timeout: Optional[float],
        refresh: bool,
    ) -> Optional[Model]:
        """Probe and parse the RegionRouter entry with one catalog read."""
        resp = await self._models_get(
            {"model": model, "page_no": 1, "page_size": 1},
            refresh=refresh,
            raw=False,
            timeout=timeout,
            api_key=None,
            workspace=None,
            options=None,
        )
        _check_catalog_probe(resp, self._client.raise_on_error)
        entries = _parse_get_models(resp.json())
        if not entries:
            return None
        return _parse_catalog_model(
            resp,
            entries[0],
            region=self._client.config.region,
            workspace=self._client.config.workspace,
            strict=self._client.config.strict_validation,
        )


class ModelLimits:
    """Read-only model rate-limit quotas (``GET /models/limits``).

    Reports the account-level (``model_limit``) and workspace-level
    (``workspace_limit``) RPM/QPS, token-usage and async-task caps for the
    current Client Region/workspace, so callers can plan concurrency and
    choose between realtime and Batch without waiting for a 429. Updating
    limits is a management-plane write and is not exposed here.
    """

    def __init__(self, client) -> None:
        self._client = client

    @overload
    def list(
        self,
        *,
        name: Optional[str] = None,
        model: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> QuotaList:
        ...

    @overload
    def list(
        self,
        *,
        name: Optional[str] = None,
        model: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 20,
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
        name: Optional[str] = None,
        model: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[QuotaList, "APIResponse"]:
        """List model quotas; ``name`` fuzzy-matches, ``model`` is exact.

        Filters are server-side, so ``total_count`` is the filtered total and
        ``iter_pages()`` walks the filtered set. ``raw=True`` returns the
        :class:`~dashscope.transport.APIResponse` before any error check.
        """
        resp = self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/limits",
            params=_build_limits_params(
                page_no,
                page_size,
                name=name,
                model=model,
            ),
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
        result = _parse_meta_page(
            QuotaList,
            resp.json(),
            items_key="quotas",
            strict=self._client.config.strict_validation,
            response=resp.http_response,
        )
        result.bind_pagination(
            lambda next_no, next_size: self.list(
                name=name,
                model=model,
                page_no=next_no,
                page_size=next_size,
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
    ) -> ModelQuota:
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
    ) -> Union[ModelQuota, "APIResponse"]:
        """Fetch one model's quotas by exact id.

        Raises ``Model.NotExist`` (``NotFoundError``) when the gateway returns
        no quota for ``model`` and ``raise_on_error`` is set; otherwise returns
        an empty :class:`ModelQuota` envelope. ``raw=True`` skips the check.
        """
        _require_model_id(model)
        resp = self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/limits",
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
        items = _meta_items(body, "quotas")
        if not items:
            if self._client.raise_on_error:
                raise error_from_business_code(
                    message=f"Model '{model}' not found",
                    code=ErrorCode.MODEL_NOT_EXIST.value,
                    body=body,
                    headers=resp.http_response.headers,
                )
            return with_headers(
                parse_model(
                    ModelQuota,
                    response_envelope(body),
                    strict=self._client.config.strict_validation,
                    response=resp.http_response,
                ),
                resp.http_response,
            )
        return with_headers(
            parse_model(
                ModelQuota,
                {**items[0], **response_envelope(body)},
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )


class AsyncModelLimits:
    """Async :class:`ModelLimits` (``GET /models/limits``)."""

    def __init__(self, client) -> None:
        self._client = client

    @overload
    async def list(
        self,
        *,
        name: Optional[str] = None,
        model: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> QuotaList:
        ...

    @overload
    async def list(
        self,
        *,
        name: Optional[str] = None,
        model: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 20,
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
        name: Optional[str] = None,
        model: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[QuotaList, "APIResponse"]:
        """Async :meth:`ModelLimits.list`."""
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/limits",
            params=_build_limits_params(
                page_no,
                page_size,
                name=name,
                model=model,
            ),
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
        result = _parse_meta_page(
            QuotaList,
            resp.json(),
            items_key="quotas",
            strict=self._client.config.strict_validation,
            response=resp.http_response,
        )
        result.bind_pagination(
            lambda next_no, next_size: self.list(
                name=name,
                model=model,
                page_no=next_no,
                page_size=next_size,
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
    ) -> ModelQuota:
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
    ) -> Union[ModelQuota, "APIResponse"]:
        """Async :meth:`ModelLimits.get`."""
        _require_model_id(model)
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/limits",
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
        items = _meta_items(body, "quotas")
        if not items:
            if self._client.raise_on_error:
                raise error_from_business_code(
                    message=f"Model '{model}' not found",
                    code=ErrorCode.MODEL_NOT_EXIST.value,
                    body=body,
                    headers=resp.http_response.headers,
                )
            return with_headers(
                parse_model(
                    ModelQuota,
                    response_envelope(body),
                    strict=self._client.config.strict_validation,
                    response=resp.http_response,
                ),
                resp.http_response,
            )
        return with_headers(
            parse_model(
                ModelQuota,
                {**items[0], **response_envelope(body)},
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )


class ModelPermissions:
    """Read-only model authorization state (``GET /models/permissions``).

    Reports which models are authorizable or already authorized in the current
    workspace and their per-action flags (inference / fine_tune / deploy), so
    callers can diagnose authorization before an inference permission error.
    ``authorization_scope`` selects AUTHORIZABLE (default) or AUTHORIZED;
    ``action`` (currently only INFERENCE) drives the filter when AUTHORIZED.
    Granting/revoking authorization is a write and is not exposed here.
    """

    def __init__(self, client) -> None:
        self._client = client

    @overload
    def list(
        self,
        *,
        name: Optional[str] = None,
        model: Optional[str] = None,
        authorization_scope: Optional[str] = None,
        action: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> PermissionList:
        ...

    @overload
    def list(
        self,
        *,
        name: Optional[str] = None,
        model: Optional[str] = None,
        authorization_scope: Optional[str] = None,
        action: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 20,
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
        name: Optional[str] = None,
        model: Optional[str] = None,
        authorization_scope: Optional[str] = None,
        action: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[PermissionList, "APIResponse"]:
        """List model permissions; ``name`` fuzzy-matches, ``model`` is exact.

        Omitting ``authorization_scope`` lets the gateway default to
        AUTHORIZABLE. Filters are server-side, so ``total_count`` is the
        filtered total and ``iter_pages()`` walks the filtered set.
        ``raw=True`` returns the APIResponse before any error check.
        """
        resp = self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/permissions",
            params=_build_permissions_params(
                page_no,
                page_size,
                name=name,
                model=model,
                authorization_scope=authorization_scope,
                action=action,
            ),
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
        result = _parse_meta_page(
            PermissionList,
            resp.json(),
            items_key="permissions",
            strict=self._client.config.strict_validation,
            response=resp.http_response,
        )
        result.bind_pagination(
            lambda next_no, next_size: self.list(
                name=name,
                model=model,
                authorization_scope=authorization_scope,
                action=action,
                page_no=next_no,
                page_size=next_size,
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
        authorization_scope: Optional[str] = None,
        action: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> ModelPermission:
        ...

    @overload
    def get(
        self,
        model: str,
        *,
        authorization_scope: Optional[str] = None,
        action: Optional[str] = None,
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
        authorization_scope: Optional[str] = None,
        action: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ModelPermission, "APIResponse"]:
        """Fetch one model's authorization by exact id.

        Pass ``authorization_scope="AUTHORIZED"`` to read the granted flags;
        the default AUTHORIZABLE lists models that *can* be authorized. Raises
        ``Model.NotExist`` when no entry matches and ``raise_on_error`` is set;
        otherwise returns an empty :class:`ModelPermission` envelope.
        """
        _require_model_id(model)
        resp = self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/permissions",
            params=_build_permissions_params(
                1,
                1,
                name=None,
                model=model,
                authorization_scope=authorization_scope,
                action=action,
            ),
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
        items = _meta_items(body, "permissions")
        if not items:
            if self._client.raise_on_error:
                raise error_from_business_code(
                    message=f"Model '{model}' not found",
                    code=ErrorCode.MODEL_NOT_EXIST.value,
                    body=body,
                    headers=resp.http_response.headers,
                )
            return with_headers(
                parse_model(
                    ModelPermission,
                    response_envelope(body),
                    strict=self._client.config.strict_validation,
                    response=resp.http_response,
                ),
                resp.http_response,
            )
        return with_headers(
            parse_model(
                ModelPermission,
                {**items[0], **response_envelope(body)},
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )


class AsyncModelPermissions:
    """Async :class:`ModelPermissions` (``GET /models/permissions``)."""

    def __init__(self, client) -> None:
        self._client = client

    @overload
    async def list(
        self,
        *,
        name: Optional[str] = None,
        model: Optional[str] = None,
        authorization_scope: Optional[str] = None,
        action: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> PermissionList:
        ...

    @overload
    async def list(
        self,
        *,
        name: Optional[str] = None,
        model: Optional[str] = None,
        authorization_scope: Optional[str] = None,
        action: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 20,
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
        name: Optional[str] = None,
        model: Optional[str] = None,
        authorization_scope: Optional[str] = None,
        action: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 20,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[PermissionList, "APIResponse"]:
        """Async :meth:`ModelPermissions.list`."""
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/permissions",
            params=_build_permissions_params(
                page_no,
                page_size,
                name=name,
                model=model,
                authorization_scope=authorization_scope,
                action=action,
            ),
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
        result = _parse_meta_page(
            PermissionList,
            resp.json(),
            items_key="permissions",
            strict=self._client.config.strict_validation,
            response=resp.http_response,
        )
        result.bind_pagination(
            lambda next_no, next_size: self.list(
                name=name,
                model=model,
                authorization_scope=authorization_scope,
                action=action,
                page_no=next_no,
                page_size=next_size,
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
        authorization_scope: Optional[str] = None,
        action: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> ModelPermission:
        ...

    @overload
    async def get(
        self,
        model: str,
        *,
        authorization_scope: Optional[str] = None,
        action: Optional[str] = None,
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
        authorization_scope: Optional[str] = None,
        action: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ModelPermission, "APIResponse"]:
        """Async :meth:`ModelPermissions.get`."""
        _require_model_id(model)
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/permissions",
            params=_build_permissions_params(
                1,
                1,
                name=None,
                model=model,
                authorization_scope=authorization_scope,
                action=action,
            ),
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
        items = _meta_items(body, "permissions")
        if not items:
            if self._client.raise_on_error:
                raise error_from_business_code(
                    message=f"Model '{model}' not found",
                    code=ErrorCode.MODEL_NOT_EXIST.value,
                    body=body,
                    headers=resp.http_response.headers,
                )
            return with_headers(
                parse_model(
                    ModelPermission,
                    response_envelope(body),
                    strict=self._client.config.strict_validation,
                    response=resp.http_response,
                ),
                resp.http_response,
            )
        return with_headers(
            parse_model(
                ModelPermission,
                {**items[0], **response_envelope(body)},
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )
