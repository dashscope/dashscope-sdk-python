# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.deployments resource (design doc M3 3.6).

Wire contract follows the 1.x ``Deployments`` REST surface:
``POST /deployments``, ``GET /deployments``, ``GET /deployments/{id}``,
``PUT /deployments/{id}/scale``, ``DELETE /deployments/{id}``,
``GET /deployments/{id}/stream`` (SSE deployment events).
"""

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

from dashscope.resources._ptu import (
    AsyncCapacityInstances,
    AsyncCapacityOperations,
    CapacityInstance,
    CapacityInstanceList,
    CapacityInstances,
    CapacityOperation,
    CapacityOperations,
    PrePaidInfo,
    PtuCapacity,
    _idempotency_headers,
    _pre_paid_info_body,
    _ptu_capacity_body,
    _purchase_retries,
    _renew_body,
    _validate_order_type,
)
from dashscope.resources._sse import (
    _aiter_sse,
    _iter_sse,
    _log_stream_error,
    _stream_error,
    _try_json,
)
from dashscope.transport._options import request_options
from dashscope.types._parse import parse_model
from dashscope.types.common import (
    JobStatus,
    ResponseBase,
    flatten_envelope,
    response_envelope,
    with_headers,
)

if TYPE_CHECKING:
    import httpx

    from dashscope.transport._response import APIResponse

_BASE_PATH = "/deployments"

# 1.x tags this API "finetune" (Deployments lives in dashscope.finetune).
MODULE = "finetune"

__all__ = [
    "AsyncCapacityInstances",
    "AsyncCapacityOperations",
    "AsyncDeployments",
    "CapacityInstance",
    "CapacityInstanceList",
    "CapacityInstances",
    "CapacityOperation",
    "CapacityOperations",
    "Deployment",
    "DeploymentEvent",
    "DeploymentList",
    "Deployments",
    "PrePaidInfo",
    "PtuCapacity",
]


class Deployment(ResponseBase):
    """One model deployment.

    The same type comes back from ``create`` / ``get`` / ``scale`` and is
    nested in ``DeploymentList.deployments``, so there is one shape to learn;
    the response envelope therefore rides along on the nested copies too,
    empty there because the server reports it once per response.
    """

    deployed_model: Optional[str] = None
    model: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("model", "model_name"),
    )
    base_model: Optional[str] = None
    status: Optional[str] = None
    capacity: Optional[int] = None
    base_capacity: Optional[int] = None
    ready_capacity: Optional[int] = None
    workspace_id: Optional[str] = None
    charge_type: Optional[str] = None
    created_at: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("created_at", "gmt_create"),
    )
    updated_at: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("updated_at", "gmt_modified"),
    )
    modifier: Optional[str] = None
    creator: Optional[str] = None
    # PTU throughput-reservation fields (plan="ptu"); empty for MU/lora.
    name: Optional[str] = None
    plan: Optional[str] = None
    ptu_service_tier: Optional[str] = None
    ptu_capacity: Optional[PtuCapacity] = None
    pre_paid_info: Optional[PrePaidInfo] = None
    pre_paid_instance_id: Optional[str] = None
    pre_paid_gmt_expired: Optional[str] = None
    overflow_strategy: Optional[str] = None
    fail_reason: Optional[str] = None
    operation_id: Optional[str] = None
    instance_id: Optional[str] = None


class DeploymentList(ResponseBase):
    page_no: int = 1
    page_size: int = 10
    total_count: int = 0
    deployments: List[Deployment] = []


class DeploymentEvent(BaseModel):
    """One SSE frame from ``/deployments/{id}/stream``."""

    model_config = ConfigDict(extra="allow")

    output: Union[Dict[str, Any], str, None] = None
    is_error: bool = False


def _parse(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> Deployment:
    return parse_model(
        Deployment,
        flatten_envelope(body),
        strict=strict,
        response=response,
    )


def _parse_list(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> DeploymentList:
    data = flatten_envelope(body)
    deployments = data.get("deployments")
    if not isinstance(deployments, list):
        # A retyped ``deployments`` would poison the ``len()`` below (a
        # string's length is not a page total). Empty is the honest
        # reading of "not a list", matching ``models._parse_list``.
        deployments = []
    return parse_model(
        DeploymentList,
        {
            "page_no": data.get("page_no", 1),
            "page_size": data.get("page_size", 10),
            "total_count": data.get("total", len(deployments)),
            "deployments": deployments,
            **response_envelope(data),
        },
        strict=strict,
        response=response,
    )


def _parse_status(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> JobStatus:
    return parse_model(
        JobStatus,
        flatten_envelope(body),
        strict=strict,
        response=response,
    )


def _parse_event(
    data: Any,
    *,
    is_error: bool = False,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> DeploymentEvent:
    """One SSE frame, degrading rather than breaking the iteration.

    A drifted frame used to raise a bare ``ValidationError`` partway
    through the generator -- earlier frames were already consumed and
    ``except DashScopeError`` could not catch it. ``response`` is the
    stream itself, which is the only response a frame belongs to.
    """
    return parse_model(
        DeploymentEvent,
        {"output": _try_json(data), "is_error": is_error},
        strict=strict,
        response=response,
    )


def _build_create_body(
    model: str,
    capacity: Optional[int],
    model_version: Optional[str],
    suffix: Optional[str],
    plan: Optional[str],
    template_id: Optional[str],
    deploy_spec: Optional[Dict[str, Any]],
    billing_method: Optional[str],
    service_tier: Optional[str] = None,
    charge_type: Optional[str] = None,
    ptu_capacity: Any = None,
    pre_paid_info: Any = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    # Wire spellings; the public names are ``model`` / ``model_version``.
    body: Dict[str, Any] = {"model_name": model}
    if capacity is not None:
        body["capacity"] = capacity
    if model_version is not None:
        body["model_version"] = model_version
    if suffix is not None:
        body["suffix"] = suffix
    if plan is not None:
        body["plan"] = plan
    if template_id is not None:
        body["template_id"] = template_id

    # MU deployments reject the create call unless both are present.
    if deploy_spec is not None:
        body["deploy_spec"] = deploy_spec
    if billing_method is not None:
        body["billing_method"] = billing_method

    # PTU throughput reservation: capacity is expressed as ptu_capacity
    # (kTPM), not instance count. charge_type and ptu_capacity are required;
    # pre_paid billing additionally requires pre_paid_info.
    if name is not None:
        body["name"] = name
    if service_tier is not None:
        body["service_tier"] = service_tier
    if charge_type is not None:
        body["charge_type"] = charge_type
    if ptu_capacity is not None:
        body["ptu_capacity"] = _ptu_capacity_body(ptu_capacity)
    if pre_paid_info is not None:
        body["pre_paid_info"] = _pre_paid_info_body(pre_paid_info)
    if plan == "ptu":
        if charge_type is None:
            raise ValueError("plan='ptu' requires charge_type")
        if ptu_capacity is None:
            raise ValueError("plan='ptu' requires ptu_capacity")
        if charge_type == "pre_paid" and pre_paid_info is None:
            raise ValueError("pre_paid charge_type requires pre_paid_info")
    return body


def _build_scale_body(
    deployed_model: str,
    capacity: Optional[int],
    ptu_capacity: Any,
    instance_id: Optional[str],
    pre_paid_info: Any,
    order_type: Optional[str],
) -> Dict[str, Any]:
    if capacity is None and ptu_capacity is None:
        raise ValueError("scale requires capacity or ptu_capacity")
    _validate_order_type(order_type)
    body: Dict[str, Any] = {"deployed_model": deployed_model}
    if capacity is not None:
        body["capacity"] = capacity
    if ptu_capacity is not None:
        # PTU scale is an ABSOLUTE target, not an increment.
        body["ptu_capacity"] = _ptu_capacity_body(ptu_capacity)
    if instance_id is not None:
        body["instance_id"] = instance_id
    if pre_paid_info is not None:
        body["pre_paid_info"] = _pre_paid_info_body(pre_paid_info)
    if order_type is not None:
        body["order_type"] = order_type
    return body


class Deployments:
    def __init__(self, client) -> None:
        self._client = client
        self.capacity_instances = CapacityInstances(client)
        self.capacity_operations = CapacityOperations(client)

    @overload
    def create(
        self,
        *,
        model: str,
        capacity: Optional[int] = None,
        model_version: Optional[str] = None,
        suffix: Optional[str] = None,
        plan: Optional[str] = None,
        template_id: Optional[str] = None,
        deploy_spec: Optional[Dict[str, Any]] = None,
        billing_method: Optional[str] = None,
        service_tier: Optional[str] = None,
        charge_type: Optional[str] = None,
        ptu_capacity: Any = None,
        pre_paid_info: Any = None,
        name: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    def create(
        self,
        *,
        model: str,
        capacity: Optional[int] = None,
        model_version: Optional[str] = None,
        suffix: Optional[str] = None,
        plan: Optional[str] = None,
        template_id: Optional[str] = None,
        deploy_spec: Optional[Dict[str, Any]] = None,
        billing_method: Optional[str] = None,
        service_tier: Optional[str] = None,
        charge_type: Optional[str] = None,
        ptu_capacity: Any = None,
        pre_paid_info: Any = None,
        name: Optional[str] = None,
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
        capacity: Optional[int] = None,
        model_version: Optional[str] = None,
        suffix: Optional[str] = None,
        plan: Optional[str] = None,
        template_id: Optional[str] = None,
        deploy_spec: Optional[Dict[str, Any]] = None,
        billing_method: Optional[str] = None,
        service_tier: Optional[str] = None,
        charge_type: Optional[str] = None,
        ptu_capacity: Any = None,
        pre_paid_info: Any = None,
        name: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Create a deployment of ``model``.

        For MU/lora plans pass ``capacity`` (instance count). For PTU
        throughput reservation pass ``plan="ptu"`` with ``charge_type``
        (``pre_paid`` / ``post_paid``) and ``ptu_capacity`` (kTPM mapping);
        ``pre_paid`` additionally requires ``pre_paid_info``. Optional PTU
        fields: ``service_tier`` (``ptu_fast`` default / ``ptu_default``),
        ``name``, ``suffix``.

        A PTU create is a billable purchase, so it defaults to
        ``max_retries=0`` (a blind retry could spawn a duplicate ModelCode).
        The first ModelCode creation is deliberately NOT given an
        idempotency key -- the contract reserves that for instance writes;
        on a create timeout, confirm the existing ModelCode before retrying.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        body = _build_create_body(
            model,
            capacity,
            model_version,
            suffix,
            plan,
            template_id,
            deploy_spec,
            billing_method,
            service_tier,
            charge_type,
            ptu_capacity,
            pre_paid_info,
            name,
        )
        if plan == "ptu":
            max_retries = _purchase_retries(max_retries)
        resp = self._client.transport.request_api(
            "POST",
            _BASE_PATH,
            json=body,
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
        return with_headers(
            _parse(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    def get(
        self,
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    def get(
        self,
        deployed_model: str,
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
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Fetch one deployment by ``deployed_model`` id.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/{deployed_model}",
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
        return with_headers(
            _parse(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> DeploymentList:
        ...

    @overload
    def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
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
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[DeploymentList, "APIResponse"]:
        """List deployments, one page at a time.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = self._client.transport.request_api(
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
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_list(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    def scale(
        self,
        deployed_model: str,
        *,
        capacity: Optional[int] = None,
        ptu_capacity: Any = None,
        instance_id: Optional[str] = None,
        pre_paid_info: Any = None,
        order_type: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    def scale(
        self,
        deployed_model: str,
        *,
        capacity: Optional[int] = None,
        ptu_capacity: Any = None,
        instance_id: Optional[str] = None,
        pre_paid_info: Any = None,
        order_type: Optional[str] = None,
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

    def scale(
        self,
        deployed_model: str,
        *,
        capacity: Optional[int] = None,
        ptu_capacity: Any = None,
        instance_id: Optional[str] = None,
        pre_paid_info: Any = None,
        order_type: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Resize a deployment.

        MU/lora: pass ``capacity`` (instance count). PTU: pass
        ``ptu_capacity`` as an ABSOLUTE kTPM target (not an increment),
        optionally ``instance_id`` (required when the ModelCode holds more
        than one instance, else CAPACITY_INSTANCE_REQUIRED),
        ``pre_paid_info`` and ``order_type`` (UPGRADE / DOWNGRADE). A PTU
        scale is billable, so it defaults to ``max_retries=0`` and carries
        an idempotency key. Scaling to all-zero is not a delete.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        body = _build_scale_body(
            deployed_model,
            capacity,
            ptu_capacity,
            instance_id,
            pre_paid_info,
            order_type,
        )
        headers = extra_headers
        if ptu_capacity is not None:
            max_retries = _purchase_retries(max_retries)
            headers = _idempotency_headers(extra_headers)
        resp = self._client.transport.request_api(
            "PUT",
            f"{_BASE_PATH}/{deployed_model}/scale",
            json=body,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                headers,
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

    @overload
    def renew(
        self,
        deployed_model: str,
        *,
        pre_paid_info: Any,
        instance_id: Optional[str] = None,
        is_change: Optional[bool] = None,
        ptu_capacity: Any = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    def renew(
        self,
        deployed_model: str,
        *,
        pre_paid_info: Any,
        instance_id: Optional[str] = None,
        is_change: Optional[bool] = None,
        ptu_capacity: Any = None,
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

    def renew(
        self,
        deployed_model: str,
        *,
        pre_paid_info: Any,
        instance_id: Optional[str] = None,
        is_change: Optional[bool] = None,
        ptu_capacity: Any = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Renew a pre-paid PTU ModelCode (no ``order_type`` on renew).

        ``pre_paid_info`` is required; ``instance_id`` selects one instance
        when the ModelCode holds several. Changing ``ptu_capacity`` requires
        ``is_change=True``. Billable, so defaults to ``max_retries=0`` with
        an idempotency key. Returns the refreshed :class:`Deployment`.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check.
        """
        body = _renew_body(pre_paid_info, is_change, ptu_capacity, instance_id)
        resp = self._client.transport.request_api(
            "PUT",
            f"{_BASE_PATH}/{deployed_model}/renew",
            json=body,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                _idempotency_headers(extra_headers),
                extra_query,
                extra_body,
                max_retries=_purchase_retries(max_retries),
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

    @overload
    def update_overflow_strategy(
        self,
        deployed_model: str,
        *,
        overflow_strategy: str,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    def update_overflow_strategy(
        self,
        deployed_model: str,
        *,
        overflow_strategy: str,
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

    def update_overflow_strategy(
        self,
        deployed_model: str,
        *,
        overflow_strategy: str,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Set the whole-ModelCode overflow strategy (lowercase).

        ``enable`` lets traffic beyond the reserved capacity overflow to
        pay-as-you-go (extra cost); ``disable`` throttles it. This is a
        config toggle, not a purchase, so normal retries apply.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check.
        """
        if overflow_strategy not in ("enable", "disable"):
            raise ValueError("overflow_strategy must be 'enable' or 'disable'")
        resp = self._client.transport.request_api(
            "PUT",
            f"{_BASE_PATH}/{deployed_model}/update-overflowstrategy",
            json={"overflow_strategy": overflow_strategy},
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
        return with_headers(
            _parse(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    def delete(
        self,
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> JobStatus:
        ...

    @overload
    def delete(
        self,
        deployed_model: str,
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

    def delete(
        self,
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[JobStatus, "APIResponse"]:
        """Delete a deployment; the job status reports the teardown.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = self._client.transport.request_api(
            "DELETE",
            f"{_BASE_PATH}/{deployed_model}",
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
        return with_headers(
            _parse_status(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    def stream_events(
        self,
        deployed_model_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> Iterator[DeploymentEvent]:
        response = self._client.transport.open_stream(
            "GET",
            f"{_BASE_PATH}/{deployed_model_id}/stream",
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
        )
        try:
            for is_error, status_code, data in _iter_sse(response):
                if is_error:
                    if self._client.raise_on_error:
                        raise _stream_error(
                            status_code,
                            data,
                            response.headers,
                        )
                    _log_stream_error(data, response.headers)
                    yield _parse_event(
                        data,
                        is_error=True,
                        strict=self._client.config.strict_validation,
                        response=response,
                    )
                    return
                yield _parse_event(
                    data,
                    strict=self._client.config.strict_validation,
                    response=response,
                )
        finally:
            response.close()


class AsyncDeployments:
    def __init__(self, client) -> None:
        self._client = client
        self.capacity_instances = AsyncCapacityInstances(client)
        self.capacity_operations = AsyncCapacityOperations(client)

    @overload
    async def create(
        self,
        *,
        model: str,
        capacity: Optional[int] = None,
        model_version: Optional[str] = None,
        suffix: Optional[str] = None,
        plan: Optional[str] = None,
        template_id: Optional[str] = None,
        deploy_spec: Optional[Dict[str, Any]] = None,
        billing_method: Optional[str] = None,
        service_tier: Optional[str] = None,
        charge_type: Optional[str] = None,
        ptu_capacity: Any = None,
        pre_paid_info: Any = None,
        name: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    async def create(
        self,
        *,
        model: str,
        capacity: Optional[int] = None,
        model_version: Optional[str] = None,
        suffix: Optional[str] = None,
        plan: Optional[str] = None,
        template_id: Optional[str] = None,
        deploy_spec: Optional[Dict[str, Any]] = None,
        billing_method: Optional[str] = None,
        service_tier: Optional[str] = None,
        charge_type: Optional[str] = None,
        ptu_capacity: Any = None,
        pre_paid_info: Any = None,
        name: Optional[str] = None,
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
        capacity: Optional[int] = None,
        model_version: Optional[str] = None,
        suffix: Optional[str] = None,
        plan: Optional[str] = None,
        template_id: Optional[str] = None,
        deploy_spec: Optional[Dict[str, Any]] = None,
        billing_method: Optional[str] = None,
        service_tier: Optional[str] = None,
        charge_type: Optional[str] = None,
        ptu_capacity: Any = None,
        pre_paid_info: Any = None,
        name: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Create a deployment of ``model`` (async twin of ``Deployments.create``).

        PTU throughput reservation (``plan="ptu"``) is a billable purchase
        and defaults to ``max_retries=0``; see the sync method for the full
        contract and safeguards.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check.
        """
        body = _build_create_body(
            model,
            capacity,
            model_version,
            suffix,
            plan,
            template_id,
            deploy_spec,
            billing_method,
            service_tier,
            charge_type,
            ptu_capacity,
            pre_paid_info,
            name,
        )
        if plan == "ptu":
            max_retries = _purchase_retries(max_retries)
        resp = await self._client.transport.request_api(
            "POST",
            _BASE_PATH,
            json=body,
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
        return with_headers(
            _parse(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    async def get(
        self,
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    async def get(
        self,
        deployed_model: str,
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
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Fetch one deployment by ``deployed_model`` id.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/{deployed_model}",
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
        return with_headers(
            _parse(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    async def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> DeploymentList:
        ...

    @overload
    async def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
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
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[DeploymentList, "APIResponse"]:
        """List deployments, one page at a time.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = await self._client.transport.request_api(
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
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_list(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    async def scale(
        self,
        deployed_model: str,
        *,
        capacity: Optional[int] = None,
        ptu_capacity: Any = None,
        instance_id: Optional[str] = None,
        pre_paid_info: Any = None,
        order_type: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    async def scale(
        self,
        deployed_model: str,
        *,
        capacity: Optional[int] = None,
        ptu_capacity: Any = None,
        instance_id: Optional[str] = None,
        pre_paid_info: Any = None,
        order_type: Optional[str] = None,
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

    async def scale(
        self,
        deployed_model: str,
        *,
        capacity: Optional[int] = None,
        ptu_capacity: Any = None,
        instance_id: Optional[str] = None,
        pre_paid_info: Any = None,
        order_type: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Resize a deployment (async twin of ``Deployments.scale``).

        PTU ``ptu_capacity`` is an absolute target and billable, so it
        defaults to ``max_retries=0`` with an idempotency key.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check.
        """
        body = _build_scale_body(
            deployed_model,
            capacity,
            ptu_capacity,
            instance_id,
            pre_paid_info,
            order_type,
        )
        headers = extra_headers
        if ptu_capacity is not None:
            max_retries = _purchase_retries(max_retries)
            headers = _idempotency_headers(extra_headers)
        resp = await self._client.transport.request_api(
            "PUT",
            f"{_BASE_PATH}/{deployed_model}/scale",
            json=body,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                headers,
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

    @overload
    async def renew(
        self,
        deployed_model: str,
        *,
        pre_paid_info: Any,
        instance_id: Optional[str] = None,
        is_change: Optional[bool] = None,
        ptu_capacity: Any = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    async def renew(
        self,
        deployed_model: str,
        *,
        pre_paid_info: Any,
        instance_id: Optional[str] = None,
        is_change: Optional[bool] = None,
        ptu_capacity: Any = None,
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

    async def renew(
        self,
        deployed_model: str,
        *,
        pre_paid_info: Any,
        instance_id: Optional[str] = None,
        is_change: Optional[bool] = None,
        ptu_capacity: Any = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Renew a pre-paid PTU ModelCode (async twin of ``Deployments.renew``).

        Billable, so defaults to ``max_retries=0`` with an idempotency key.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check.
        """
        body = _renew_body(pre_paid_info, is_change, ptu_capacity, instance_id)
        resp = await self._client.transport.request_api(
            "PUT",
            f"{_BASE_PATH}/{deployed_model}/renew",
            json=body,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                _idempotency_headers(extra_headers),
                extra_query,
                extra_body,
                max_retries=_purchase_retries(max_retries),
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

    @overload
    async def update_overflow_strategy(
        self,
        deployed_model: str,
        *,
        overflow_strategy: str,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Deployment:
        ...

    @overload
    async def update_overflow_strategy(
        self,
        deployed_model: str,
        *,
        overflow_strategy: str,
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

    async def update_overflow_strategy(
        self,
        deployed_model: str,
        *,
        overflow_strategy: str,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Deployment, "APIResponse"]:
        """Set the whole-ModelCode overflow strategy (async twin).

        ``enable`` overflows to pay-as-you-go (extra cost); ``disable``
        throttles. Config toggle, so normal retries apply.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check.
        """
        if overflow_strategy not in ("enable", "disable"):
            raise ValueError("overflow_strategy must be 'enable' or 'disable'")
        resp = await self._client.transport.request_api(
            "PUT",
            f"{_BASE_PATH}/{deployed_model}/update-overflowstrategy",
            json={"overflow_strategy": overflow_strategy},
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
        return with_headers(
            _parse(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    async def delete(
        self,
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> JobStatus:
        ...

    @overload
    async def delete(
        self,
        deployed_model: str,
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

    async def delete(
        self,
        deployed_model: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[JobStatus, "APIResponse"]:
        """Delete a deployment; the job status reports the teardown.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        resp = await self._client.transport.request_api(
            "DELETE",
            f"{_BASE_PATH}/{deployed_model}",
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
        return with_headers(
            _parse_status(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    async def stream_events(
        self,
        deployed_model_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> AsyncIterator[DeploymentEvent]:
        response = await self._client.transport.open_stream(
            "GET",
            f"{_BASE_PATH}/{deployed_model_id}/stream",
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
        )
        try:
            async for is_error, status_code, data in _aiter_sse(response):
                if is_error:
                    if self._client.raise_on_error:
                        raise _stream_error(
                            status_code,
                            data,
                            response.headers,
                        )
                    _log_stream_error(data, response.headers)
                    yield _parse_event(
                        data,
                        is_error=True,
                        strict=self._client.config.strict_validation,
                        response=response,
                    )
                    return
                yield _parse_event(
                    data,
                    strict=self._client.config.strict_validation,
                    response=response,
                )
        finally:
            await response.aclose()
