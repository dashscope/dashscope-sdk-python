# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""PTU throughput-reservation capacity subresources (milestone D3).

The native gateway exposes reserved throughput (``plan="ptu"``) under the
same ``/deployments`` base path as MU/lora deployments, plus two
subresources scoped to one ``deployed_model`` (ModelCode):

* ``capacity-instances`` -- the billable capacity instances behind a
  ModelCode. Write operations (create/scale/renew/delete) are asynchronous
  and answer with a :class:`CapacityOperation`, not the instance itself.
* ``capacity-operations`` -- poll one write operation until it reaches a
  terminal status (``SUCCEEDED`` / ``FAILED``).

Billing safeguards (authoritative throughput-reservation contract):

* Purchase/renew/scale/delete default to ``max_retries=0``: the wire has no
  automatic dedup for a retried write, so a blind retry could spawn a
  duplicate billable instance. Callers opt back in explicitly.
* Instance writes carry an idempotency key (``x-acs-req-uuid`` mirrored to
  ``X-DashScope-RequestId``). Reusing the key with identical params reuses
  the operation; a new business operation must use a new key.
* HTTP 200 is not operation success: ``output.operation_status`` may be
  ``FAILED`` with ``error_code`` / ``error_message``. ``_check_operation``
  surfaces that as a catalog error when ``raise_on_error`` is set.

The first ModelCode creation (``deployments.create(plan="ptu")``) is NOT
given an idempotency key -- the contract reserves the key for instance
operations and warns against deduping the initial ModelCode purchase.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    overload,
)

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from dashscope.errors import DashScopeError, error_from_business_code
from dashscope.resources._jobs import (
    DEFAULT_MAX_INTERVAL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
    FAILED,
    SUCCEEDED,
)
from dashscope.transport._options import request_options
from dashscope.types._parse import parse_model
from dashscope.types.common import (
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

PROCESSING = "PROCESSING"
_OP_TERMINAL = frozenset({SUCCEEDED, FAILED})

__all__ = [
    "AsyncCapacityInstances",
    "AsyncCapacityOperations",
    "CapacityInstance",
    "CapacityInstanceList",
    "CapacityInstances",
    "CapacityOperation",
    "CapacityOperations",
    "PrePaidInfo",
    "PtuCapacity",
]


class PtuCapacity(BaseModel):
    """Reserved throughput in kTPM (thousands of tokens per minute).

    ``input_tpm`` / ``output_tpm`` are required on a write; the optional
    ``thinking_output_tpm`` reserves reasoning-output throughput separately.
    Extra wire counters pass through untouched.
    """

    model_config = ConfigDict(extra="allow")

    input_tpm: Optional[int] = None
    output_tpm: Optional[int] = None
    thinking_output_tpm: Optional[int] = None


class PrePaidInfo(BaseModel):
    """Pre-paid (subscription) billing terms for a PTU instance.

    ``pricing_cycle`` is case-sensitive: ``Day`` (default) or ``Hour``
    (8-hour trial, ``ptu_default`` only). ``duration`` is in cycles and must
    be positive; an ``Hour`` purchase is fixed at 8 and cannot auto-renew.
    """

    model_config = ConfigDict(extra="allow")

    pricing_cycle: Optional[str] = None
    duration: Optional[int] = None
    auto_renewal: Optional[bool] = None
    auto_renewal_duration: Optional[int] = None
    auto_renewal_cycle: Optional[int] = None


class CapacityInstance(ResponseBase):
    """One billable capacity instance behind a ModelCode.

    ``model_service_id`` is the ModelCode (the ``deployed_model`` the
    instance belongs to); ``instance_id`` identifies this instance. The
    ``can_*`` flags report which operations the platform currently allows
    (an 8-hour trial instance, for example, cannot scale/renew/delete).
    A released instance is ``deleted=True`` -- ``STOPPED`` alone is not a
    release.
    """

    model_service_id: Optional[str] = None
    instance_id: Optional[str] = None
    charge_type: Optional[str] = None
    status: Optional[str] = None
    deleted: bool = False
    effective_capacity: Optional[PtuCapacity] = None
    configured_capacity: Optional[PtuCapacity] = None
    target_capacity: Optional[PtuCapacity] = None
    pre_paid_info: Optional[PrePaidInfo] = None
    gmt_effective: Optional[str] = None
    gmt_expired: Optional[str] = None
    can_scale: Optional[bool] = None
    can_renew: Optional[bool] = None
    can_enable_auto_renew: Optional[bool] = None
    can_disable_auto_renew: Optional[bool] = None
    can_delete: Optional[bool] = None
    fail_reason: Optional[str] = None
    gmt_created: Optional[str] = None
    gmt_modified: Optional[str] = None
    gmt_deleted: Optional[str] = None


class CapacityInstanceList(ResponseBase):
    """A page of capacity instances.

    This subresource paginates in camelCase, unlike the ModelCode-level
    ``DeploymentList``: the wire sends ``records`` / ``items`` (total) /
    ``page`` / ``itemsPerPage`` / ``pageCount``. The fields normalize to
    snake_case while keeping the wire aliases readable.
    """

    records: List[CapacityInstance] = []
    total_count: int = Field(
        0,
        validation_alias=AliasChoices("total_count", "items"),
    )
    page: int = Field(1, validation_alias=AliasChoices("page", "page_no"))
    items_per_page: int = Field(
        20,
        validation_alias=AliasChoices(
            "items_per_page",
            "itemsPerPage",
            "page_size",
        ),
    )
    page_count: int = Field(
        0,
        validation_alias=AliasChoices("page_count", "pageCount"),
    )


class CapacityOperation(ResponseBase):
    """The asynchronous result of one capacity write operation.

    ``operation_request_id`` is the request id of the ORIGINAL write call
    that spawned this operation, preserved separately from the envelope
    ``request_id`` (which is the id of this poll response). The two differ;
    ``flatten_envelope`` would otherwise let the poll id overwrite the
    write id, so ``_parse_operation`` captures it before flattening.

    ``operation_type`` is CREATE / SCALE / RENEW / DELETE for public writes
    (STOP / REFUND are platform lifecycle events with no public write API).
    """

    operation_id: Optional[str] = None
    operation_request_id: Optional[str] = None
    operation_type: Optional[str] = None
    operation_status: Optional[str] = None
    model_service_id: Optional[str] = None
    instance_id: Optional[str] = None
    from_status: Optional[str] = None
    current_status: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    gmt_created: Optional[str] = None
    gmt_finished: Optional[str] = None

    @property
    def finished(self) -> bool:
        return self.operation_status in _OP_TERMINAL

    @property
    def succeeded(self) -> bool:
        return self.operation_status == SUCCEEDED

    @property
    def failed(self) -> bool:
        return self.operation_status == FAILED


def _idempotency_headers(
    extra_headers: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    """Mirror one idempotency key across both header spellings.

    Read priority is ``x-acs-req-uuid`` > ``X-DashScope-RequestId``, matched
    case-insensitively (header names are case-insensitive on the wire); when
    neither is supplied a fresh UUID is generated. Both headers are set to
    the SAME value so the gateway dedups a network retry of this write.
    A caller-supplied key is preserved, letting them reuse an operation
    deliberately or rotate to a new business operation. Variant casings of
    either name are normalized to the canonical pair so two conflicting
    spellings never reach the wire together.
    """
    headers = dict(extra_headers or {})
    lowered = {name.lower(): value for name, value in headers.items()}
    key = lowered.get("x-acs-req-uuid") or lowered.get("x-dashscope-requestid")
    if not key:
        key = uuid.uuid4().hex
    for name in list(headers):
        if name.lower() in ("x-acs-req-uuid", "x-dashscope-requestid"):
            del headers[name]
    headers["x-acs-req-uuid"] = key
    headers["X-DashScope-RequestId"] = key
    return headers


def _purchase_retries(max_retries: Optional[int]) -> int:
    """Default a billable write to no automatic retry.

    The wire has no dedup for a blindly retried purchase/renew/scale, so a
    transport retry could create a duplicate billable instance. ``None``
    (the caller said nothing) becomes 0; an explicit value is honored.
    """
    return 0 if max_retries is None else max_retries


def _ptu_capacity_body(capacity: Any) -> Dict[str, Any]:
    """Normalize and validate a ``ptu_capacity`` write payload."""
    if isinstance(capacity, PtuCapacity):
        capacity = capacity.model_dump(exclude_none=True)
    if not isinstance(capacity, Mapping):
        raise ValueError("ptu_capacity must be a mapping of TPM counters")
    body = dict(capacity)
    for key in ("input_tpm", "output_tpm"):
        if key not in body:
            raise ValueError(f"ptu_capacity requires '{key}'")
        value = body[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"ptu_capacity '{key}' must be a non-negative int",
            )
    return body


def _pre_paid_info_body(info: Any) -> Dict[str, Any]:
    """Normalize and validate a ``pre_paid_info`` write payload."""
    if isinstance(info, PrePaidInfo):
        info = info.model_dump(exclude_none=True)
    if not isinstance(info, Mapping):
        raise ValueError("pre_paid_info must be a mapping of billing terms")
    body = dict(info)
    duration = body.get("duration")
    if duration is not None:
        if not isinstance(duration, int) or isinstance(duration, bool):
            raise ValueError("pre_paid_info 'duration' must be an int")
        if duration <= 0:
            raise ValueError("pre_paid_info 'duration' must be positive")
    cycle = body.get("pricing_cycle")
    if cycle is not None and cycle not in ("Day", "Hour"):
        raise ValueError(
            "pre_paid_info 'pricing_cycle' must be 'Day' or 'Hour'",
        )
    return body


def _validate_order_type(order_type: Optional[str]) -> None:
    if order_type is not None and order_type not in ("UPGRADE", "DOWNGRADE"):
        raise ValueError("order_type must be 'UPGRADE' or 'DOWNGRADE'")


def _instance_create_body(
    billing_method: str,
    ptu_capacity: Any,
    pre_paid_info: Any,
) -> Dict[str, Any]:
    if billing_method not in ("PRE_PAY", "POST_PAY"):
        raise ValueError("billing_method must be 'PRE_PAY' or 'POST_PAY'")
    body: Dict[str, Any] = {
        "billing_method": billing_method,
        "ptu_capacity": _ptu_capacity_body(ptu_capacity),
    }
    if pre_paid_info is not None:
        body["pre_paid_info"] = _pre_paid_info_body(pre_paid_info)
    elif billing_method == "PRE_PAY":
        raise ValueError("pre_paid_info is required for PRE_PAY billing")
    return body


def _instance_scale_body(
    ptu_capacity: Any,
    order_type: Optional[str],
) -> Dict[str, Any]:
    _validate_order_type(order_type)
    body: Dict[str, Any] = {"ptu_capacity": _ptu_capacity_body(ptu_capacity)}
    if order_type is not None:
        body["order_type"] = order_type
    return body


def _renew_body(
    pre_paid_info: Any,
    is_change: Optional[bool],
    ptu_capacity: Any,
    instance_id: Optional[str] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "pre_paid_info": _pre_paid_info_body(pre_paid_info),
    }
    if instance_id is not None:
        body["instance_id"] = instance_id
    if is_change is not None:
        body["is_change"] = is_change
    if ptu_capacity is not None:
        if not is_change:
            raise ValueError(
                "changing ptu_capacity on renew requires is_change=True",
            )
        body["ptu_capacity"] = _ptu_capacity_body(ptu_capacity)
    return body


def _parse_operation(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> CapacityOperation:
    output = body.get("output")
    output = output if isinstance(output, dict) else {}
    flat = flatten_envelope(body)
    # The inner request_id is the original write's id; capture it before
    # the envelope's poll id wins the flattened ``request_id`` field.
    original = output.get("request_id")
    if original is not None:
        flat["operation_request_id"] = original
    return parse_model(
        CapacityOperation,
        flat,
        strict=strict,
        response=response,
    )


def _parse_instance(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> CapacityInstance:
    return parse_model(
        CapacityInstance,
        flatten_envelope(body),
        strict=strict,
        response=response,
    )


def _parse_instance_list(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> CapacityInstanceList:
    data = flatten_envelope(body)
    records = data.get("records")
    if not isinstance(records, list):
        records = []
    # Honor every alias the model declares: a snake_case answer must not
    # fall back to len(records)/20 and truncate pagination early.
    return parse_model(
        CapacityInstanceList,
        {
            "records": records,
            "total_count": data.get(
                "items",
                data.get("total_count", len(records)),
            ),
            "page": data.get("page", data.get("page_no", 1)),
            "items_per_page": data.get(
                "itemsPerPage",
                data.get("items_per_page", data.get("page_size", 20)),
            ),
            "page_count": data.get(
                "pageCount",
                data.get("page_count", 0),
            ),
            **response_envelope(data),
        },
        strict=strict,
        response=response,
    )


def _check_operation(
    client: Any,
    operation: CapacityOperation,
) -> CapacityOperation:
    """Surface an HTTP-200 operation failure as a catalog error.

    A capacity write can return HTTP 200 with ``operation_status=FAILED``;
    the dual-layer HTTP/business check in ``raise_for_status`` does not see
    it because the failure lives inside ``output``. In raising mode this
    converts it to the matching catalog error; in compat mode the operation
    is returned untouched for the caller to inspect.
    """
    if operation.operation_status == FAILED and client.raise_on_error:
        raise error_from_business_code(
            message=(
                operation.error_message
                or f"capacity operation {operation.operation_id} failed"
            ),
            code=operation.error_code,
            body=operation.model_dump(exclude={"headers"}),
            headers=operation.headers,
        )
    return operation


class CapacityInstances:
    """Capacity-instance subresource for one ModelCode (sync).

    Mounted as ``client.deployments.capacity_instances``. Every method takes
    the ``deployed_model`` (ModelCode) first; instance writes are scoped to
    it and return a :class:`CapacityOperation` to poll.
    """

    def __init__(self, client) -> None:
        self._client = client

    def _path(self, deployed_model: str) -> str:
        return f"{_BASE_PATH}/{deployed_model}/capacity-instances"

    @overload
    def create(
        self,
        deployed_model: str,
        *,
        billing_method: str,
        ptu_capacity: Any,
        pre_paid_info: Any = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CapacityOperation:
        ...

    @overload
    def create(
        self,
        deployed_model: str,
        *,
        billing_method: str,
        ptu_capacity: Any,
        pre_paid_info: Any = None,
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
        deployed_model: str,
        *,
        billing_method: str,
        ptu_capacity: Any,
        pre_paid_info: Any = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CapacityOperation, "APIResponse"]:
        """Buy an additional capacity instance (stacked purchase).

        ``billing_method`` is case-sensitive ``PRE_PAY`` / ``POST_PAY``; a
        ModelCode may hold only one undeleted post-paid instance. Defaults
        to ``max_retries=0`` and carries an idempotency key. Returns the
        :class:`CapacityOperation` to poll (HTTP 200 is not success).
        """
        body = _instance_create_body(
            billing_method,
            ptu_capacity,
            pre_paid_info,
        )
        resp = self._client.transport.request_api(
            "POST",
            self._path(deployed_model),
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
        operation = with_headers(
            _parse_operation(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )
        return _check_operation(self._client, operation)

    @overload
    def list(
        self,
        deployed_model: str,
        *,
        page_no: int = 1,
        page_size: int = 20,
        include_deleted: bool = True,
        statuses: Optional[str] = None,
        charge_types: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CapacityInstanceList:
        ...

    @overload
    def list(
        self,
        deployed_model: str,
        *,
        page_no: int = 1,
        page_size: int = 20,
        include_deleted: bool = True,
        statuses: Optional[str] = None,
        charge_types: Optional[str] = None,
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
        deployed_model: str,
        *,
        page_no: int = 1,
        page_size: int = 20,
        include_deleted: bool = True,
        statuses: Optional[str] = None,
        charge_types: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CapacityInstanceList, "APIResponse"]:
        """List capacity instances of a ModelCode, one page at a time.

        ``statuses`` / ``charge_types`` are comma-separated filters (e.g.
        ``"RUNNING,STOPPED"``, ``"pre_paid,post_paid"``). ``include_deleted``
        defaults to True so released instances stay visible. Pagination is
        camelCase on the wire; see :class:`CapacityInstanceList`.
        """
        params: Dict[str, Any] = {
            "page_no": page_no,
            "page_size": page_size,
            "include_deleted": include_deleted,
        }
        if statuses is not None:
            params["statuses"] = statuses
        if charge_types is not None:
            params["charge_types"] = charge_types
        resp = self._client.transport.request_api(
            "GET",
            self._path(deployed_model),
            params=params,
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
            _parse_instance_list(
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
        instance_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CapacityInstance:
        ...

    @overload
    def get(
        self,
        deployed_model: str,
        instance_id: str,
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
        instance_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CapacityInstance, "APIResponse"]:
        """Fetch one capacity instance (deleted instances stay queryable).

        The ModelCode and instance id must match; a post-paid repurchase
        reuses the same instance id.
        """
        resp = self._client.transport.request_api(
            "GET",
            f"{self._path(deployed_model)}/{instance_id}",
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
            _parse_instance(
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
        instance_id: str,
        *,
        ptu_capacity: Any,
        order_type: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CapacityOperation:
        ...

    @overload
    def scale(
        self,
        deployed_model: str,
        instance_id: str,
        *,
        ptu_capacity: Any,
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
        instance_id: str,
        *,
        ptu_capacity: Any,
        order_type: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CapacityOperation, "APIResponse"]:
        """Resize one instance to an ABSOLUTE ``ptu_capacity``.

        The instance id comes from the path (no body id). Check
        ``can_scale`` first; an 8-hour trial instance cannot scale.
        Defaults to ``max_retries=0`` with an idempotency key.
        """
        body = _instance_scale_body(ptu_capacity, order_type)
        resp = self._client.transport.request_api(
            "PUT",
            f"{self._path(deployed_model)}/{instance_id}/scale",
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
        operation = with_headers(
            _parse_operation(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )
        return _check_operation(self._client, operation)

    @overload
    def renew(
        self,
        deployed_model: str,
        instance_id: str,
        *,
        pre_paid_info: Any,
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
    ) -> CapacityOperation:
        ...

    @overload
    def renew(
        self,
        deployed_model: str,
        instance_id: str,
        *,
        pre_paid_info: Any,
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
        instance_id: str,
        *,
        pre_paid_info: Any,
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
    ) -> Union[CapacityOperation, "APIResponse"]:
        """Renew one pre-paid instance (no ``order_type`` on renew).

        Changing ``ptu_capacity`` requires ``is_change=True``. Check
        ``can_renew`` first; an 8-hour trial instance cannot renew.
        Defaults to ``max_retries=0`` with an idempotency key.
        """
        body = _renew_body(pre_paid_info, is_change, ptu_capacity)
        resp = self._client.transport.request_api(
            "PUT",
            f"{self._path(deployed_model)}/{instance_id}/renew",
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
        operation = with_headers(
            _parse_operation(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )
        return _check_operation(self._client, operation)

    @overload
    def delete(
        self,
        deployed_model: str,
        instance_id: str,
        *,
        reason: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CapacityOperation:
        ...

    @overload
    def delete(
        self,
        deployed_model: str,
        instance_id: str,
        *,
        reason: Optional[str] = None,
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
        instance_id: str,
        *,
        reason: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CapacityOperation, "APIResponse"]:
        """Release one instance (queued asynchronously, no JSON body).

        A post-paid release marks the instance ``deleted=True`` / STOPPED;
        an effective pre-paid instance must be unsubscribed first
        (PREPAID_UNSUBSCRIBE_REQUIRED). Defaults to ``max_retries=0`` with
        an idempotency key. ``reason`` is sent as a query parameter.
        """
        params: Dict[str, Any] = {}
        if reason is not None:
            params["reason"] = reason
        resp = self._client.transport.request_api(
            "DELETE",
            f"{self._path(deployed_model)}/{instance_id}",
            params=params or None,
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
        operation = with_headers(
            _parse_operation(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )
        return _check_operation(self._client, operation)


class CapacityOperations:
    """Capacity-operation polling subresource for one ModelCode (sync).

    Mounted as ``client.deployments.capacity_operations``. ``get`` reads one
    operation; ``wait`` polls with exponential backoff until terminal.
    """

    def __init__(self, client) -> None:
        self._client = client

    def _path(self, deployed_model: str, operation_id: str) -> str:
        return (
            f"{_BASE_PATH}/{deployed_model}/capacity-operations/{operation_id}"
        )

    @overload
    def get(
        self,
        deployed_model: str,
        operation_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CapacityOperation:
        ...

    @overload
    def get(
        self,
        deployed_model: str,
        operation_id: str,
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
        operation_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CapacityOperation, "APIResponse"]:
        """Read one capacity operation by the id a write returned.

        Use the ``operation_id`` from the write response, not a guessed
        number: a nonexistent numeric id answers 404
        CAPACITY_OPERATION_NOT_FOUND. A terminal FAILED operation raises in
        raising mode (compat mode returns it for inspection).
        """
        resp = self._client.transport.request_api(
            "GET",
            self._path(deployed_model, operation_id),
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
        operation = with_headers(
            _parse_operation(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )
        return _check_operation(self._client, operation)

    def wait(
        self,
        deployed_model: str,
        operation_id: str,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        _sleep: Optional[Callable[[float], Any]] = None,
    ) -> CapacityOperation:
        """Poll with exponential backoff until the operation is terminal.

        Mirrors ``jobs.wait``: a terminal FAILED operation raises in raising
        mode (via ``get``); with ``raise_on_error=False`` it comes back as
        the :class:`CapacityOperation` envelope -- check ``operation_status``
        / ``failed``. ``_sleep`` is a test seam replacing the pause between
        polls. After a terminal result, refresh the instance/deployment to
        read the new effective state.
        """
        sleep_fn = _sleep or time.sleep
        deadline = time.monotonic() + timeout
        current = self.get(
            deployed_model,
            operation_id,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
        )
        interval = poll_interval
        while not current.finished:
            if current.is_error:
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for capacity "
                    f"operation {operation_id}",
                )
            sleep_fn(min(interval, remaining))
            interval = min(interval * 2, max_interval)
            current = self.get(
                deployed_model,
                operation_id,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            )
        return current


class AsyncCapacityInstances:
    """Async twin of :class:`CapacityInstances`."""

    def __init__(self, client) -> None:
        self._client = client

    def _path(self, deployed_model: str) -> str:
        return f"{_BASE_PATH}/{deployed_model}/capacity-instances"

    @overload
    async def create(
        self,
        deployed_model: str,
        *,
        billing_method: str,
        ptu_capacity: Any,
        pre_paid_info: Any = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CapacityOperation:
        ...

    @overload
    async def create(
        self,
        deployed_model: str,
        *,
        billing_method: str,
        ptu_capacity: Any,
        pre_paid_info: Any = None,
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
        deployed_model: str,
        *,
        billing_method: str,
        ptu_capacity: Any,
        pre_paid_info: Any = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CapacityOperation, "APIResponse"]:
        """Buy an additional capacity instance (stacked purchase)."""
        body = _instance_create_body(
            billing_method,
            ptu_capacity,
            pre_paid_info,
        )
        resp = await self._client.transport.request_api(
            "POST",
            self._path(deployed_model),
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
        operation = with_headers(
            _parse_operation(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )
        return _check_operation(self._client, operation)

    @overload
    async def list(
        self,
        deployed_model: str,
        *,
        page_no: int = 1,
        page_size: int = 20,
        include_deleted: bool = True,
        statuses: Optional[str] = None,
        charge_types: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CapacityInstanceList:
        ...

    @overload
    async def list(
        self,
        deployed_model: str,
        *,
        page_no: int = 1,
        page_size: int = 20,
        include_deleted: bool = True,
        statuses: Optional[str] = None,
        charge_types: Optional[str] = None,
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
        deployed_model: str,
        *,
        page_no: int = 1,
        page_size: int = 20,
        include_deleted: bool = True,
        statuses: Optional[str] = None,
        charge_types: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CapacityInstanceList, "APIResponse"]:
        """List capacity instances of a ModelCode, one page at a time."""
        params: Dict[str, Any] = {
            "page_no": page_no,
            "page_size": page_size,
            "include_deleted": include_deleted,
        }
        if statuses is not None:
            params["statuses"] = statuses
        if charge_types is not None:
            params["charge_types"] = charge_types
        resp = await self._client.transport.request_api(
            "GET",
            self._path(deployed_model),
            params=params,
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
            _parse_instance_list(
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
        instance_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CapacityInstance:
        ...

    @overload
    async def get(
        self,
        deployed_model: str,
        instance_id: str,
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
        instance_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CapacityInstance, "APIResponse"]:
        """Fetch one capacity instance (deleted instances stay queryable)."""
        resp = await self._client.transport.request_api(
            "GET",
            f"{self._path(deployed_model)}/{instance_id}",
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
            _parse_instance(
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
        instance_id: str,
        *,
        ptu_capacity: Any,
        order_type: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CapacityOperation:
        ...

    @overload
    async def scale(
        self,
        deployed_model: str,
        instance_id: str,
        *,
        ptu_capacity: Any,
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
        instance_id: str,
        *,
        ptu_capacity: Any,
        order_type: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CapacityOperation, "APIResponse"]:
        """Resize one instance to an ABSOLUTE ``ptu_capacity``."""
        body = _instance_scale_body(ptu_capacity, order_type)
        resp = await self._client.transport.request_api(
            "PUT",
            f"{self._path(deployed_model)}/{instance_id}/scale",
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
        operation = with_headers(
            _parse_operation(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )
        return _check_operation(self._client, operation)

    @overload
    async def renew(
        self,
        deployed_model: str,
        instance_id: str,
        *,
        pre_paid_info: Any,
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
    ) -> CapacityOperation:
        ...

    @overload
    async def renew(
        self,
        deployed_model: str,
        instance_id: str,
        *,
        pre_paid_info: Any,
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
        instance_id: str,
        *,
        pre_paid_info: Any,
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
    ) -> Union[CapacityOperation, "APIResponse"]:
        """Renew one pre-paid instance (no ``order_type`` on renew)."""
        body = _renew_body(pre_paid_info, is_change, ptu_capacity)
        resp = await self._client.transport.request_api(
            "PUT",
            f"{self._path(deployed_model)}/{instance_id}/renew",
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
        operation = with_headers(
            _parse_operation(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )
        return _check_operation(self._client, operation)

    @overload
    async def delete(
        self,
        deployed_model: str,
        instance_id: str,
        *,
        reason: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CapacityOperation:
        ...

    @overload
    async def delete(
        self,
        deployed_model: str,
        instance_id: str,
        *,
        reason: Optional[str] = None,
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
        instance_id: str,
        *,
        reason: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CapacityOperation, "APIResponse"]:
        """Release one instance (queued asynchronously, no JSON body)."""
        params: Dict[str, Any] = {}
        if reason is not None:
            params["reason"] = reason
        resp = await self._client.transport.request_api(
            "DELETE",
            f"{self._path(deployed_model)}/{instance_id}",
            params=params or None,
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
        operation = with_headers(
            _parse_operation(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )
        return _check_operation(self._client, operation)


class AsyncCapacityOperations:
    """Async twin of :class:`CapacityOperations`."""

    def __init__(self, client) -> None:
        self._client = client

    def _path(self, deployed_model: str, operation_id: str) -> str:
        return (
            f"{_BASE_PATH}/{deployed_model}/capacity-operations/{operation_id}"
        )

    @overload
    async def get(
        self,
        deployed_model: str,
        operation_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> CapacityOperation:
        ...

    @overload
    async def get(
        self,
        deployed_model: str,
        operation_id: str,
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
        operation_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[CapacityOperation, "APIResponse"]:
        """Read one capacity operation by the id a write returned."""
        resp = await self._client.transport.request_api(
            "GET",
            self._path(deployed_model, operation_id),
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
        operation = with_headers(
            _parse_operation(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )
        return _check_operation(self._client, operation)

    async def wait(
        self,
        deployed_model: str,
        operation_id: str,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        _sleep: Optional[Callable[[float], Any]] = None,
    ) -> CapacityOperation:
        """Poll with exponential backoff until the operation is terminal."""
        sleep_fn = _sleep or asyncio.sleep
        deadline = time.monotonic() + timeout
        current = await self.get(
            deployed_model,
            operation_id,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
        )
        interval = poll_interval
        while not current.finished:
            if current.is_error:
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for capacity "
                    f"operation {operation_id}",
                )
            await sleep_fn(min(interval, remaining))
            interval = min(interval * 2, max_interval)
            current = await self.get(
                deployed_model,
                operation_id,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            )
        return current
