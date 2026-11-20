# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Atomic read-op registry for the management surface (G1, read-only).

Only the 14 read ops needed by G2 (workspaces / api_keys / billing) and
G5 (token_plan) are coded here. The other 24 ops of ModelStudio/2026-02-10
-- 21 writes plus 3 out-of-scope reads -- are deliberately absent and
catalogued as high-risk / deferred in the milestone doc; no write op is
implemented in this revision.

Each :class:`OpSpec` records the wire contract verified against the live
OpenAPI metadata: HTTP method, ROA path template, and every parameter's
snake_case name, wire name, serialization kind and location. Array
parameters carry the metadata's ``style`` -- ``json`` (a JSON-stringified
value) for providers/capabilities/features/groupBy and the billing
objects, ``flat`` (``Client.query`` ``.N`` expansion) for StatusList and
the model-permission ``filter`` object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

__all__ = ["READ_OPS", "OpSpec", "Param", "build_op_request"]

SCALAR = "scalar"
BOOL = "bool"
JSON = "json"
FLAT = "flat"
QUERY = "query"
PATH = "path"


@dataclass(frozen=True)
class Param:
    """One operation parameter's snake_case -> wire mapping."""

    snake: str
    wire: str
    kind: str = SCALAR
    location: str = QUERY
    required: bool = False


@dataclass(frozen=True)
class OpSpec:
    """A read op's wire contract and response payload location."""

    action: str
    method: str
    path: str
    params: Tuple[Param, ...]
    # Response key holding the payload; ``None`` means the whole body.
    data_field: Optional[str]
    # "token" (nextToken), "page" (PageNo/PageNum) or "none".
    pagination: str = "none"


def _p(
    snake: str,
    wire: str,
    kind: str = SCALAR,
    location: str = QUERY,
    required: bool = False,
) -> Param:
    return Param(snake, wire, kind, location, required)


_NEXT_TOKEN = _p("next_token", "nextToken")
_MAX_RESULTS = _p("max_results", "maxResults")
_WORKSPACE_ID = _p("workspace_id", "workspaceId")


READ_OPS: Dict[str, OpSpec] = {
    "ListWorkspaces": OpSpec(
        "ListWorkspaces",
        "GET",
        "/modelstudio/workspaces",
        (
            _NEXT_TOKEN,
            _MAX_RESULTS,
            _p("workspace_name", "workspaceName"),
            _WORKSPACE_ID,
        ),
        "workspaces",
        "token",
    ),
    "ListModels": OpSpec(
        "ListModels",
        "GET",
        "/modelstudio/models",
        (
            _p("providers", "providers", JSON),
            _p("capabilities", "capabilities", JSON),
            _p("features", "features", JSON),
            _p("language", "language"),
            _p("context_window", "contextWindow"),
            _p("name", "name"),
            _p("model", "model"),
            _NEXT_TOKEN,
            _MAX_RESULTS,
        ),
        "models",
        "token",
    ),
    "ListModelLimits": OpSpec(
        "ListModelLimits",
        "GET",
        "/modelstudio/models/limits",
        (
            _NEXT_TOKEN,
            _MAX_RESULTS,
            _WORKSPACE_ID,
            _p("name", "name"),
            _p("model", "model"),
        ),
        "limits",
        "token",
    ),
    "ListModelPermissions": OpSpec(
        "ListModelPermissions",
        "GET",
        "/modelstudio/models/permissions",
        (
            _p("workspace_id", "workspaceId", SCALAR, QUERY, True),
            _MAX_RESULTS,
            _NEXT_TOKEN,
            _p("authorization_scope", "authorizationScope"),
            _p("model_action", "modelAction"),
            _p("filter_", "filter", FLAT),
        ),
        "list",
        "token",
    ),
    "ListApiKeys": OpSpec(
        "ListApiKeys",
        "GET",
        "/modelstudio/apikeys",
        (
            _NEXT_TOKEN,
            _MAX_RESULTS,
            _WORKSPACE_ID,
            _p("api_key_id", "apiKeyId"),
            _p("description", "description"),
            _p("order_by", "orderBy"),
            _p("order", "order"),
        ),
        "apiKeys",
        "token",
    ),
    "GetApiKey": OpSpec(
        "GetApiKey",
        "GET",
        "/modelstudio/apikeys/{apiKeyId}",
        (_p("api_key_id", "apiKeyId", SCALAR, PATH, True),),
        "apiKey",
    ),
    "GetBillingOverview": OpSpec(
        "GetBillingOverview",
        "GET",
        "/modelstudio/billing/overview",
        (
            _p("bill_month", "billMonth"),
            _p("group_by", "groupBy", JSON),
            _p("filter_", "filter", JSON),
            _p("top_num", "topNum"),
            _p("zero_filter", "zeroFilter", BOOL),
            _p("region_id", "regionId"),
            _p("locale", "locale"),
            _p("granularity", "granularity"),
            _p("time_period", "timePeriod", JSON),
        ),
        "data",
    ),
    "GetBillingTrend": OpSpec(
        "GetBillingTrend",
        "GET",
        "/modelstudio/billing/trend",
        (
            _p("granularity", "granularity"),
            _p("time_period", "timePeriod", JSON),
            _p("group_by", "groupBy", JSON),
            _p("filter_", "filter", JSON),
            _p("top_num", "topNum"),
            _p("zero_filter", "zeroFilter", BOOL),
            _p("include_history_period", "includeHistoryPeriod", BOOL),
            _p("region_id", "regionId"),
            _p("locale", "locale"),
        ),
        "data",
    ),
    "GetTokenPlanAccountDetail": OpSpec(
        "GetTokenPlanAccountDetail",
        "GET",
        "/tokenplan/account",
        (),
        "Data",
    ),
    "ListOrganizationMembers": OpSpec(
        "ListOrganizationMembers",
        "GET",
        "/tokenplan/organization/members",
        (
            _p("name", "Name"),
            _p("status", "Status"),
            _p("has_seat", "HasSeat", BOOL),
            _p("page_num", "PageNum"),
            _p("page_size", "PageSize"),
        ),
        "Data",
        "page",
    ),
    "GetSubscriptionSeatDetails": OpSpec(
        "GetSubscriptionSeatDetails",
        "GET",
        "/tokenplan/subscription/seat-detail",
        (
            _p("page_no", "PageNo"),
            _p("page_size", "PageSize"),
            _p("status_list", "StatusList", FLAT),
            _p("seat_id", "SeatId"),
            _p("seat_type", "SeatType"),
            _p("query_assigned", "QueryAssigned", BOOL),
        ),
        "Data",
        "page",
    ),
    "GetOrganizationMemberSeatStats": OpSpec(
        "GetOrganizationMemberSeatStats",
        "GET",
        "/tokenplan/organization/member-seat-stats",
        (),
        None,
    ),
    "GetSubscriptionStats": OpSpec(
        "GetSubscriptionStats",
        "GET",
        "/tokenplan/subscription/stats",
        (),
        "Data",
    ),
    "ListSubscriptionSharedPackages": OpSpec(
        "ListSubscriptionSharedPackages",
        "GET",
        "/tokenplan/subscription/shared-packages",
        (
            _p("page_no", "PageNo"),
            _p("page_size", "PageSize"),
            _p("status_list", "StatusList", FLAT),
        ),
        "Data",
        "page",
    ),
}


def build_op_request(
    action: str,
    **kwargs: Any,
) -> Tuple[str, str, Dict[str, Any]]:
    """Map snake_case call kwargs to a wire ``(method, path, query)``.

    Path parameters are substituted into the ROA template (the signer
    encodes the whole path once, so values go in raw). Query parameters
    are split out and serialized per their metadata style. ``None``
    values are omitted; a missing required parameter raises ``ValueError``
    and an unknown keyword raises ``TypeError`` so a typo fails loudly
    rather than silently dropping a filter.
    """
    spec = READ_OPS[action]
    known = {param.snake for param in spec.params}
    unknown = set(kwargs) - known
    if unknown:
        raise TypeError(
            f"unexpected parameter(s) for {action}: {sorted(unknown)}",
        )
    path = spec.path
    query: Dict[str, Any] = {}
    for param in spec.params:
        value = kwargs.get(param.snake)
        if value is None:
            if param.required:
                raise ValueError(f"{action} requires '{param.snake}'")
            continue
        if param.location == PATH:
            path = path.replace("{" + param.wire + "}", str(value))
        else:
            query[param.wire] = _serialize(value, param.kind)
    return spec.method, path, query


def _serialize(value: Any, kind: str) -> Any:
    if kind == BOOL:
        return "true" if value else "false"
    if kind == JSON:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value
