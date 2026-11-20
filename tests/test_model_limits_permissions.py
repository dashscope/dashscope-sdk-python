# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the read-only models.limits / models.permissions sub-resources (A3).

Wire contract verified against the official Bailian docs:
- 查询模型限额  GET /api/v1/models/limits      -> output.quotas[]
- 查询模型授权  GET /api/v1/models/permissions -> output.permissions[]
Both are API-Key Bearer endpoints under the models path, paginated with the
same output.total/page_no/page_size envelope as the catalog. There is no public
model-versions endpoint, so A3 exposes only limits and permissions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.errors import NotFoundError
from dashscope.resources.models import (
    ModelPermission,
    ModelQuota,
    PermissionList,
    QuotaList,
)
from dashscope.types.pagination import BasePage

BASE_URL = "https://api.test.local/api/v1"


def _make_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def _make_async_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return AsyncClient(api_key="sk-test", http_client=http_client, **kwargs)


def _slice_page(
    catalog: List[Dict[str, Any]],
    key: str,
    params: Dict[str, str],
) -> Dict[str, Any]:
    page_no = int(params.get("page_no", 1))
    page_size = int(params.get("page_size", 20))
    start = (page_no - 1) * page_size
    return {
        "code": None,
        "message": None,
        "success": True,
        "request_id": "req-meta",
        "output": {
            key: catalog[start : start + page_size],
            "page_no": page_no,
            "page_size": page_size,
            "total": len(catalog),
        },
    }


def _quota(model: str, **limits: Any) -> Dict[str, Any]:
    return {
        "model": model,
        "workspace_id": "llm-ws1",
        "model_limit": limits or None,
        "workspace_limit": None,
    }


# --- models.limits (sync) -------------------------------------------------


def _limits_handler(catalog, seen):
    def handler(request):
        assert request.url.path == "/api/v1/models/limits"
        seen.append(dict(request.url.params))
        seen[-1]["sdk"] = request.headers.get("x-dashscope-sdk-client")
        return httpx.Response(
            200,
            json=_slice_page(catalog, "quotas", dict(request.url.params)),
        )

    return handler


def test_limits_list_parses_quota_fields():
    catalog = [
        _quota(
            "qwen-plus",
            request_limit=500,
            request_limit_period=1,
            usage_limit=500000,
            usage_limit_field="total_tokens",
            usage_limit_period=6,
        ),
    ]
    seen: List[Any] = []
    client = _make_client(_limits_handler(catalog, seen))

    page = client.models.limits.list(page_no=1, page_size=20)
    assert isinstance(page, QuotaList)
    assert page.total_count == 1
    assert page.request_id == "req-meta"
    quota = page.quotas[0]
    assert isinstance(quota, ModelQuota)
    assert quota.model == "qwen-plus"
    assert quota.workspace_id == "llm-ws1"
    assert quota.model_limit.request_limit == 500
    assert quota.model_limit.request_limit_period == 1
    assert quota.model_limit.usage_limit == 500000
    assert quota.model_limit.usage_limit_field == "total_tokens"
    assert quota.workspace_limit is None
    # Functional-area ident is the models module, path is the limits sub-path.
    assert seen[0]["sdk"].endswith("/models2")


def test_limits_list_sends_name_and_model_filters():
    seen: List[Any] = []
    client = _make_client(_limits_handler([_quota("qwen-plus")], seen))
    client.models.limits.list(name="qwen", model="qwen-plus")
    assert seen[0]["name"] == "qwen"
    assert seen[0]["model"] == "qwen-plus"


def test_limits_iter_pages_crosses_all_pages_and_keeps_filters():
    catalog = [_quota(f"m{i}") for i in range(5)]
    seen: List[Any] = []
    client = _make_client(_limits_handler(catalog, seen))

    page = client.models.limits.list(page_no=1, page_size=2, model="m")
    collected = [q.model for p in page.iter_pages() for q in p]
    assert collected == ["m0", "m1", "m2", "m3", "m4"]
    assert [int(s["page_no"]) for s in seen] == [1, 2, 3]
    # The filter survives traversal: every fetched page carries model.
    assert all(s["model"] == "m" for s in seen)


def test_limits_iter_pages_without_total_walks_full_pages():
    catalog = [_quota(f"m{i}") for i in range(5)]
    seen: List[Any] = []

    def handler(request):
        seen.append(dict(request.url.params))
        body = _slice_page(catalog, "quotas", dict(request.url.params))
        del body["output"]["total"]
        return httpx.Response(200, json=body)

    client = _make_client(handler)
    page = client.models.limits.list(page_no=1, page_size=2)
    collected = [q.model for p in page.iter_pages() for q in p]
    assert collected == ["m0", "m1", "m2", "m3", "m4"]
    assert [int(s["page_no"]) for s in seen] == [1, 2, 3]


def test_limits_null_model_limit_is_preserved():
    catalog = [
        {
            "model": "wan-i2v",
            "workspace_id": "llm-ws1",
            "model_limit": None,
            "workspace_limit": {
                "request_limit": 5,
                "request_limit_period": 1,
                "async_user_queue_limit": 500,
                "async_user_concurrency_limit": 5,
            },
        },
    ]
    client = _make_client(_limits_handler(catalog, []))
    quota = client.models.limits.list().quotas[0]
    assert quota.model_limit is None
    assert quota.workspace_limit.async_user_queue_limit == 500
    assert quota.workspace_limit.async_user_concurrency_limit == 5


def test_limits_get_returns_single_quota_with_envelope():
    catalog = [_quota("qwen-plus", request_limit=2, request_limit_period=60)]

    def handler(request):
        params = dict(request.url.params)
        assert params["model"] == "qwen-plus"
        assert params["page_size"] == "1"
        return httpx.Response(
            200,
            json=_slice_page(catalog, "quotas", params),
        )

    client = _make_client(handler)
    quota = client.models.limits.get("qwen-plus")
    assert isinstance(quota, ModelQuota)
    assert quota.model == "qwen-plus"
    assert quota.request_id == "req-meta"
    assert quota.model_limit.request_limit == 2


def test_limits_get_not_found_raises():
    client = _make_client(_limits_handler([], []))
    with pytest.raises(NotFoundError, match="not found"):
        client.models.limits.get("absent-model")


def test_limits_get_not_found_compat_returns_empty_envelope():
    client = _make_client(_limits_handler([], []), raise_on_error=False)
    quota = client.models.limits.get("absent-model")
    assert quota.model is None
    assert quota.request_id == "req-meta"
    assert quota.is_error is False


def test_limits_raw_returns_response():
    client = _make_client(_limits_handler([_quota("m0")], []))
    resp = client.models.limits.list(raw=True)
    assert resp.http_response.status_code == 200
    assert "quotas" in resp.json()["output"]


def test_limits_model_dump_excludes_callback():
    client = _make_client(_limits_handler([_quota("m0")], []))
    page = client.models.limits.list()
    dumped = page.model_dump()
    assert "_fetch_next" not in dumped
    assert not any(k.startswith("_") for k in dumped)
    assert "quotas" in dumped and "total_count" in dumped


# --- models.permissions (sync) --------------------------------------------


def _permission(model: str, name: str, **flags: Any) -> Dict[str, Any]:
    return {
        "model": model,
        "name": name,
        "permissions": {
            "inference": flags.get("inference", True),
            "fine_tune": flags.get("fine_tune", False),
            "deploy": flags.get("deploy", False),
        },
    }


def _permissions_handler(catalog, seen):
    def handler(request):
        assert request.url.path == "/api/v1/models/permissions"
        params = dict(request.url.params)
        params["sdk"] = request.headers.get("x-dashscope-sdk-client")
        seen.append(params)
        scope = params.get("authorization_scope", "AUTHORIZABLE")
        # AUTHORIZED narrows to models whose inference flag is granted.
        filtered = (
            [p for p in catalog if p["permissions"]["inference"]]
            if scope == "AUTHORIZED"
            else catalog
        )
        return httpx.Response(
            200,
            json=_slice_page(filtered, "permissions", params),
        )

    return handler


def test_permissions_list_parses_flags():
    catalog = [_permission("qwen-plus", "通义千问-Plus", inference=True)]
    seen: List[Any] = []
    client = _make_client(_permissions_handler(catalog, seen))

    page = client.models.permissions.list()
    assert isinstance(page, PermissionList)
    assert page.total_count == 1
    perm = page.permissions[0]
    assert isinstance(perm, ModelPermission)
    assert perm.model == "qwen-plus"
    assert perm.name == "通义千问-Plus"
    assert perm.permissions.inference is True
    assert perm.permissions.fine_tune is False
    assert perm.permissions.deploy is False
    assert seen[0]["sdk"].endswith("/models2")


def test_permissions_scope_and_action_are_sent_and_preserved():
    catalog = [
        _permission("a", "A", inference=True),
        _permission("b", "B", inference=False),
    ]
    seen: List[Any] = []
    client = _make_client(_permissions_handler(catalog, seen))

    page = client.models.permissions.list(
        page_no=1,
        page_size=1,
        authorization_scope="AUTHORIZED",
        action="INFERENCE",
    )
    # Server-side filter: only the authorized model counts toward the total.
    assert page.total_count == 1
    collected = [p.model for pg in page.iter_pages() for p in pg]
    assert collected == ["a"]
    assert all(s["authorization_scope"] == "AUTHORIZED" for s in seen)
    assert all(s["action"] == "INFERENCE" for s in seen)


def test_permissions_get_authorized_scope():
    catalog = [_permission("qwen-plus", "Plus", inference=True)]

    def handler(request):
        params = dict(request.url.params)
        assert params["model"] == "qwen-plus"
        assert params["authorization_scope"] == "AUTHORIZED"
        return httpx.Response(
            200,
            json=_slice_page(catalog, "permissions", params),
        )

    client = _make_client(handler)
    perm = client.models.permissions.get(
        "qwen-plus",
        authorization_scope="AUTHORIZED",
    )
    assert perm.model == "qwen-plus"
    assert perm.permissions.inference is True
    assert perm.request_id == "req-meta"


def test_permissions_get_not_found_raises():
    client = _make_client(_permissions_handler([], []))
    with pytest.raises(NotFoundError, match="not found"):
        client.models.permissions.get("absent-model")


def test_permissions_get_not_found_compat_returns_empty_envelope():
    client = _make_client(
        _permissions_handler([], []),
        raise_on_error=False,
    )
    perm = client.models.permissions.get("absent-model")
    assert perm.model is None
    assert perm.permissions is None
    assert perm.request_id == "req-meta"


def test_permissions_raw_returns_response():
    client = _make_client(_permissions_handler([_permission("a", "A")], []))
    resp = client.models.permissions.list(raw=True)
    assert resp.http_response.status_code == 200
    assert "permissions" in resp.json()["output"]


# --- shared BasePage integration ------------------------------------------


def test_meta_lists_share_base_page_and_total_alias():
    for cls in (QuotaList, PermissionList):
        assert issubclass(cls, BasePage)
        assert cls.model_validate({"total": 7}).total_count == 7
    assert BasePage().items == []


def test_meta_page_without_callback_cannot_advance():
    page = QuotaList(page_no=1, page_size=2, total_count=5)
    assert page.has_next_page() is True
    assert page.next_page() is None


# --- async ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_limits_async_list_and_iter_pages():
    catalog = [_quota(f"m{i}") for i in range(5)]
    seen: List[Any] = []
    client = _make_async_client(_limits_handler(catalog, seen))

    page = await client.models.limits.list(page_no=1, page_size=2)
    collected = [q.model async for p in page.iter_pages_async() for q in p]
    assert collected == ["m0", "m1", "m2", "m3", "m4"]
    assert [int(s["page_no"]) for s in seen] == [1, 2, 3]


@pytest.mark.asyncio
async def test_limits_async_get():
    catalog = [_quota("qwen-plus", request_limit=2)]
    client = _make_async_client(_limits_handler(catalog, []))
    quota = await client.models.limits.get("qwen-plus")
    assert quota.model == "qwen-plus"
    assert quota.request_id == "req-meta"


@pytest.mark.asyncio
async def test_permissions_async_list_and_get():
    catalog = [_permission("a", "A", inference=True)]
    seen: List[Any] = []
    client = _make_async_client(_permissions_handler(catalog, seen))

    page = await client.models.permissions.list(
        authorization_scope="AUTHORIZED",
    )
    assert [p.model async for p in page] == ["a"]
    perm = await client.models.permissions.get("a")
    assert perm.permissions.inference is True


@pytest.mark.asyncio
async def test_async_meta_page_rejects_sync_traversal():
    catalog = [_quota(f"m{i}") for i in range(5)]
    client = _make_async_client(_limits_handler(catalog, []))
    page = await client.models.limits.list(page_no=1, page_size=2)
    with pytest.raises(TypeError, match="iter_pages_async"):
        list(page.iter_pages())


# --- empty-id guards -------------------------------------------------------


@pytest.mark.parametrize("model", ["", "   "])
@pytest.mark.parametrize("sub", ["limits", "permissions"])
def test_meta_get_empty_id_rejected_before_wire(sub, model):
    # An empty model filter matches the whole catalog, so get("") would
    # silently answer with the first entry instead of failing.
    def handler(request):
        raise AssertionError("empty model id must not reach the wire")

    client = _make_client(handler)
    with pytest.raises(ValueError, match="non-empty model id"):
        getattr(client.models, sub).get(model)


@pytest.mark.asyncio
@pytest.mark.parametrize("sub", ["limits", "permissions"])
async def test_async_meta_get_empty_id_rejected_before_wire(sub):
    def handler(request):
        raise AssertionError("empty model id must not reach the wire")

    client = _make_async_client(handler)
    with pytest.raises(ValueError, match="non-empty model id"):
        await getattr(client.models, sub).get("")
