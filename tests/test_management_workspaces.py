# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``client.workspaces`` read-resource tests (sync + async)."""

from __future__ import annotations

import httpx
import pytest

# pylint: disable=wrong-import-position
pytest.importorskip("alibabacloud_credentials")
pytest.importorskip("alibabacloud_openapi_util")

from dashscope.errors import PermissionDeniedError  # noqa: E402
from dashscope.management._pagination import TokenPage  # noqa: E402
from dashscope.management.types import (  # noqa: E402
    ManagementModel,
    ModelLimit,
    ModelPermission,
    Workspace,
)
from dashscope.transport._response import APIResponse  # noqa: E402

from tests.management_helpers import (  # noqa: E402
    action_handler,
    make_async_client,
    make_client,
    ms_body,
    query_of,
)

BODIES = {
    "ListWorkspaces": ms_body(
        workspaces=[{"workspaceId": "ws-1", "workspaceName": "demo"}],
        nextToken="t2",
        maxResults=10,
        totalCount=25,
    ),
    "ListModels": ms_body(
        models=[{"model": "qwen-plus", "provider": "dashscope"}],
        totalCount=1,
    ),
    "ListModelLimits": ms_body(
        limits=[{"model": "qwen-plus", "workspaceId": "ws-1"}],
        totalCount=1,
    ),
    "ListModelPermissions": ms_body(
        list=[{"model": "qwen-plus", "permissions": {"inference": True}}],
        totalCount=1,
    ),
}


def test_workspaces_list_typed_and_paginated():
    client = make_client(action_handler(BODIES))
    page = client.workspaces.list(max_results=10)
    assert isinstance(page, TokenPage)
    assert isinstance(page.items[0], Workspace)
    assert page.items[0].workspace_id == "ws-1"
    assert page.items[0].workspace_name == "demo"
    assert page.next_token == "t2"
    assert page.total_count == 25
    assert page.has_next_page() is True
    assert page.is_error is False
    assert page.request_id == "req-ms"
    client.close()


def test_workspaces_list_wire_query():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=BODIES["ListWorkspaces"])

    client = make_client(handler)
    client.workspaces.list(workspace_name="demo", max_results=10)
    q = query_of(seen[0])
    assert q["workspaceName"] == "demo"
    assert q["maxResults"] == "10"
    assert seen[0].url.path == "/modelstudio/workspaces"
    client.close()


def test_models_list_json_array_query():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=BODIES["ListModels"])

    client = make_client(handler)
    page = client.workspaces.models.list(providers=["dashscope"])
    assert isinstance(page.items[0], ManagementModel)
    assert page.items[0].model == "qwen-plus"
    # json-style array is one JSON-stringified query value.
    assert query_of(seen[0])["providers"] == '["dashscope"]'
    client.close()


def test_model_limits_list_typed():
    client = make_client(action_handler(BODIES))
    page = client.workspaces.model_limits.list(workspace_id="ws-1")
    assert isinstance(page.items[0], ModelLimit)
    assert page.items[0].workspace_id == "ws-1"
    client.close()


def test_model_permissions_requires_workspace_id():
    client = make_client(action_handler(BODIES))
    # Mandatory in the signature ...
    with pytest.raises(TypeError):
        client.workspaces.model_permissions.list()
    # ... and rejected by the op registry when explicitly None.
    with pytest.raises(ValueError):
        client.workspaces.model_permissions.list(workspace_id=None)
    client.close()


def test_model_permissions_flat_filter_query():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=BODIES["ListModelPermissions"])

    client = make_client(handler)
    page = client.workspaces.model_permissions.list(
        workspace_id="ws-1",
        filter_={"model": "qwen-plus"},
    )
    assert isinstance(page.items[0], ModelPermission)
    assert page.items[0].permissions.inference is True
    q = query_of(seen[0])
    assert q["workspaceId"] == "ws-1"
    # flat-style object expands to dotted subkeys via Client.query.
    assert q["filter.model"] == "qwen-plus"
    client.close()


def test_workspaces_raw_overload():
    client = make_client(action_handler(BODIES))
    resp = client.workspaces.list(raw=True)
    assert isinstance(resp, APIResponse)
    assert resp.http_response.status_code == 200
    client.close()


def test_workspaces_error_raises():
    body = {"success": False, "code": "Forbidden", "message": "no"}
    client = make_client(action_handler({"ListWorkspaces": body}))
    with pytest.raises(PermissionDeniedError):
        client.workspaces.list()
    client.close()


def test_workspaces_error_non_raising_folds_envelope():
    body = {"success": False, "code": "Forbidden", "message": "no"}
    client = make_client(
        action_handler({"ListWorkspaces": body}),
        raise_on_error=False,
    )
    page = client.workspaces.list()
    assert page.is_error is True
    assert page.code == "Forbidden"
    assert page.message == "no"
    assert page.items == []
    client.close()


async def test_async_workspaces_and_subresources():
    client = make_async_client(action_handler(BODIES))
    page = await client.workspaces.list()
    assert page.items[0].workspace_id == "ws-1"
    models = await client.workspaces.models.list()
    assert models.items[0].model == "qwen-plus"
    limits = await client.workspaces.model_limits.list()
    assert isinstance(limits.items[0], ModelLimit)
    perms = await client.workspaces.model_permissions.list(workspace_id="ws-1")
    assert isinstance(perms.items[0], ModelPermission)
    await client.aclose()


async def test_async_workspaces_raw_overload():
    client = make_async_client(action_handler(BODIES))
    resp = await client.workspaces.list(raw=True)
    assert isinstance(resp, APIResponse)
    await client.aclose()
