# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Wire-level transport tests for the management surface (sync + async)."""

from __future__ import annotations

import httpx
import pytest

# pylint: disable=wrong-import-position
pytest.importorskip("alibabacloud_credentials")
pytest.importorskip("alibabacloud_openapi_util")

from dashscope._config import DEFAULT_TIMEOUT  # noqa: E402
from dashscope.errors import APIConnectionError  # noqa: E402
from dashscope.management import _transport  # noqa: E402
from dashscope.management import (  # noqa: E402
    AsyncManagementClient,
    ManagementClient,
)

from tests.management_helpers import (  # noqa: E402
    AK,
    SK,
    STS_TOKEN,
    make_async_client,
    make_client,
    ms_body,
    query_of,
)
from tests.test_management_signer import (  # noqa: E402
    _gateway_recompute,
)

WORKSPACES = ms_body(workspaces=[{"workspaceId": "ws-1"}], totalCount=1)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(_transport.time, "sleep", lambda *_: None)
    monkeypatch.setattr(_transport.asyncio, "sleep", _async_noop)


async def _async_noop(*_):
    return None


def _capture_handler(record, body=None, status=200):
    payload = WORKSPACES if body is None else body

    def handler(request):
        record.append(request)
        return httpx.Response(status, json=payload)

    return handler


# --- wire form ------------------------------------------------------------


def test_wire_method_path_action_version():
    seen = []
    client = make_client(_capture_handler(seen))
    client.workspaces.list(max_results=10)
    req = seen[0]
    assert req.method == "GET"
    assert req.url.path == "/modelstudio/workspaces"
    assert req.url.host == "modelstudio.cn-beijing.aliyuncs.com"
    assert req.headers["x-acs-action"] == "ListWorkspaces"
    assert req.headers["x-acs-version"] == "2026-02-10"
    assert query_of(req)["maxResults"] == "10"
    client.close()


def test_wire_authorization_and_sdk_headers():
    seen = []
    client = make_client(_capture_handler(seen))
    client.workspaces.list()
    req = seen[0]
    assert req.headers["authorization"].startswith("ACS3-HMAC-SHA256 ")
    assert req.headers["x-dashscope-sdk-client"].endswith("management2")
    assert req.headers["x-dashscope-sdk-session-id"]
    assert req.headers["x-dashscope-client-request-id"]
    assert (
        "authorization"
        not in req.headers["authorization"].split(
            "SignedHeaders=",
        )[1]
    )
    client.close()


def test_no_bearer_authorization():
    seen = []
    client = make_client(_capture_handler(seen))
    client.workspaces.list()
    assert "Bearer" not in seen[0].headers["authorization"]
    client.close()


def test_special_char_next_token_survives_httpx():
    seen = []
    client = make_client(_capture_handler(seen))
    token = "abc=+/def~ghi"
    client.workspaces.list(next_token=token)
    # httpx must not re-encode the canonical query the signer produced.
    assert "nextToken=abc%3D%2B%2Fdef~ghi" in str(seen[0].url)
    client.close()


def test_sts_token_on_wire():
    seen = []
    client = make_client(
        _capture_handler(seen),
        security_token=STS_TOKEN,
    )
    client.workspaces.list()
    assert seen[0].headers["x-acs-security-token"] == STS_TOKEN
    assert seen[0].headers["x-acs-accesskey-id"] == AK
    client.close()


def test_sdk_headers_kill_switch(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_DISABLE_SDK_HEADERS", "1")
    seen = []
    client = make_client(_capture_handler(seen))
    client.workspaces.list()
    assert "x-dashscope-sdk-client" not in seen[0].headers
    assert "x-dashscope-client-request-id" not in seen[0].headers
    # Still signed and sent correctly without the identity headers.
    assert seen[0].headers["authorization"].startswith("ACS3-HMAC-SHA256 ")
    client.close()


# --- endpoint / region ----------------------------------------------------


def test_default_region_endpoint():
    client = ManagementClient(access_key_id=AK, access_key_secret=SK)
    assert client.transport.host == "modelstudio.cn-beijing.aliyuncs.com"
    client.close()


def test_custom_endpoint_override():
    seen = []
    client = make_client(
        _capture_handler(seen),
        endpoint="modelstudio-vpc.cn-beijing.aliyuncs.com",
    )
    client.workspaces.list()
    assert seen[0].url.host == "modelstudio-vpc.cn-beijing.aliyuncs.com"
    client.close()


def test_with_options_region_changes_endpoint():
    client = ManagementClient(access_key_id=AK, access_key_secret=SK)
    other = client.with_options(region="ap-southeast-1")
    assert other.transport.host == "modelstudio.ap-southeast-1.aliyuncs.com"
    assert other.region == "ap-southeast-1"
    # Independent connection pool, shared credential resolver.
    assert other.transport.http_client is not client.transport.http_client
    assert other._resolver is client._resolver
    client.close()
    other.close()


def test_invalid_region_rejected():
    with pytest.raises(ValueError):
        ManagementClient(
            access_key_id=AK,
            access_key_secret=SK,
            region="bad_region!",
        )


def test_unknown_safe_region_warns(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        client = ManagementClient(
            access_key_id=AK,
            access_key_secret=SK,
            region="cn-shanghai",
        )
    assert "cn-shanghai" in caplog.text
    client.close()


# --- retry ----------------------------------------------------------------


def test_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"code": "Throttling"})
        return httpx.Response(200, json=WORKSPACES)

    client = make_client(handler, max_retries=2)
    page = client.workspaces.list()
    assert [w.workspace_id for w in page] == ["ws-1"]
    assert calls["n"] == 3
    client.close()


def test_no_retry_when_max_retries_zero():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, json={"code": "ServiceUnavailable"})

    client = make_client(handler, max_retries=0)
    with pytest.raises(Exception):
        client.workspaces.list()
    assert calls["n"] == 1
    client.close()


def test_retry_resigns_but_reuses_client_request_id():
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) < 2:
            return httpx.Response(500, json={})
        return httpx.Response(200, json=WORKSPACES)

    client = make_client(handler, max_retries=2)
    client.workspaces.list()
    assert len(seen) == 2
    nonce1 = seen[0].headers["x-acs-signature-nonce"]
    nonce2 = seen[1].headers["x-acs-signature-nonce"]
    assert nonce1 != nonce2
    assert seen[0].headers["authorization"] != seen[1].headers["authorization"]
    crid1 = seen[0].headers["x-dashscope-client-request-id"]
    crid2 = seen[1].headers["x-dashscope-client-request-id"]
    assert crid1 == crid2
    client.close()


def test_connection_error_maps_to_dashscope_error():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    client = make_client(handler, max_retries=0)
    with pytest.raises(APIConnectionError):
        client.workspaces.list()
    client.close()


def test_write_method_not_retried_on_5xx():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, json={"code": "InternalError"})

    client = make_client(handler, max_retries=2)
    resp = client.transport.request(
        "POST",
        "/modelstudio/workspaces",
        action="CreateWorkspace",
        body={"name": "prod"},
    )
    assert resp.status_code == 500
    assert calls["n"] == 1
    client.close()


def test_write_method_not_retried_on_transport_error():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        raise httpx.ConnectError("boom", request=request)

    client = make_client(handler, max_retries=2)
    with pytest.raises(APIConnectionError):
        client.transport.request(
            "DELETE",
            "/modelstudio/apikeys/1",
            action="DeleteApiKey",
        )
    assert calls["n"] == 1
    client.close()


# --- timeout ----------------------------------------------------------------


def test_per_call_timeout_overrides_client_timeout():
    seen = []
    client = make_client(_capture_handler(seen), timeout=30.0)
    client.workspaces.list(timeout=1.5)
    assert seen[0].extensions["timeout"]["read"] == 1.5
    client.close()


def test_client_timeout_applies_without_per_call_override():
    seen = []
    client = make_client(_capture_handler(seen), timeout=30.0)
    client.workspaces.list()
    timeout = seen[0].extensions["timeout"]
    assert timeout["read"] == 30.0
    assert timeout["connect"] == 5.0
    client.close()


def test_default_timeout_without_any_override(monkeypatch):
    # The helper injects a client (whose own httpx default would apply),
    # so force the SDK-owned path: keep the constructor kwargs but swap
    # in a mock transport.
    seen = []
    real_client = httpx.Client

    def mock_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(_capture_handler(seen))
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", mock_client)
    client = ManagementClient(access_key_id=AK, access_key_secret=SK)
    client.workspaces.list()
    timeout = seen[0].extensions["timeout"]
    assert timeout["read"] == DEFAULT_TIMEOUT
    assert timeout["connect"] == 5.0
    client.close()


def test_injected_client_timeout_preserved():
    seen = []
    http_client = httpx.Client(
        transport=httpx.MockTransport(_capture_handler(seen)),
        timeout=99.0,
    )
    client = ManagementClient(
        http_client=http_client,
        access_key_id=AK,
        access_key_secret=SK,
    )
    client.workspaces.list()
    assert seen[0].extensions["timeout"]["read"] == 99.0
    # A per-call override still wins over the injected client default.
    client.workspaces.list(timeout=2.0)
    assert seen[1].extensions["timeout"]["read"] == 2.0
    client.close()
    http_client.close()


# --- async ----------------------------------------------------------------


async def test_async_wire_and_retry():
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) < 2:
            return httpx.Response(429, json={})
        return httpx.Response(200, json=WORKSPACES)

    client = make_async_client(handler, max_retries=2)
    page = await client.workspaces.list()
    assert [w.workspace_id for w in page] == ["ws-1"]
    assert seen[0].headers["x-acs-action"] == "ListWorkspaces"
    assert seen[0].headers["x-dashscope-sdk-client"].endswith("management2")
    assert len(seen) == 2
    await client.aclose()


async def test_async_with_options_region():
    client = AsyncManagementClient(access_key_id=AK, access_key_secret=SK)
    other = client.with_options(region="eu-central-1")
    assert other.transport.host == "modelstudio.eu-central-1.aliyuncs.com"
    await client.aclose()
    await other.aclose()


async def test_async_write_method_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, json={})

    client = make_async_client(handler, max_retries=2)
    resp = await client.transport.request(
        "POST",
        "/modelstudio/workspaces",
        action="CreateWorkspace",
        body={"name": "prod"},
    )
    assert resp.status_code == 503
    assert calls["n"] == 1
    await client.aclose()


async def test_async_per_call_timeout_overrides_client_timeout():
    seen = []
    client = make_async_client(_capture_handler(seen), timeout=30.0)
    await client.workspaces.list(timeout=1.5)
    assert seen[0].extensions["timeout"]["read"] == 1.5
    await client.aclose()


def _assert_wire_signature(request):
    from types import SimpleNamespace

    auth = request.headers["authorization"]
    names = auth.split("SignedHeaders=", 1)[1].split(",", 1)[0].split(";")
    wire = SimpleNamespace(
        method=request.method,
        url=str(request.url),
        headers={name: request.headers[name] for name in names},
    )
    assert auth == _gateway_recompute(wire, ak=AK, sk=SK)


def _assert_default_query_and_filters(request):
    query = query_of(request)
    assert query["locale"] == "zh-CN"
    assert query["maxResults"] == "10"
    assert query["workspaceName"] == "a b+/~"
    assert query["nextToken"] == "token=+/~"
    assert "workspaceName=a%20b%2B%2F~" in str(request.url)
    _assert_wire_signature(request)


def test_injected_client_default_query_is_signed_without_losing_filters():
    seen = []
    http_client = httpx.Client(
        transport=httpx.MockTransport(_capture_handler(seen)),
        params={"locale": "zh-CN", "maxResults": "99"},
    )
    client = ManagementClient(
        access_key_id=AK,
        access_key_secret=SK,
        http_client=http_client,
    )
    client.workspaces.list(
        max_results=10,
        workspace_name="a b+/~",
        next_token="token=+/~",
    )
    _assert_default_query_and_filters(seen[0])
    assert http_client.params["maxResults"] == "99"
    http_client.close()


async def test_async_injected_client_default_query_is_signed():
    seen = []
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_capture_handler(seen)),
        params={"locale": "zh-CN", "maxResults": "99"},
    )
    client = AsyncManagementClient(
        access_key_id=AK,
        access_key_secret=SK,
        http_client=http_client,
    )
    await client.workspaces.list(
        max_results=10,
        workspace_name="a b+/~",
        next_token="token=+/~",
    )
    _assert_default_query_and_filters(seen[0])
    await http_client.aclose()


def test_injected_default_query_preserves_openapi_array_serialization():
    seen = []
    http_client = httpx.Client(
        transport=httpx.MockTransport(_capture_handler(seen)),
        params={"locale": "zh-CN"},
    )
    client = ManagementClient(
        access_key_id=AK,
        access_key_secret=SK,
        http_client=http_client,
    )
    client.token_plan.seats.list(status_list=["NORMAL", "STOP"])
    assert query_of(seen[0])["StatusList.1"] == "NORMAL"
    assert query_of(seen[0])["StatusList.2"] == "STOP"
    _assert_wire_signature(seen[0])
    http_client.close()


def test_repeated_default_query_is_rejected_instead_of_losing_values():
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail("no wire")),
        params=[("StatusList", "NORMAL"), ("StatusList", "STOP")],
    )
    client = ManagementClient(
        access_key_id=AK,
        access_key_secret=SK,
        http_client=http_client,
    )
    with pytest.raises(ValueError, match="Repeated HTTP-client"):
        client.workspaces.list()
    http_client.close()
