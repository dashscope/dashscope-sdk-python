# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.batches job plane (B2: create/get/list/cancel/wait).

Same compatible-mode wire as B1's File plane (/compatible-mode/v1/batches,
protocol-independent, flat OpenAI Batch object, after/has_more cursor), so
these mirror test_batch_files.py conventions. B2-specific coverage: create
defaults to max_retries=0 (no server idempotency key), completion_window
validation, the Batch-list cursor (limit [1,100], first_id/last_id), the
polling waiter (terminal completed/failed/expired/cancelled) and the
pre-upload validate_input row-level JSONL check.
"""

# Mock handlers intentionally ignore the request argument; pytest fixture
# injection looks like unused/redefined names to pylint.
# pylint: disable=unused-argument,redefined-outer-name

from __future__ import annotations

import json

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.errors import (
    BadRequestError,
    DashScopeError,
    RateLimitError,
    SdkError,
)
from dashscope.resources.batches import Batch, BatchList
from dashscope.transport._ident import sdk_client_value

BASE_URL = "https://api.test.local/api/v1"
_WS_HOST = "https://ws-123.cn-beijing.maas.aliyuncs.com"
_COMPAT_BATCHES = "/compatible-mode/v1/batches"
_ENDPOINT = "/v1/chat/completions"

# A flat OpenAI-compatible Batch object (create / get / cancel response).
_BATCH_BODY = {
    "id": "batch_abc123",
    "object": "batch",
    "endpoint": _ENDPOINT,
    "input_file_id": "file-batch-abc123",
    "completion_window": "24h",
    "status": "validating",
    "created_at": 1730000000,
    "request_counts": {"total": 0, "completed": 0, "failed": 0},
    "metadata": {"ds_name": "my-batch"},
    "errors": None,
    "output_file_id": None,
    "error_file_id": None,
}

_GOOD_ROWS = [
    {
        "custom_id": "1",
        "method": "POST",
        "url": _ENDPOINT,
        "body": {"model": "qwen-plus", "messages": []},
    },
    {
        "custom_id": "2",
        "method": "POST",
        "url": _ENDPOINT,
        "body": {"model": "qwen-plus", "messages": []},
    },
]


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


@pytest.fixture
def clean_env(monkeypatch):
    from dashscope._config import settings

    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_HTTP_BASE_URL",
        "DASHSCOPE_WORKSPACE",
        "DASHSCOPE_REGION",
        "DASHSCOPE_PROTOCOL",
        "DASHSCOPE_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)
    settings.clear()
    yield monkeypatch
    settings.clear()


def _rows_file(tmp_path, rows, name="input.jsonl"):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )
    return str(path)


# --------------------------------------------------------------------------
# create
# --------------------------------------------------------------------------


def test_create_wire_body_and_parse():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_BATCH_BODY)

    client = _make_client(handler)
    resp = client.batches.create(
        input_file_id="file-batch-abc123",
        endpoint=_ENDPOINT,
        completion_window="48h",
        metadata={"ds_name": "n", "ds_description": "d"},
    )
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v1/batches"
    assert captured["body"] == {
        "input_file_id": "file-batch-abc123",
        "endpoint": _ENDPOINT,
        "completion_window": "48h",
        "metadata": {"ds_name": "n", "ds_description": "d"},
    }
    assert resp.id == "batch_abc123"
    assert resp.batch_id == "batch_abc123"
    assert resp.object == "batch"
    assert resp.status == "validating"
    assert resp.finished is False
    assert resp.succeeded is False
    assert resp.request_counts.total == 0
    assert resp.metadata == {"ds_name": "my-batch"}
    assert resp.errors is None
    assert resp.is_error is False
    client.close()


def test_create_omits_metadata_when_unset():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_BATCH_BODY)

    client = _make_client(handler)
    client.batches.create(input_file_id="f", endpoint=_ENDPOINT)
    assert "metadata" not in captured["body"]
    assert captured["body"]["completion_window"] == "24h"
    client.close()


def test_create_defaults_to_no_retry():
    attempts = []

    def handler(request):
        attempts.append(request.url.path)
        return httpx.Response(503, json={"message": "busy"})

    client = _make_client(handler)
    with pytest.raises(DashScopeError):
        client.batches.create(input_file_id="f", endpoint=_ENDPOINT)
    # No idempotency key on the wire -> create must not auto-retry.
    assert len(attempts) == 1
    client.close()


def test_create_explicit_max_retries_retries(monkeypatch):
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, json={"message": "busy"})
        return httpx.Response(200, json=_BATCH_BODY)

    from dashscope.transport import http as transport_http

    monkeypatch.setattr(transport_http, "_sleep", lambda seconds: None)
    client = _make_client(handler)
    resp = client.batches.create(
        input_file_id="f",
        endpoint=_ENDPOINT,
        max_retries=2,
    )
    assert resp.id == "batch_abc123"
    assert len(attempts) == 3
    client.close()


@pytest.mark.parametrize(
    "window",
    ["", "24", "h", "12h", "337h", "0d", "15d", "abc", "24x"],
)
def test_create_rejects_bad_completion_window(window):
    client = _make_client(lambda r: httpx.Response(200, json=_BATCH_BODY))
    with pytest.raises(ValueError, match="completion_window"):
        client.batches.create(
            input_file_id="f",
            endpoint=_ENDPOINT,
            completion_window=window,
        )
    client.close()


@pytest.mark.parametrize("window", ["24h", "336h", "1d", "14d", "7d"])
def test_create_accepts_valid_windows(window):
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_BATCH_BODY)

    client = _make_client(handler)
    client.batches.create(
        input_file_id="f",
        endpoint=_ENDPOINT,
        completion_window=window,
    )
    assert captured["body"]["completion_window"] == window
    client.close()


def test_create_requires_input_file_id_and_endpoint():
    client = _make_client(lambda r: httpx.Response(200, json=_BATCH_BODY))
    with pytest.raises(ValueError, match="input_file_id"):
        client.batches.create(input_file_id="", endpoint=_ENDPOINT)
    with pytest.raises(ValueError, match="endpoint"):
        client.batches.create(input_file_id="f", endpoint="   ")
    client.close()


# --------------------------------------------------------------------------
# get / cancel
# --------------------------------------------------------------------------


def test_get_wire_and_parse():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                **_BATCH_BODY,
                "status": "completed",
                "output_file_id": "file-out",
                "request_counts": {"total": 10, "completed": 10, "failed": 0},
            },
        )

    client = _make_client(handler)
    resp = client.batches.get("batch_abc123")
    assert captured == {
        "method": "GET",
        "path": "/api/v1/batches/batch_abc123",
    }
    assert resp.status == "completed"
    assert resp.finished is True
    assert resp.succeeded is True
    assert resp.output_file_id == "file-out"
    assert resp.request_counts.completed == 10
    client.close()


def test_get_parses_errors_and_counts():
    body = {
        **_BATCH_BODY,
        "status": "failed",
        "errors": {
            "object": "list",
            "data": [
                {"code": "c", "message": "m", "param": "p", "line": 5},
            ],
        },
        "request_counts": {"total": 5, "completed": 3, "failed": 2},
    }

    client = _make_client(lambda r: httpx.Response(200, json=body))
    resp = client.batches.get("b1")
    assert resp.errors.data[0].line == 5
    assert resp.errors.data[0].code == "c"
    assert resp.request_counts.failed == 2
    assert resp.finished is True
    assert resp.succeeded is False
    client.close()


def test_cancel_wire_and_parse():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={**_BATCH_BODY, "status": "cancelling"},
        )

    client = _make_client(handler)
    resp = client.batches.cancel("batch_abc123")
    assert captured == {
        "method": "POST",
        "path": "/api/v1/batches/batch_abc123/cancel",
    }
    assert resp.status == "cancelling"
    assert resp.finished is False
    client.close()


@pytest.mark.parametrize("batch_id", ["", "   "])
def test_empty_batch_id_rejected_locally(batch_id):
    client = _make_client(lambda r: httpx.Response(200, json=_BATCH_BODY))
    for call in (client.batches.get, client.batches.cancel):
        with pytest.raises(ValueError, match="batch_id"):
            call(batch_id)
    client.close()


def test_batch_id_percent_encoded_in_path():
    seen = []

    def handler(request):
        seen.append(request.url.raw_path)
        return httpx.Response(200, json=_BATCH_BODY)

    client = _make_client(handler)
    tricky = "batch/a?b#c"
    client.batches.get(tricky)
    client.batches.cancel(tricky)
    # '/', '?' and '#' stay inside one path segment instead of retargeting
    # the request to another path or query.
    encoded = b"/api/v1/batches/batch%2Fa%3Fb%23c"
    assert seen == [encoded, encoded + b"/cancel"]
    client.close()


async def test_async_batch_id_percent_encoded_in_path():
    seen = []

    def handler(request):
        seen.append(request.url.raw_path)
        return httpx.Response(200, json=_BATCH_BODY)

    client = _make_async_client(handler)
    await client.batches.get("batch/a?b#c")
    assert seen == [b"/api/v1/batches/batch%2Fa%3Fb%23c"]
    await client.aclose()


# --------------------------------------------------------------------------
# list (cursor pagination)
# --------------------------------------------------------------------------


def test_list_cursor_pagination_uses_last_id():
    seen = []

    def handler(request):
        after = request.url.params.get("after")
        seen.append(after)
        if after is None:
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "b1"}, {"id": "b2"}],
                    "first_id": "b1",
                    "last_id": "b2",
                    "has_more": True,
                },
            )
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "b3"}],
                "first_id": "b3",
                "last_id": "b3",
                "has_more": False,
            },
        )

    client = _make_client(handler)
    page = client.batches.list(limit=2, status="completed")
    assert page.object == "list"
    assert page.first_id == "b1"
    assert page.last_id == "b2"
    assert [b.id for b in page.items] == ["b1", "b2"]
    assert page.has_next_page() is True
    pages = list(page.iter_pages())
    assert [[b.id for b in p] for p in pages] == [["b1", "b2"], ["b3"]]
    assert pages[-1].has_next_page() is False
    assert pages[-1].next_page() is None
    # The cursor advances by last_id and re-sends the status filter.
    assert seen == [None, "b2"]
    client.close()


def test_list_empty_page_has_no_next():
    def handler(request):
        return httpx.Response(
            200,
            json={"object": "list", "data": [], "has_more": False},
        )

    client = _make_client(handler)
    page = client.batches.list()
    assert page.items == []
    assert page.has_next_page() is False
    assert list(page.iter_pages()) == [page]
    client.close()


def test_list_params_validation_and_csv():
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={"object": "list", "data": [], "has_more": False},
        )

    client = _make_client(handler)
    for bad_limit in (0, 101):
        with pytest.raises(ValueError, match="limit"):
            client.batches.list(limit=bad_limit)
    with pytest.raises(ValueError, match="input_file_ids"):
        client.batches.list(input_file_ids=[f"f{i}" for i in range(21)])
    # Unset optionals stay off the wire.
    client.batches.list()
    assert captured["params"] == {}
    client.batches.list(
        after="b9",
        limit=100,
        ds_name="foo",
        input_file_ids=["f1", "f2"],
        status=["completed", "failed"],
        create_after="20260101000000",
        create_before="20260201000000",
    )
    assert captured["params"] == {
        "after": "b9",
        "limit": "100",
        "ds_name": "foo",
        "input_file_ids": "f1,f2",
        "status": "completed,failed",
        "create_after": "20260101000000",
        "create_before": "20260201000000",
    }
    # Comma-separated string forms pass through unchanged.
    client.batches.list(input_file_ids="f1,f2", status="completed")
    assert captured["params"]["input_file_ids"] == "f1,f2"
    assert captured["params"]["status"] == "completed"
    client.close()


def test_list_error_page_does_not_bind_pagination():
    def handler(request):
        return httpx.Response(
            400,
            json={"error": {"code": "InvalidParameter", "message": "bad"}},
        )

    client = _make_client(handler, raise_on_error=False)
    page = client.batches.list()
    assert page.is_error is True
    assert page.has_next_page() is False
    assert page.next_page() is None
    client.close()


def test_sync_next_page_rejects_async_callback():
    page = BatchList(data=[Batch(id="b1")], has_more=True, last_id="b1")

    async def fetch(cursor):
        return BatchList(data=[Batch(id="b2")], has_more=False)

    page.bind_pagination(fetch)
    with pytest.raises(TypeError, match="next_page_async"):
        page.next_page()


# --------------------------------------------------------------------------
# error mapping
# --------------------------------------------------------------------------


def test_http_error_maps_openai_nested_body():
    def handler(request):
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "InvalidParameter",
                    "message": "endpoint is invalid",
                },
            },
            headers={"x-request-id": "rid-err"},
        )

    client = _make_client(handler)
    with pytest.raises(BadRequestError) as ei:
        client.batches.get("batch_x")
    assert ei.value.code == "InvalidParameter"
    assert ei.value.status_code == 400
    assert ei.value.request_id == "rid-err"
    client.close()


def test_http_429_maps_rate_limit():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    client = _make_client(handler)
    with pytest.raises(RateLimitError):
        client.batches.list()
    client.close()


def test_error_envelope_when_not_raising():
    def handler(request):
        return httpx.Response(
            404,
            json={
                "error": {
                    "code": "Batch.NotFound",
                    "message": "batch batch_typo does not exist",
                },
            },
            headers={"x-request-id": "rid-404"},
        )

    client = _make_client(handler, raise_on_error=False)
    resp = client.batches.get("batch_typo")
    assert resp.is_error is True
    assert resp.status_code == 404
    assert resp.code == "Batch.NotFound"
    assert resp.message == "batch batch_typo does not exist"
    assert resp.request_id == "rid-404"
    assert resp.id is None
    assert resp.headers["x-request-id"] == "rid-404"
    client.close()


def test_business_error_layer_raise_and_report():
    body = {
        "code": "InvalidParameter",
        "message": "endpoint unsupported",
        "request_id": "rid-biz",
    }

    def handler(request):
        return httpx.Response(200, json=body)

    client = _make_client(handler)
    with pytest.raises(BadRequestError) as ei:
        client.batches.get("b1")
    assert ei.value.status_code == 200
    client.close()

    client = _make_client(handler, raise_on_error=False)
    resp = client.batches.get("b1")
    assert resp.is_error is True
    assert resp.code == "InvalidParameter"
    assert resp.status_code == 200
    client.close()


# --------------------------------------------------------------------------
# waiter
# --------------------------------------------------------------------------


def test_wait_polls_until_completed():
    statuses = iter(["validating", "in_progress", "completed"])
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json={**_BATCH_BODY, "status": next(statuses)},
        )

    client = _make_client(handler)
    sleeps = []
    result = client.batches.wait(
        "batch_abc123",
        _sleep=lambda s: sleeps.append(s),
    )
    assert result.status == "completed"
    assert result.succeeded is True
    assert len(calls) == 3
    # Slept between polls, never after the terminal status.
    assert len(sleeps) == 2
    client.close()


def test_wait_accepts_batch_object_skipping_initial_get():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(
            200,
            json={**_BATCH_BODY, "status": "completed"},
        )

    client = _make_client(handler)
    seed = Batch(id="batch_abc123", status="in_progress")
    result = client.batches.wait(seed, _sleep=lambda s: None)
    assert result.status == "completed"
    assert len(calls) == 1
    client.close()


def test_wait_failed_raises_with_errors_detail():
    def handler(request):
        return httpx.Response(
            200,
            json={
                **_BATCH_BODY,
                "status": "failed",
                "errors": {
                    "object": "list",
                    "data": [
                        {
                            "code": "invalid_row",
                            "message": "row 3 bad",
                            "line": 3,
                        },
                    ],
                },
            },
        )

    client = _make_client(handler)
    with pytest.raises(DashScopeError) as ei:
        client.batches.wait("batch_abc123", _sleep=lambda s: None)
    assert "row 3 bad" in ei.value.message
    client.close()


def test_wait_failed_no_raise_returns_batch():
    def handler(request):
        return httpx.Response(200, json={**_BATCH_BODY, "status": "failed"})

    client = _make_client(handler, raise_on_error=False)
    result = client.batches.wait("batch_abc123", _sleep=lambda s: None)
    assert result.status == "failed"
    assert result.finished is True
    assert result.succeeded is False
    # HTTP 200 with no business code: the call succeeded, the job failed.
    assert result.is_error is False
    client.close()


@pytest.mark.parametrize("status", ["cancelled", "expired"])
def test_wait_cancelled_expired_raise(status):
    def handler(request):
        return httpx.Response(200, json={**_BATCH_BODY, "status": status})

    client = _make_client(handler)
    with pytest.raises(DashScopeError, match=status):
        client.batches.wait("batch_abc123", _sleep=lambda s: None)
    client.close()


def test_wait_timeout():
    def handler(request):
        return httpx.Response(
            200,
            json={**_BATCH_BODY, "status": "in_progress"},
        )

    client = _make_client(handler)
    with pytest.raises(DashScopeError, match="timed out"):
        client.batches.wait(
            "batch_abc123",
            timeout=0,
            _sleep=lambda s: None,
        )
    client.close()


def test_wait_transport_error_no_raise_returns_envelope():
    def handler(request):
        return httpx.Response(503, json={"message": "busy"})

    client = _make_client(handler, raise_on_error=False, max_retries=0)
    result = client.batches.wait("batch_abc123", _sleep=lambda s: None)
    assert result.is_error is True
    assert result.status is None
    client.close()


# --------------------------------------------------------------------------
# validate_input (pre-upload row-level JSONL check)
# --------------------------------------------------------------------------


def test_validate_input_accepts_good_rows(tmp_path):
    client = _make_client(lambda r: httpx.Response(200, json=_BATCH_BODY))
    n = client.batches.validate_input(
        _rows_file(tmp_path, _GOOD_ROWS),
        endpoint=_ENDPOINT,
    )
    assert n == 2
    client.close()


def test_validate_input_duplicate_custom_id(tmp_path):
    rows = [dict(_GOOD_ROWS[0]), dict(_GOOD_ROWS[0])]
    client = _make_client(lambda r: httpx.Response(200, json=_BATCH_BODY))
    with pytest.raises(ValueError, match="duplicate custom_id"):
        client.batches.validate_input(_rows_file(tmp_path, rows))
    client.close()


def test_validate_input_model_mismatch(tmp_path):
    rows = [
        dict(_GOOD_ROWS[0]),
        {**_GOOD_ROWS[1], "body": {"model": "qwen-max"}},
    ]
    client = _make_client(lambda r: httpx.Response(200, json=_BATCH_BODY))
    with pytest.raises(ValueError, match="same model"):
        client.batches.validate_input(_rows_file(tmp_path, rows))
    client.close()


def test_validate_input_endpoint_mismatch(tmp_path):
    client = _make_client(lambda r: httpx.Response(200, json=_BATCH_BODY))
    with pytest.raises(ValueError, match="does not match"):
        client.batches.validate_input(
            _rows_file(tmp_path, _GOOD_ROWS),
            endpoint="/v1/embeddings",
        )
    client.close()


def test_validate_input_bad_method(tmp_path):
    rows = [{**_GOOD_ROWS[0], "method": "GET"}]
    client = _make_client(lambda r: httpx.Response(200, json=_BATCH_BODY))
    with pytest.raises(ValueError, match="method"):
        client.batches.validate_input(_rows_file(tmp_path, rows))
    client.close()


def test_validate_input_missing_body_model(tmp_path):
    rows = [
        {"custom_id": "1", "method": "POST", "url": _ENDPOINT, "body": {}},
    ]
    client = _make_client(lambda r: httpx.Response(200, json=_BATCH_BODY))
    with pytest.raises(ValueError, match="body.model"):
        client.batches.validate_input(_rows_file(tmp_path, rows))
    client.close()


def test_validate_input_not_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    client = _make_client(lambda r: httpx.Response(200, json=_BATCH_BODY))
    with pytest.raises(ValueError, match="not valid JSON"):
        client.batches.validate_input(str(path))
    client.close()


def test_validate_input_missing_file_is_io_error(tmp_path):
    client = _make_client(lambda r: httpx.Response(200, json=_BATCH_BODY))
    with pytest.raises(SdkError) as ei:
        client.batches.validate_input(str(tmp_path / "nope.jsonl"))
    assert ei.value.code == "sdk.IoError"
    client.close()


def test_validate_input_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    client = _make_client(lambda r: httpx.Response(200, json=_BATCH_BODY))
    with pytest.raises(ValueError, match="no batch input rows"):
        client.batches.validate_input(str(path))
    client.close()


# --------------------------------------------------------------------------
# SDK identification header
# --------------------------------------------------------------------------


def test_sdk_module_header():
    seen = {}

    def handler(request):
        seen["client"] = request.headers.get("x-dashscope-sdk-client")
        return httpx.Response(200, json=_BATCH_BODY)

    client = _make_client(handler)
    client.batches.get("b1")
    assert seen["client"] == sdk_client_value("batches")
    assert seen["client"].endswith("/batches2")
    client.close()


# --------------------------------------------------------------------------
# routing (compatible-mode path, protocol-independent)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", ["native", "openai", "anthropic"])
def test_routing_default_gateway_ignores_chat_protocol(clean_env, protocol):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_BATCH_BODY)

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=_WS_HOST + "/api/v1",
    )
    client = Client(
        api_key="sk-test",
        workspace="ws-123",
        protocol=protocol,
        http_client=http_client,
    )
    client.batches.get("b1")
    assert urls == [_WS_HOST + _COMPAT_BATCHES + "/b1"], protocol
    client.close()


def test_routing_explicit_base_url():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_BATCH_BODY)

    client = _make_client(handler)
    client.batches.create(input_file_id="f", endpoint=_ENDPOINT)
    assert urls == [BASE_URL + "/batches"]
    client.close()


def test_raw_returns_apiresponse_before_error_check():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad"}})

    client = _make_client(handler)
    raw = client.batches.get("b1", raw=True)
    assert raw.status_code == 400
    assert raw.json()["error"]["message"] == "bad"
    client.close()


# --------------------------------------------------------------------------
# async symmetry
# --------------------------------------------------------------------------


async def test_async_create_get_cancel():
    captured = []

    def handler(request):
        captured.append((request.method, request.url.path))
        if request.url.path.endswith("/cancel"):
            return httpx.Response(
                200,
                json={**_BATCH_BODY, "status": "cancelling"},
            )
        return httpx.Response(200, json=_BATCH_BODY)

    client = _make_async_client(handler)
    created = await client.batches.create(
        input_file_id="f",
        endpoint=_ENDPOINT,
    )
    assert created.id == "batch_abc123"
    got = await client.batches.get("batch_abc123")
    assert got.status == "validating"
    cancelled = await client.batches.cancel("batch_abc123")
    assert cancelled.status == "cancelling"
    assert captured == [
        ("POST", "/api/v1/batches"),
        ("GET", "/api/v1/batches/batch_abc123"),
        ("POST", "/api/v1/batches/batch_abc123/cancel"),
    ]
    await client.aclose()


async def test_async_create_defaults_to_no_retry():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(503, json={"message": "busy"})

    client = _make_async_client(handler)
    with pytest.raises(DashScopeError):
        await client.batches.create(input_file_id="f", endpoint=_ENDPOINT)
    assert len(attempts) == 1
    await client.aclose()


async def test_async_list_cursor_pagination():
    seen = []

    def handler(request):
        after = request.url.params.get("after")
        seen.append(after)
        if after is None:
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "b1"}],
                    "first_id": "b1",
                    "last_id": "b1",
                    "has_more": True,
                },
            )
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "b2"}],
                "first_id": "b2",
                "last_id": "b2",
                "has_more": False,
            },
        )

    client = _make_async_client(handler)
    page = await client.batches.list()
    pages = [p async for p in page.iter_pages_async()]
    assert [[b.id for b in p] for p in pages] == [["b1"], ["b2"]]
    assert seen == [None, "b1"]
    assert [b.id async for b in page] == ["b1"]
    await client.aclose()


async def test_async_wait_until_completed():
    statuses = iter(["in_progress", "completed"])

    def handler(request):
        return httpx.Response(
            200,
            json={**_BATCH_BODY, "status": next(statuses)},
        )

    async def nosleep(seconds):
        return None

    client = _make_async_client(handler)
    result = await client.batches.wait("batch_abc123", _sleep=nosleep)
    assert result.status == "completed"
    await client.aclose()


async def test_async_wait_failed_raises():
    def handler(request):
        return httpx.Response(
            200,
            json={
                **_BATCH_BODY,
                "status": "failed",
                "errors": {"object": "list", "data": [{"message": "boom"}]},
            },
        )

    async def nosleep(seconds):
        return None

    client = _make_async_client(handler)
    with pytest.raises(DashScopeError, match="boom"):
        await client.batches.wait("batch_abc123", _sleep=nosleep)
    await client.aclose()


async def test_async_routing_default_gateway(clean_env):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_BATCH_BODY)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=_WS_HOST + "/api/v1",
    )
    client = AsyncClient(
        api_key="sk-test",
        workspace="ws-123",
        http_client=http_client,
    )
    await client.batches.get("b1")
    assert urls == [_WS_HOST + _COMPAT_BATCHES + "/b1"]
    await client.aclose()


async def test_async_validate_input_is_local(tmp_path):
    client = _make_async_client(
        lambda r: httpx.Response(200, json=_BATCH_BODY),
    )
    # Plain (non-async) method: local file I/O, no network.
    n = client.batches.validate_input(_rows_file(tmp_path, _GOOD_ROWS))
    assert n == 2
    await client.aclose()


async def test_async_empty_batch_id_rejected_locally():
    client = _make_async_client(
        lambda r: httpx.Response(200, json=_BATCH_BODY),
    )
    with pytest.raises(ValueError, match="batch_id"):
        await client.batches.get("")
    await client.aclose()
