# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.batches.files (B1: compatible-mode File sub-resource).

The batch File plane sits under /compatible-mode/v1/files on every gateway,
independent of the client's chat ``protocol`` -- the routing tests pin that
for native / openai / anthropic alike, mirroring test_decision.py. It is a
different wire from the native client.files (multipart ``file`` singular,
flat OpenAI object, ``after``/``has_more`` cursor), so these tests assert
that shape rather than the native ``data.uploaded_files`` / page_no one.
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
    NotFoundError,
    RateLimitError,
    SdkError,
)
from dashscope.resources.batches import (
    BatchFile,
    BatchFileList,
    FileContent,
)
from dashscope.transport._ident import sdk_client_value

BASE_URL = "https://api.test.local/api/v1"
_WS_HOST = "https://ws-123.cn-beijing.maas.aliyuncs.com"
_COMPAT_FILES = "/compatible-mode/v1/files"

# A flat OpenAI-compatible file object (upload / get response).
_FILE_BODY = {
    "id": "file-batch-abc123",
    "object": "file",
    "bytes": 2048,
    "created_at": 1730000000,
    "filename": "batch_input.jsonl",
    "purpose": "batch",
    "status": "processed",
}

_JSONL = b'{"custom_id": "1"}\n{"custom_id": "2"}\n'


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


def _jsonl_file(tmp_path, name="batch_input.jsonl"):
    path = tmp_path / name
    path.write_bytes(_JSONL)
    return str(path)


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------


def test_upload_builds_multipart_and_parses_flat_object(tmp_path):
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["method"] = request.method
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content.decode("utf-8", "replace")
        return httpx.Response(200, json=_FILE_BODY)

    client = _make_client(handler)
    resp = client.batches.files.upload(file_path=_jsonl_file(tmp_path))
    # Explicit base_url stays the service root (same rule as chat/decision).
    assert captured["path"] == "/api/v1/files"
    assert captured["method"] == "POST"
    assert captured["content_type"].startswith("multipart/form-data")
    # Compat-mode uses ``file`` (singular) and DOES send purpose, unlike the
    # native client.files upload (``files`` plural, purpose omitted).
    assert 'name="file"' in captured["body"]
    assert 'name="files"' not in captured["body"]
    assert 'name="purpose"' in captured["body"]
    assert "batch" in captured["body"]
    assert "batch_input.jsonl" in captured["body"]
    assert resp.id == "file-batch-abc123"
    assert resp.file_id == "file-batch-abc123"
    assert resp.object == "file"
    assert resp.bytes == 2048
    assert resp.created_at == 1730000000
    assert resp.filename == "batch_input.jsonl"
    assert resp.purpose == "batch"
    assert resp.is_error is False
    client.close()


def test_upload_missing_file_is_catalog_io_error(tmp_path):
    client = _make_client(lambda r: httpx.Response(200, json=_FILE_BODY))
    with pytest.raises(SdkError) as ei:
        client.batches.files.upload(file_path=str(tmp_path / "nope.jsonl"))
    assert ei.value.code == "sdk.IoError"
    assert "nope.jsonl" in ei.value.message
    assert ei.value.allow_retry is False
    client.close()


def test_upload_batch_purpose_requires_jsonl(tmp_path):
    client = _make_client(lambda r: httpx.Response(200, json=_FILE_BODY))
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl"):
        client.batches.files.upload(file_path=str(bad), purpose="batch")
    # file-extract accepts non-JSONL documents: no JSONL gate for it.
    resp = client.batches.files.upload(
        file_path=str(bad),
        purpose="file-extract",
    )
    assert resp.id == "file-batch-abc123"
    client.close()


def test_upload_retry_resends_full_body(tmp_path, monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(503, json={"message": "busy"})
        return httpx.Response(200, json=_FILE_BODY)

    from dashscope.transport import http as transport_http

    monkeypatch.setattr(transport_http, "_sleep", lambda seconds: None)
    client = _make_client(handler)
    resp = client.batches.files.upload(file_path=_jsonl_file(tmp_path))
    assert resp.id == "file-batch-abc123"
    assert len(bodies) == 2
    for body in bodies:
        assert b'{"custom_id": "1"}' in body
        assert b"batch_input.jsonl" in body
    client.close()


# --------------------------------------------------------------------------
# content download
# --------------------------------------------------------------------------


def test_content_download_returns_raw_bytes(tmp_path):
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["method"] = request.method
        # JSONL is many JSON docs, not one: a JSON parse would fail. The
        # download must hand back the raw bytes untouched.
        return httpx.Response(
            200,
            content=_JSONL,
            headers={
                "content-type": "application/jsonl",
                "x-request-id": "rid-dl",
            },
        )

    client = _make_client(handler)
    content = client.batches.files.content("file-batch-out")
    assert captured["path"] == "/api/v1/files/file-batch-out/content"
    assert captured["method"] == "GET"
    assert isinstance(content, FileContent)
    assert isinstance(content, bytes)
    assert bytes(content) == _JSONL
    assert content.text == _JSONL.decode("utf-8")
    assert content.request_id == "rid-dl"
    assert content.status_code == 200
    assert content.is_error is False
    dest = content.write_to_file(tmp_path / "nested" / "result.jsonl")
    assert dest.read_bytes() == _JSONL
    client.close()


def test_content_error_raises():
    def handler(request):
        return httpx.Response(
            404,
            json={"error": {"message": "file does not exist"}},
            headers={"x-request-id": "rid-404"},
        )

    client = _make_client(handler)
    with pytest.raises(NotFoundError) as ei:
        client.batches.files.content("missing")
    assert ei.value.status_code == 404
    assert ei.value.request_id == "rid-404"
    client.close()


def test_content_error_no_raise_reports_status():
    def handler(request):
        return httpx.Response(
            404,
            json={"error": {"message": "file does not exist"}},
            headers={"x-request-id": "rid-404"},
        )

    client = _make_client(handler, raise_on_error=False)
    content = client.batches.files.content("missing")
    assert content.is_error is True
    assert content.status_code == 404
    assert content.request_id == "rid-404"
    # The body is the gateway's error bytes, not a parsed envelope.
    assert b"file does not exist" in bytes(content)
    client.close()


# --------------------------------------------------------------------------
# list (cursor pagination)
# --------------------------------------------------------------------------


def test_list_cursor_pagination_preserves_filters():
    seen = []

    def handler(request):
        after = request.url.params.get("after")
        seen.append(
            {
                "path": request.url.path,
                "purpose": request.url.params.get("purpose"),
                "after": after,
                "limit": request.url.params.get("limit"),
            },
        )
        if after is None:
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "f1"}, {"id": "f2"}],
                    "has_more": True,
                },
            )
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "f3"}],
                "has_more": False,
            },
        )

    client = _make_client(handler)
    page = client.batches.files.list(purpose="batch", limit=2)
    assert page.object == "list"
    assert [f.id for f in page.items] == ["f1", "f2"]
    assert page.has_more is True
    assert page.has_next_page() is True
    pages = list(page.iter_pages())
    assert [[f.id for f in p] for p in pages] == [["f1", "f2"], ["f3"]]
    assert pages[-1].has_next_page() is False
    assert pages[-1].next_page() is None
    # The cursor callback re-sends purpose/limit and advances ``after``.
    assert seen[0] == {
        "path": "/api/v1/files",
        "purpose": "batch",
        "after": None,
        "limit": "2",
    }
    assert seen[1]["after"] == "f2"
    assert seen[1]["purpose"] == "batch"
    assert seen[1]["limit"] == "2"
    client.close()


def test_list_empty_page_has_no_next():
    def handler(request):
        return httpx.Response(
            200,
            json={"object": "list", "data": [], "has_more": False},
        )

    client = _make_client(handler)
    page = client.batches.files.list()
    assert page.items == []
    assert page.has_next_page() is False
    assert list(page.iter_pages()) == [page]
    client.close()


def test_list_params_validation_and_omission():
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={"object": "list", "data": [], "has_more": False},
        )

    client = _make_client(handler)
    for bad_limit in (0, 2001):
        with pytest.raises(ValueError, match="limit"):
            client.batches.files.list(limit=bad_limit)
    # Unset optionals stay off the wire.
    client.batches.files.list()
    assert captured["params"] == {}
    client.batches.files.list(
        purpose="batch",
        after="f9",
        limit=2000,
        create_before="fb",
        create_after="fa",
    )
    assert captured["params"] == {
        "purpose": "batch",
        "after": "f9",
        "limit": "2000",
        "create_before": "fb",
        "create_after": "fa",
    }
    client.close()


def test_sync_next_page_rejects_async_callback():
    page = BatchFileList(data=[BatchFile(id="f1")], has_more=True)

    async def fetch(cursor):
        return BatchFileList(data=[BatchFile(id="f2")], has_more=False)

    page.bind_pagination(fetch)
    with pytest.raises(TypeError, match="next_page_async"):
        page.next_page()


# --------------------------------------------------------------------------
# get / delete
# --------------------------------------------------------------------------


def test_get_and_delete():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "DELETE":
            return httpx.Response(
                200,
                json={
                    "id": "file-batch-abc123",
                    "object": "file",
                    "deleted": True,
                },
            )
        return httpx.Response(200, json=_FILE_BODY)

    client = _make_client(handler)
    got = client.batches.files.get("file-batch-abc123")
    assert got.id == "file-batch-abc123"
    assert got.purpose == "batch"
    deleted = client.batches.files.delete("file-batch-abc123")
    assert deleted.deleted is True
    assert deleted.id == "file-batch-abc123"
    assert seen == [
        ("GET", "/api/v1/files/file-batch-abc123"),
        ("DELETE", "/api/v1/files/file-batch-abc123"),
    ]
    client.close()


@pytest.mark.parametrize("file_id", ["", "   "])
def test_empty_file_id_rejected_locally(file_id):
    client = _make_client(lambda r: httpx.Response(200, json=_FILE_BODY))
    for call in (
        client.batches.files.get,
        client.batches.files.delete,
        client.batches.files.content,
    ):
        with pytest.raises(ValueError, match="file_id"):
            call(file_id)
    client.close()


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
                    "message": "purpose is invalid",
                },
            },
            headers={"x-request-id": "rid-err"},
        )

    client = _make_client(handler)
    with pytest.raises(BadRequestError) as ei:
        client.batches.files.get("file-batch-abc123")
    assert ei.value.code == "InvalidParameter"
    assert ei.value.status_code == 400
    assert ei.value.request_id == "rid-err"
    client.close()


def test_http_429_maps_rate_limit():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    client = _make_client(handler)
    with pytest.raises(RateLimitError):
        client.batches.files.list()
    client.close()


def test_error_envelope_when_not_raising():
    def handler(request):
        return httpx.Response(
            404,
            json={
                "error": {
                    "code": "File.NotFound",
                    "message": "file file-batch-typo does not exist",
                },
            },
            headers={"x-request-id": "rid-404"},
        )

    client = _make_client(handler, raise_on_error=False)
    resp = client.batches.files.get("file-batch-typo")
    assert resp.is_error is True
    assert resp.status_code == 404
    assert resp.code == "File.NotFound"
    assert resp.message == "file file-batch-typo does not exist"
    assert resp.request_id == "rid-404"
    assert resp.id is None
    assert resp.headers["x-request-id"] == "rid-404"
    client.close()


def test_business_error_layer_raise_and_report():
    body = {
        "code": "InvalidParameter",
        "message": "purpose unsupported",
        "request_id": "rid-biz",
    }

    def handler(request):
        return httpx.Response(200, json=body)

    client = _make_client(handler)
    with pytest.raises(BadRequestError) as ei:
        client.batches.files.get("f1")
    assert ei.value.status_code == 200
    client.close()

    client = _make_client(handler, raise_on_error=False)
    resp = client.batches.files.get("f1")
    assert resp.is_error is True
    assert resp.code == "InvalidParameter"
    assert resp.status_code == 200
    client.close()


def test_list_error_page_does_not_bind_pagination():
    def handler(request):
        return httpx.Response(
            400,
            json={"error": {"code": "InvalidParameter", "message": "bad"}},
        )

    client = _make_client(handler, raise_on_error=False)
    page = client.batches.files.list()
    assert page.is_error is True
    assert page.has_next_page() is False
    assert page.next_page() is None
    client.close()


# --------------------------------------------------------------------------
# SDK identification header
# --------------------------------------------------------------------------


def test_sdk_module_header():
    seen = {}

    def handler(request):
        seen["client"] = request.headers.get("x-dashscope-sdk-client")
        return httpx.Response(200, json=_FILE_BODY)

    client = _make_client(handler)
    client.batches.files.get("f1")
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
        return httpx.Response(200, json=_FILE_BODY)

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
    client.batches.files.get("f1")
    assert urls == [_WS_HOST + _COMPAT_FILES + "/f1"], protocol
    client.close()


@pytest.mark.parametrize("region", ["cn-beijing", "ap-southeast-1"])
def test_routing_documented_region_gateways(clean_env, region):
    urls = []
    ws_host = f"https://ws-123.{region}.maas.aliyuncs.com"

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(
            200,
            json={"object": "list", "data": [], "has_more": False},
        )

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=ws_host + "/api/v1",
    )
    client = Client(
        api_key="sk-test",
        workspace="ws-123",
        region=region,
        http_client=http_client,
    )
    client.batches.files.list(purpose="batch")
    assert urls[0].startswith(ws_host + _COMPAT_FILES)
    client.close()


def test_routing_explicit_base_url():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_FILE_BODY)

    client = _make_client(handler)
    client.batches.files.get("f1")
    assert urls == [BASE_URL + "/files/f1"]
    client.close()


# --------------------------------------------------------------------------
# async symmetry
# --------------------------------------------------------------------------


async def test_async_upload_and_get(tmp_path):
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = request.content.decode("utf-8", "replace")
        return httpx.Response(200, json=_FILE_BODY)

    client = _make_async_client(handler)
    resp = await client.batches.files.upload(file_path=_jsonl_file(tmp_path))
    assert captured["path"] == "/api/v1/files"
    assert 'name="file"' in captured["body"]
    assert resp.file_id == "file-batch-abc123"
    got = await client.batches.files.get("file-batch-abc123")
    assert got.id == "file-batch-abc123"
    await client.aclose()


async def test_async_content_download(tmp_path):
    def handler(request):
        return httpx.Response(
            200,
            content=_JSONL,
            headers={"x-request-id": "rid-dl"},
        )

    client = _make_async_client(handler)
    content = await client.batches.files.content("file-batch-out")
    assert bytes(content) == _JSONL
    assert content.text == _JSONL.decode("utf-8")
    assert content.request_id == "rid-dl"
    dest = content.write_to_file(tmp_path / "out.jsonl")
    assert dest.read_bytes() == _JSONL
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
                    "data": [{"id": "f1"}],
                    "has_more": True,
                },
            )
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "f2"}],
                "has_more": False,
            },
        )

    client = _make_async_client(handler)
    page = await client.batches.files.list(purpose="batch")
    pages = [p async for p in page.iter_pages_async()]
    assert [[f.id for f in p] for p in pages] == [["f1"], ["f2"]]
    assert seen == [None, "f1"]
    # __aiter__ walks the current page's items.
    assert [f.id async for f in page] == ["f1"]
    await client.aclose()


async def test_async_delete_and_error_envelope():
    def handler(request):
        if request.method == "DELETE":
            return httpx.Response(
                200,
                json={"id": "f1", "object": "file", "deleted": True},
            )
        return httpx.Response(
            404,
            json={"error": {"code": "File.NotFound", "message": "gone"}},
            headers={"x-request-id": "rid-404"},
        )

    client = _make_async_client(handler)
    deleted = await client.batches.files.delete("f1")
    assert deleted.deleted is True
    await client.aclose()

    client = _make_async_client(handler, raise_on_error=False)
    resp = await client.batches.files.get("missing")
    assert resp.is_error is True
    assert resp.status_code == 404
    assert resp.code == "File.NotFound"
    assert resp.request_id == "rid-404"
    await client.aclose()


async def test_async_routing_default_gateway(clean_env):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_FILE_BODY)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=_WS_HOST + "/api/v1",
    )
    client = AsyncClient(
        api_key="sk-test",
        workspace="ws-123",
        http_client=http_client,
    )
    await client.batches.files.get("f1")
    assert urls == [_WS_HOST + _COMPAT_FILES + "/f1"]
    await client.aclose()


async def test_async_empty_file_id_rejected_locally():
    client = _make_async_client(lambda r: httpx.Response(200, json=_FILE_BODY))
    with pytest.raises(ValueError, match="file_id"):
        await client.batches.files.get("")
    await client.aclose()


def test_raw_returns_apiresponse_before_error_check():
    def handler(request):
        return httpx.Response(
            400,
            json={"error": {"message": "bad"}},
        )

    client = _make_client(handler)
    raw = client.batches.files.get("f1", raw=True)
    # raw=True bypasses the error check: no raise, status readable.
    assert raw.status_code == 400
    assert raw.json()["error"]["message"] == "bad"
    client.close()


def test_upload_wire_body_is_jsonl_lines(tmp_path):
    captured = {}

    def handler(request):
        captured["body"] = request.content
        return httpx.Response(200, json=_FILE_BODY)

    path = tmp_path / "rows.jsonl"
    rows = [{"custom_id": str(i), "method": "POST"} for i in range(3)]
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )
    client = _make_client(handler)
    client.batches.files.upload(file_path=str(path))
    for row in rows:
        assert json.dumps(row).encode() in captured["body"]
    client.close()