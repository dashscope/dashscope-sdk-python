# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.batches result download + input reuse (B3).

B3 layers ``download_output`` / ``download_error`` on the B2 job plane:
resolve a batch's ``output_file_id`` / ``error_file_id`` (fetching the batch
first when given an id) and download the results JSONL through B1's
``files.content`` (``GET /files/{id}/content``). Returns ``None`` when the
file id is absent. Also covers input-file reuse (the same uploaded id drives
many creates; ``files.list(purpose="batch")`` discovers existing ids).

Same compatible-mode wire and mock conventions as test_batches.py /
test_batch_files.py: with an explicit ``base_url`` the resource appends its
path to that root, so paths read ``/api/v1/batches/...`` and
``/api/v1/files/.../content``.
"""

# Mock handlers intentionally ignore the request argument; pytest fixture
# injection looks like unused/redefined names to pylint.
# pylint: disable=unused-argument,redefined-outer-name

from __future__ import annotations

import json

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.errors import NotFoundError
from dashscope.resources.batches import Batch, BatchFileList, FileContent

BASE_URL = "https://api.test.local/api/v1"
_ENDPOINT = "/v1/chat/ds-test"
_INPUT_FILE_ID = "file-batch-abc123"
_OUTPUT_FILE_ID = "file-batch-out"
_ERROR_FILE_ID = "file-batch-err"
_BATCH_ID = "batch_abc123"

_OUTPUT_JSONL = (
    json.dumps(
        {
            "id": "r1",
            "custom_id": "1",
            "response": {
                "status_code": 200,
                "request_id": "r1",
                "body": {
                    "model": "batch-test-model",
                    "choices": [
                        {"message": {"content": "This is a test result."}},
                    ],
                },
            },
            "error": None,
        },
    )
    + "\n"
    + json.dumps(
        {
            "id": "r2",
            "custom_id": "2",
            "response": {
                "status_code": 200,
                "request_id": "r2",
                "body": {
                    "model": "batch-test-model",
                    "choices": [
                        {"message": {"content": "This is a test result."}},
                    ],
                },
            },
            "error": None,
        },
    )
    + "\n"
).encode("utf-8")

_ERROR_JSONL = (
    json.dumps(
        {
            "id": "r3",
            "custom_id": "3",
            "response": None,
            "error": {"code": "InvalidParameter", "message": "bad row"},
        },
    )
    + "\n"
).encode("utf-8")


def _batch_body(**overrides):
    body = {
        "id": _BATCH_ID,
        "object": "batch",
        "endpoint": _ENDPOINT,
        "input_file_id": _INPUT_FILE_ID,
        "completion_window": "24h",
        "status": "completed",
        "created_at": 1730000000,
        "request_counts": {"total": 2, "completed": 2, "failed": 0},
        "errors": None,
        "output_file_id": _OUTPUT_FILE_ID,
        "error_file_id": None,
    }
    body.update(overrides)
    return body


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


def _content_handler(captured, *, content=_OUTPUT_JSONL, status=200):
    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["sdk"] = request.headers.get("x-dashscope-sdk-client")
        if status >= 400:
            return httpx.Response(
                status,
                json={"error": {"message": "file does not exist"}},
                headers={"x-request-id": "rid-404"},
            )
        return httpx.Response(
            status,
            content=content,
            headers={
                "content-type": "application/jsonl",
                "x-request-id": "rid-dl",
            },
        )

    return handler


# --------------------------------------------------------------------------
# download_output / download_error (sync)
# --------------------------------------------------------------------------


def test_download_output_from_batch_object(tmp_path):
    captured = {}
    client = _make_client(_content_handler(captured))
    batch = Batch(**_batch_body())
    content = client.batches.download_output(batch)
    # A Batch object is used as-is: no GET /batches round-trip.
    assert captured["method"] == "GET"
    assert captured["path"] == f"/api/v1/files/{_OUTPUT_FILE_ID}/content"
    assert captured["sdk"].endswith("/batches2")
    assert isinstance(content, FileContent)
    assert bytes(content) == _OUTPUT_JSONL
    assert "This is a test result." in content.text
    assert content.request_id == "rid-dl"
    assert content.status_code == 200
    assert content.is_error is False
    dest = content.write_to_file(tmp_path / "nested" / "result.jsonl")
    assert dest.read_bytes() == _OUTPUT_JSONL
    rows = [json.loads(line) for line in content.text.splitlines()]
    assert [r["custom_id"] for r in rows] == ["1", "2"]
    client.close()


def test_download_output_from_batch_id_fetches_first():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == f"/api/v1/batches/{_BATCH_ID}":
            return httpx.Response(200, json=_batch_body())
        if request.url.path == f"/api/v1/files/{_OUTPUT_FILE_ID}/content":
            return httpx.Response(200, content=_OUTPUT_JSONL)
        raise AssertionError(f"unexpected path {request.url.path}")

    client = _make_client(handler)
    content = client.batches.download_output(_BATCH_ID)
    assert calls == [
        ("GET", f"/api/v1/batches/{_BATCH_ID}"),
        ("GET", f"/api/v1/files/{_OUTPUT_FILE_ID}/content"),
    ]
    assert bytes(content) == _OUTPUT_JSONL
    client.close()


def test_download_output_none_when_no_output_file():
    def handler(request):
        raise AssertionError("no network call expected when file id absent")

    client = _make_client(handler)
    batch = Batch(**_batch_body(status="in_progress", output_file_id=None))
    assert client.batches.download_output(batch) is None
    client.close()


def test_download_output_none_when_blank_output_file():
    def handler(request):
        raise AssertionError("no network call expected for blank file id")

    client = _make_client(handler)
    batch = Batch(**_batch_body(output_file_id="   "))
    assert client.batches.download_output(batch) is None
    client.close()


def test_download_error_from_batch_object():
    captured = {}
    client = _make_client(
        _content_handler(captured, content=_ERROR_JSONL),
    )
    batch = Batch(**_batch_body(error_file_id=_ERROR_FILE_ID))
    content = client.batches.download_error(batch)
    assert captured["path"] == f"/api/v1/files/{_ERROR_FILE_ID}/content"
    assert bytes(content) == _ERROR_JSONL
    assert "bad row" in content.text
    client.close()


def test_download_error_none_when_no_errors():
    def handler(request):
        raise AssertionError("no network call expected when no error file")

    client = _make_client(handler)
    batch = Batch(**_batch_body(error_file_id=None))
    assert client.batches.download_error(batch) is None
    client.close()


def test_download_output_content_error_raises():
    captured = {}
    client = _make_client(_content_handler(captured, status=404))
    batch = Batch(**_batch_body())
    with pytest.raises(NotFoundError) as ei:
        client.batches.download_output(batch)
    assert ei.value.status_code == 404
    assert ei.value.request_id == "rid-404"
    client.close()


def test_download_output_content_error_no_raise_reports_status():
    captured = {}
    client = _make_client(
        _content_handler(captured, status=404),
        raise_on_error=False,
    )
    batch = Batch(**_batch_body())
    content = client.batches.download_output(batch)
    assert content.is_error is True
    assert content.status_code == 404
    assert b"file does not exist" in bytes(content)
    client.close()


def test_download_output_from_id_get_error_raises():
    def handler(request):
        return httpx.Response(
            404,
            json={"error": {"message": "batch does not exist"}},
            headers={"x-request-id": "rid-b404"},
        )

    client = _make_client(handler)
    with pytest.raises(NotFoundError) as ei:
        client.batches.download_output("missing-batch")
    assert ei.value.status_code == 404
    client.close()


def test_download_output_from_id_get_error_no_raise_returns_none():
    def handler(request):
        return httpx.Response(
            404,
            json={"error": {"message": "batch does not exist"}},
            headers={"x-request-id": "rid-b404"},
        )

    client = _make_client(handler, raise_on_error=False)
    # The error envelope carries no output_file_id -> None (caller inspects
    # the failure via batches.get under raise_on_error=False).
    assert client.batches.download_output("missing-batch") is None
    client.close()


# --------------------------------------------------------------------------
# input file reuse
# --------------------------------------------------------------------------


def test_input_file_id_reused_across_creates():
    bodies = []

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/batches"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_batch_body(status="validating"))

    client = _make_client(handler)
    first = client.batches.create(
        input_file_id=_INPUT_FILE_ID,
        endpoint=_ENDPOINT,
    )
    second = client.batches.create(
        input_file_id=_INPUT_FILE_ID,
        endpoint=_ENDPOINT,
    )
    # Same uploaded id drives both creates -- no re-upload between them.
    assert bodies[0]["input_file_id"] == _INPUT_FILE_ID
    assert bodies[1]["input_file_id"] == _INPUT_FILE_ID
    assert first.input_file_id == second.input_file_id == _INPUT_FILE_ID
    client.close()


def test_input_reuse_discover_via_files_list():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["purpose"] = request.url.params.get("purpose")
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": _INPUT_FILE_ID,
                        "object": "file",
                        "bytes": 120,
                        "created_at": 1730000000,
                        "filename": "input.jsonl",
                        "purpose": "batch",
                    },
                ],
                "has_more": False,
            },
        )

    client = _make_client(handler)
    page = client.batches.files.list(purpose="batch")
    assert captured["path"] == "/api/v1/files"
    assert captured["purpose"] == "batch"
    assert isinstance(page, BatchFileList)
    assert [f.id for f in page.items] == [_INPUT_FILE_ID]
    client.close()


# --------------------------------------------------------------------------
# async twins
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_download_output_from_batch_object():
    captured = {}
    client = _make_async_client(_content_handler(captured))
    batch = Batch(**_batch_body())
    content = await client.batches.download_output(batch)
    assert captured["path"] == f"/api/v1/files/{_OUTPUT_FILE_ID}/content"
    assert captured["sdk"].endswith("/batches2")
    assert bytes(content) == _OUTPUT_JSONL
    await client.aclose()


@pytest.mark.asyncio
async def test_async_download_output_from_batch_id_fetches_first():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == f"/api/v1/batches/{_BATCH_ID}":
            return httpx.Response(200, json=_batch_body())
        if request.url.path == f"/api/v1/files/{_OUTPUT_FILE_ID}/content":
            return httpx.Response(200, content=_OUTPUT_JSONL)
        raise AssertionError(f"unexpected path {request.url.path}")

    client = _make_async_client(handler)
    content = await client.batches.download_output(_BATCH_ID)
    assert calls == [
        ("GET", f"/api/v1/batches/{_BATCH_ID}"),
        ("GET", f"/api/v1/files/{_OUTPUT_FILE_ID}/content"),
    ]
    assert bytes(content) == _OUTPUT_JSONL
    await client.aclose()


@pytest.mark.asyncio
async def test_async_download_error_none_when_no_errors():
    def handler(request):
        raise AssertionError("no network call expected when no error file")

    client = _make_async_client(handler)
    batch = Batch(**_batch_body(error_file_id=None))
    assert await client.batches.download_error(batch) is None
    await client.aclose()


@pytest.mark.asyncio
async def test_async_download_output_content_error_raises():
    captured = {}
    client = _make_async_client(_content_handler(captured, status=404))
    batch = Batch(**_batch_body())
    with pytest.raises(NotFoundError) as ei:
        await client.batches.download_output(batch)
    assert ei.value.status_code == 404
    await client.aclose()