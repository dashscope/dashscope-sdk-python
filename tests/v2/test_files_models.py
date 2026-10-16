# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for files/models resources (M1 task 1.8)."""

from __future__ import annotations

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client
from dashscope.v2.errors import BadRequestError, NotFoundError

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


def test_files_upload(tmp_path):
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content.decode("utf-8", "replace")
        return httpx.Response(
            200,
            json={
                "request_id": "req-u",
                "data": {
                    "uploaded_files": [{"name": "t.jsonl", "file_id": "f-1"}],
                },
            },
        )

    file_path = tmp_path / "t.jsonl"
    file_path.write_text('{"a": 1}\n', encoding="utf-8")
    client = _make_client(handler)
    resp = client.files.upload(
        file_path=str(file_path),
        purpose="fine-tune",
        description="train",
    )
    assert captured["path"] == "/api/v1/files"
    assert captured["content_type"].startswith("multipart/form-data")
    assert 'name="purpose"' not in captured["body"]
    assert 'name="descriptions"' in captured["body"]
    assert '["train"]' not in captured["body"]
    assert "t.jsonl" in captured["body"]
    assert resp.file_id == "f-1"
    assert resp.name == "t.jsonl"


def test_files_upload_validation(tmp_path):
    client = _make_client(lambda r: httpx.Response(200, json={"data": {}}))
    with pytest.raises(FileNotFoundError):
        client.files.upload(file_path=str(tmp_path / "nope.jsonl"))

    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl"):
        client.files.upload(file_path=str(bad))

    resp = client.files.upload(file_path=str(bad), purpose="inference")
    assert resp is not None


def test_files_upload_retry_resends_full_body(tmp_path, monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(503, json={"message": "busy"})
        return httpx.Response(
            200,
            json={
                "request_id": "req-r",
                "data": {"uploaded_files": [{"file_id": "f-r"}]},
            },
        )

    from dashscope.v2.transport import http as transport_http

    monkeypatch.setattr(transport_http, "_sleep", lambda seconds: None)
    file_path = tmp_path / "t.jsonl"
    file_path.write_text('{"a": 1}\n', encoding="utf-8")
    client = _make_client(handler)
    resp = client.files.upload(file_path=str(file_path))
    assert resp.file_id == "f-r"
    assert len(bodies) == 2
    # Multipart boundaries are random per request; what matters is the
    # retry re-sends the full payload rather than a drained handle.
    for body in bodies:
        assert b'{"a": 1}' in body
        assert b"t.jsonl" in body


def test_files_list_get_delete():
    def handler(request):
        if request.method == "GET" and request.url.path == "/api/v1/files":
            assert request.url.params["page_no"] == "2"
            return httpx.Response(
                200,
                json={
                    "request_id": "r",
                    "data": {
                        "page_no": 2,
                        "page_size": 10,
                        "total": 1,
                        "files": [{"file_id": "f-1", "name": "t.jsonl"}],
                    },
                },
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": {"file_id": "f-1", "name": "t.jsonl", "size": 10},
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json={"request_id": "r"})
        raise AssertionError("unexpected request")

    client = _make_client(handler)
    listed = client.files.list(page_no=2)
    assert listed.total_count == 1
    assert listed.files[0].file_id == "f-1"
    got = client.files.get("f-1")
    assert got.size == 10
    deleted = client.files.delete("f-1")
    assert deleted.deleted is True


def test_files_get_missing_file_raises_not_found():
    # DashScope reports a missing file as a 400 InvalidParameter ("File not
    # found.") rather than a 404, so the catalog mapper yields BadRequestError.
    # get/delete reclassify it so ``except NotFoundError`` works, while keeping
    # the server's own status/code/request_id (never an invented 404).
    client = _make_client(
        lambda r: httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "File not found.",
                "request_id": "req-nf",
            },
        ),
    )
    with pytest.raises(NotFoundError) as exc:
        client.files.get("missing-id")
    err = exc.value
    assert err.code == "InvalidParameter"
    assert err.status_code == 400
    assert err.request_id == "req-nf"
    assert err.message == "File not found."


def test_files_delete_missing_file_raises_not_found():
    client = _make_client(
        lambda r: httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "File not found.",
            },
        ),
    )
    with pytest.raises(NotFoundError):
        client.files.delete("missing-id")


def test_files_missing_file_not_found_regardless_of_status():
    # InvalidParameter selects BadRequestError at any status (the wire code
    # outranks the status), so the reclassification cannot hinge on 400.
    for status in (400, 404):
        client = _make_client(
            lambda r, s=status: httpx.Response(
                s,
                json={
                    "code": "InvalidParameter",
                    "message": "File not found.",
                },
            ),
        )
        with pytest.raises(NotFoundError) as exc:
            client.files.get("missing-id")
        assert exc.value.status_code == status


def test_files_get_delete_empty_file_id_rejected_locally():
    # "" (or blank) would collapse the path to "/files/" and hit the list
    # endpoint; the call must fail locally without touching the transport.
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"data": {}})

    client = _make_client(handler)
    for file_id in ("", "   "):
        with pytest.raises(ValueError, match="file_id"):
            client.files.get(file_id)
        with pytest.raises(ValueError, match="file_id"):
            client.files.delete(file_id)
    assert calls == []


@pytest.mark.asyncio
async def test_async_files_get_delete_empty_file_id_rejected_locally():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"data": {}})

    client = _make_async_client(handler)
    for file_id in ("", "   "):
        with pytest.raises(ValueError, match="file_id"):
            await client.files.get(file_id)
        with pytest.raises(ValueError, match="file_id"):
            await client.files.delete(file_id)
    assert calls == []
    await client.aclose()


def test_files_get_other_bad_request_stays_bad_request():
    # Only a "not found"/"not exist" message is reclassified; any other
    # InvalidParameter (a genuinely malformed request) stays BadRequestError.
    client = _make_client(
        lambda r: httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "file_id contains illegal characters",
            },
        ),
    )
    with pytest.raises(BadRequestError) as exc:
        client.files.get("bad id")
    assert type(exc.value) is BadRequestError


def test_files_get_missing_file_no_raise_reports_envelope():
    # With raise_on_error=False nothing is raised, so there is no exception to
    # reclassify; the envelope keeps the gateway's own code.
    client = _make_client(
        lambda r: httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "File not found.",
                "request_id": "req-nf",
            },
        ),
        raise_on_error=False,
    )
    info = client.files.get("missing-id")
    assert info.is_error
    assert info.code == "InvalidParameter"
    assert info.request_id == "req-nf"


@pytest.mark.asyncio
async def test_async_files_get_missing_file_raises_not_found():
    client = _make_async_client(
        lambda r: httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "File not found.",
            },
        ),
    )
    with pytest.raises(NotFoundError):
        await client.files.get("missing-id")
    await client.aclose()


@pytest.mark.asyncio
async def test_async_files_delete_missing_file_raises_not_found():
    client = _make_async_client(
        lambda r: httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "File not found.",
            },
        ),
    )
    with pytest.raises(NotFoundError):
        await client.files.delete("missing-id")
    await client.aclose()


def test_models_list_with_provider_filter():
    # ``data``-wrapped pages stay supported for gateways that re-wrap
    # the native ``output`` envelope OpenAI-style.
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "request_id": "r",
                "data": {
                    "page_no": 1,
                    "page_size": 10,
                    "total": 3,
                    "models": [
                        {"model": "qwen-plus", "source": "dashscope"},
                        {"model": "qwen3-max", "source": "bailian"},
                        {"model": "qwen-turbo", "provider": "Community"},
                    ],
                },
            },
        )

    client = _make_client(handler)
    all_models = client.models.list()
    assert captured["params"] == {"page_no": "1", "page_size": "10"}
    assert len(all_models.models) == 3

    bailian = client.models.list(provider="Bailian")
    assert [m.model for m in bailian.models] == ["qwen3-max"]

    community = client.models.list(provider="community")
    assert [m.model for m in community.models] == ["qwen-turbo"]


def test_models_list_parses_native_output_envelope():
    # The native /api/v1/models envelope nests the page under
    # ``output``; parsing only ``data`` yielded an empty list with
    # default pagination against the real endpoint.
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-page",
                "output": {
                    "page_no": 1,
                    "page_size": 3,
                    "total": 8,
                    "models": [
                        {"model": "qwen-plus", "source": "dashscope"},
                        {"model": "qwen3-max", "source": "bailian"},
                        {"model": "qwen-turbo", "source": "dashscope"},
                    ],
                },
            },
        )

    client = _make_client(handler)
    page = client.models.list(page_no=1, page_size=3)
    assert page.page_no == 1
    assert page.page_size == 3
    assert page.total_count == 8
    assert [m.model for m in page.models] == [
        "qwen-plus",
        "qwen3-max",
        "qwen-turbo",
    ]
    assert page.request_id == "req-page"


def test_models_list_carries_raw_response_headers():
    # Uniform success-path contract: every HTTP resource mirrors the
    # gateway's response headers on ``resp.headers`` (1.x parity).
    def handler(request):
        return httpx.Response(
            200,
            json={"request_id": "req-h", "output": {"models": []}},
            headers={"x-request-id": "req-h", "x-gw-region": "cn-hz"},
        )

    client = _make_client(handler)
    page = client.models.list()
    assert page.headers["x-request-id"] == "req-h"
    assert page.headers["x-gw-region"] == "cn-hz"


def test_models_get():
    def handler(request):
        assert request.url.path == "/api/v1/models"
        assert request.url.params["model"] == "qwen-plus"
        assert request.url.params["page_no"] == "1"
        assert request.url.params["page_size"] == "1"
        return httpx.Response(
            200,
            json={
                "output": {
                    "models": [
                        {"model": "qwen-plus", "source": "dashscope"},
                    ],
                },
            },
        )

    client = _make_client(handler)
    model = client.models.get("qwen-plus")
    assert model.model == "qwen-plus"
    assert model.source == "dashscope"


def test_models_get_not_found():
    client = _make_client(
        lambda r: httpx.Response(
            200,
            json={"output": {"models": []}},
            headers={"x-request-id": "req-none"},
        ),
    )
    with pytest.raises(NotFoundError, match="not found") as exc:
        client.models.get("nope")
    error = exc.value
    assert error.code == "Model.NotExist"
    # The list call itself succeeded; the model was simply absent. So the
    # status is the one the server sent, not an invented 404.
    assert error.status_code == 200
    assert error.request_id == "req-none"


@pytest.mark.asyncio
async def test_async_files_models():
    def handler(request):
        if request.url.path == "/api/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "models": [
                            {"model": "qwen-plus", "source": "dashscope"},
                        ],
                    },
                },
            )
        if request.url.path == "/api/v1/files":
            return httpx.Response(
                200,
                json={
                    "data": {"files": [{"file_id": "f-9"}], "total_count": 1},
                },
            )
        return httpx.Response(200, json={"data": {"file_id": "f-9"}})

    client = _make_async_client(handler)
    models = await client.models.list()
    assert models.models[0].model == "qwen-plus"
    files = await client.files.list()
    assert files.files[0].file_id == "f-9"
    one = await client.files.get("f-9")
    assert one.file_id == "f-9"
    await client.aclose()


def test_business_code_reaches_single_item_responses():
    # DashScope can fail with HTTP 200 plus a business code. These two
    # parsers build their result from an item nested under data/output, so
    # the envelope has to be threaded on separately for is_error to mean the
    # same thing here as it does on chat and apps.
    def handler(request):
        if request.url.path == "/api/v1/models":
            return httpx.Response(
                200,
                json={
                    "request_id": "req-m",
                    "code": "Throttling",
                    "message": "slow down",
                    "output": {"models": []},
                },
            )
        return httpx.Response(
            200,
            json={
                "request_id": "req-f",
                "code": "Throttling",
                "message": "slow down",
                "data": {"file_id": "f-1"},
            },
        )

    client = _make_client(handler, raise_on_error=False)
    model = client.models.get("qwen-plus")
    assert model.is_error
    assert model.code == "Throttling"
    assert model.request_id == "req-m"

    info = client.files.get("f-1")
    assert info.is_error
    assert info.code == "Throttling"
    assert info.request_id == "req-f"


def test_nested_list_items_carry_no_envelope():
    def handler(request):
        assert request.url.path == "/api/v1/files"
        return httpx.Response(
            200,
            json={
                "request_id": "req-l",
                "data": {"files": [{"file_id": "f-1"}], "total": 1},
            },
        )

    page = _make_client(handler).files.list()
    assert page.request_id == "req-l"
    assert not page.is_error
    # The envelope belongs to the response, not to each item inside it.
    assert page.files[0].request_id is None
    assert not page.files[0].is_error
