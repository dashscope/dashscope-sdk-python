# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for APIResponse + raw response access (T5 pilot)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict

import httpx
import pytest
from pydantic import BaseModel

from dashscope import AsyncClient, Client
from dashscope.errors import DashScopeError
from dashscope.resources._jobs import JobList
from dashscope.resources.embeddings import EmbeddingResponse
from dashscope.resources.files import FileList
from dashscope.resources.models import ModelList
from dashscope.transport import APIResponse
from dashscope.types.common import ResponseBase

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


class ProbeA(ResponseBase):
    """Plain probe model for parse() memoization tests."""


class ProbeB(ResponseBase):
    """A second, distinct type so per-type caching is observable."""


def _models_body(total: int = 1) -> Dict[str, Any]:
    return {
        "request_id": "body-id",
        "output": {
            "models": [{"model": f"m{i}"} for i in range(total)],
            "page_no": 1,
            "page_size": 10,
            "total": total,
        },
    }


def _models_handler(body=None, headers=None):
    payload = body if body is not None else _models_body()

    def handler(request):
        return httpx.Response(200, json=payload, headers=headers or {})

    return handler


def _embeddings_handler():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "emb-id",
                "output": {
                    "embeddings": [{"index": 0, "embedding": [0.1, 0.2]}],
                },
                "usage": {"total_tokens": 2},
            },
            headers={"x-request-id": "emb-header-id"},
        )

    return handler


# --- raw access -----------------------------------------------------------


def test_raw_returns_apiresponse_and_skips_error_check():
    error_body = {
        "code": "InvalidApiKey",
        "message": "bad key",
        "request_id": "e",
    }
    client = _make_client(
        _models_handler(body=error_body, headers={"x-request-id": "e"}),
    )

    resp = client.models.list(raw=True)
    # raw=True returns before check_response, so the business error is not raised.
    assert isinstance(resp, APIResponse)
    assert resp.status_code == 200
    assert resp.request_id == "e"
    assert resp.json()["code"] == "InvalidApiKey"


def test_non_raw_raises_on_business_error():
    error_body = {
        "code": "InvalidApiKey",
        "message": "bad key",
        "request_id": "e",
    }
    client = _make_client(_models_handler(body=error_body))

    with pytest.raises(DashScopeError):
        client.models.list()


def test_raw_gives_access_to_http_response_and_elapsed():
    client = _make_client(_models_handler())
    resp = client.models.list(raw=True)
    assert isinstance(resp, APIResponse)
    assert isinstance(resp.http_response, httpx.Response)
    # elapsed does not raise even on a mock transport that never set it.
    assert resp.elapsed is None or isinstance(resp.elapsed, timedelta)
    assert isinstance(resp.headers, dict)


# --- request_id -----------------------------------------------------------


def test_request_id_prefers_header():
    body = {"request_id": "body-id", "output": {"models": [], "total": 0}}
    client = _make_client(
        _models_handler(body=body, headers={"x-request-id": "header-id"}),
    )
    resp = client.models.list(raw=True)
    assert resp.request_id == "header-id"


def test_request_id_body_fallback():
    body = {"request_id": "body-id", "output": {"models": [], "total": 0}}
    client = _make_client(_models_handler(body=body))
    resp = client.models.list(raw=True)
    assert resp.request_id == "body-id"


def test_request_id_output_fallback():
    body = {"output": {"request_id": "out-id", "models": [], "total": 0}}
    client = _make_client(_models_handler(body=body))
    resp = client.models.list(raw=True)
    assert resp.request_id == "out-id"


# --- parse() --------------------------------------------------------------


def test_parse_none_returns_body():
    client = _make_client(_models_handler())
    resp = client.models.list(raw=True)
    assert resp.parse() == _models_body()


def test_parse_is_memoized_per_type():
    client = _make_client(_models_handler())
    resp = client.models.list(raw=True)

    first = resp.parse(ProbeA)
    second = resp.parse(ProbeA)
    assert first is second

    other = resp.parse(ProbeB)
    assert other is not first
    assert isinstance(other, ProbeB)
    # Each type stays independently cached.
    assert resp.parse(ProbeA) is first
    assert resp.parse(ProbeB) is other


def test_parse_attaches_headers():
    client = _make_client(
        _models_handler(headers={"x-request-id": "h", "x-custom": "v"}),
    )
    resp = client.models.list(raw=True)
    parsed = resp.parse(ProbeA)
    # with_headers copies the raw response headers onto the parsed model.
    assert parsed.headers["x-custom"] == "v"
    assert parsed.headers["x-request-id"] == "h"
    # The model's request_id comes from the body, not the header.
    assert parsed.request_id == "body-id"


# --- parse() unwrapping ---------------------------------------------------


def _api_response(body, headers=None):
    """An ``APIResponse`` over a body, without going through a client."""
    return APIResponse(
        httpx.Response(200, json=body, headers=headers or {}),
    )


def test_parse_lifts_the_output_envelope_into_the_model():
    # Public models describe the payload, not the wrapper. Without the lift
    # validation still succeeds -- every field has a default -- and the
    # caller gets a silently empty page.
    client = _make_client(_models_handler())
    page = client.models.list(raw=True).parse(ModelList)
    assert [m.model for m in page] == ["m0"]
    assert page.request_id == "body-id"


def test_parse_reads_the_wire_total_as_total_count():
    # The wire spells it ``total``; without the alias the page reads a zero
    # total and has_next_page() stops traversal after the first page.
    resp = _api_response(
        {
            "request_id": "r",
            "output": {
                "models": [{"model": "m0"}, {"model": "m1"}],
                "page_no": 1,
                "page_size": 2,
                "total": 5,
            },
        },
    )
    page = resp.parse(ModelList)
    assert page.total_count == 5
    assert page.has_next_page() is True


def test_parse_lifts_the_data_envelope_too():
    resp = _api_response(
        {
            "request_id": "f-id",
            "data": {
                "files": [{"file_id": "f1"}],
                "page_no": 1,
                "page_size": 10,
                "total": 1,
            },
        },
    )
    page = resp.parse(FileList)
    assert [f.file_id for f in page] == ["f1"]
    assert page.total_count == 1
    assert page.request_id == "f-id"


def test_parse_leaves_an_already_flat_body_alone():
    # jobs.list answers flat; there is nothing to lift, so the body is used
    # as-is rather than being mistaken for an empty envelope.
    resp = _api_response(
        {
            "request_id": "j-id",
            "tasks": [{"task_id": "t1", "task_status": "SUCCEEDED"}],
            "page_no": 1,
            "page_size": 10,
            "total": 1,
        },
    )
    page = resp.parse(JobList)
    assert [t.task_id for t in page] == ["t1"]
    assert page.total_count == 1


def test_parse_rejects_a_model_that_cannot_carry_headers():
    class Plain(BaseModel):
        x: int = 0

    resp = _api_response({"x": 1})
    # Failing here names the reason; failing inside with_headers would only
    # say that "Plain" has no field "headers".
    with pytest.raises(TypeError, match="ResponseBase"):
        resp.parse(Plain)


# --- non-raw paths unchanged ---------------------------------------------


def test_models_list_non_raw_returns_modellist_with_headers():
    client = _make_client(
        _models_handler(
            body={
                "request_id": "r",
                "output": {
                    "models": [{"model": "m0"}, {"model": "m1"}],
                    "page_no": 1,
                    "page_size": 2,
                    "total": 4,
                },
            },
            headers={"x-request-id": "r"},
        ),
    )
    page = client.models.list(page_no=1, page_size=2)
    assert isinstance(page, ModelList)
    assert not isinstance(page, APIResponse)
    assert page.headers["x-request-id"] == "r"
    assert page.request_id == "r"
    # Pagination still traverses after the migration.
    assert page.has_next_page() is True
    assert page.next_page() is not None


def test_embeddings_create_non_raw_returns_response():
    client = _make_client(_embeddings_handler())
    result = client.embeddings.create(model="text-embedding-v4", input="hi")
    assert isinstance(result, EmbeddingResponse)
    assert len(result.embeddings) == 1
    assert result.headers["x-request-id"] == "emb-header-id"


def test_embeddings_create_raw_returns_apiresponse():
    client = _make_client(_embeddings_handler())
    resp = client.embeddings.create(
        model="text-embedding-v4",
        input="hi",
        raw=True,
    )
    assert isinstance(resp, APIResponse)
    assert resp.request_id == "emb-header-id"
    assert resp.json()["output"]["embeddings"][0]["index"] == 0


# --- async ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_raw_and_parse():
    client = _make_async_client(
        _models_handler(headers={"x-request-id": "async-id"}),
    )
    resp = await client.models.list(raw=True)
    assert isinstance(resp, APIResponse)
    assert resp.request_id == "async-id"
    assert resp.parse(ProbeA) is resp.parse(ProbeA)


@pytest.mark.asyncio
async def test_async_non_raw_returns_modellist():
    client = _make_async_client(_models_handler())
    page = await client.models.list()
    assert isinstance(page, ModelList)
    assert [m.model for m in page] == ["m0"]


@pytest.mark.asyncio
async def test_async_raw_skips_error_check():
    error_body = {"code": "InvalidApiKey", "message": "bad", "request_id": "e"}
    client = _make_async_client(_models_handler(body=error_body))
    resp = await client.models.list(raw=True)
    assert isinstance(resp, APIResponse)
    assert resp.status_code == 200
