# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.fine_tunes.compress (D1: model compression)."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.errors import DashScopeError
from dashscope.resources.fine_tunes_compress import (
    CompressJob,
    CompressJobList,
    CompressTemplateList,
)

BASE_URL = "https://api.test.local/api/v1"
JOBS = "/api/v1/fine-tunes/compress/jobs"
TEMPLATES = "/api/v1/fine-tunes/compress/templates"


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


def _job_payload(status, job_id="cj-1"):
    return {
        "request_id": "req-1",
        "output": {
            "job_id": job_id,
            "job_name": "quant-run",
            "status": status,
            "model": "qwen3.5-flash-ft",
            "base_model": "qwen3.5-flash",
            "template_id": "tpl-1",
            "template_name": "int4",
            "training_type": "quantization",
            "compress_type": "quantization",
            "hyper_parameters": {"bits": 4},
            "custom_calibration_file_ids": ["file-abc"],
            "quantized_output": "qwen3.5-flash-ft-q4-cj-1",
            "group": "quantization",
            "create_time": "2026-11-17 10:00:00",
        },
    }


# --- templates.list -------------------------------------------------------


def _templates_payload():
    return {
        "request_id": "req-t",
        "output": {
            "base_models": ["qwen3.5-flash-2026-02-23"],
            "custom_models": [
                {
                    "model": "qwen3.5-flash-2026-02-23-ft-abc",
                    "model_name": "我的SFT调优模型",
                    "base_model": "qwen3.5-flash-2026-02-23",
                    "templates": [
                        {
                            "template_id": "quant-flash-nvfp4-mlp-nomtp",
                            "template_name": "W4A4 NVFP4高性能压缩",
                            "description": "更低比特压缩兼顾高精度与高性能。",
                            "compress_type": "quantization",
                            "hyper_parameters": [
                                {
                                    "name": "calib_input",
                                    "type": "string",
                                    "display_name": "校准输入",
                                    "support_values": ["true"],
                                    "defaultValue": "true",
                                    "required": False,
                                },
                            ],
                        },
                    ],
                },
            ],
        },
    }


def test_templates_list_path_params_and_parse():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=_templates_payload())

    client = _make_client(handler)
    catalog = client.fine_tunes.compress.templates.list(
        model="qwen3.5-flash-2026-02-23",
        lang="zh-CN",
    )
    assert captured["method"] == "GET"
    assert captured["path"] == TEMPLATES
    # The catalog endpoint takes only model/lang, no pagination.
    assert captured["params"] == {
        "model": "qwen3.5-flash-2026-02-23",
        "lang": "zh-CN",
    }
    assert isinstance(catalog, CompressTemplateList)
    assert catalog.request_id == "req-t"
    assert catalog.base_models == ["qwen3.5-flash-2026-02-23"]
    custom = catalog.custom_models[0]
    assert custom.model == "qwen3.5-flash-2026-02-23-ft-abc"
    assert custom.model_name == "我的SFT调优模型"
    assert custom.base_model == "qwen3.5-flash-2026-02-23"
    tpl = custom.templates[0]
    assert tpl.template_id == "quant-flash-nvfp4-mlp-nomtp"
    assert tpl.description == "更低比特压缩兼顾高精度与高性能。"
    assert tpl.compress_type == "quantization"
    hp = tpl.hyper_parameters[0]
    assert hp.name == "calib_input"
    assert hp.type == "string"
    assert hp.support_values == ["true"]
    assert hp.required is False


def test_templates_list_omits_unset_params_and_flattens():
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        payload = _templates_payload()
        # A second custom model with two templates exercises flattening.
        payload["output"]["custom_models"].append(
            {
                "model": "m2",
                "base_model": "qwen3.5-flash-2026-02-23",
                "templates": [
                    {"template_id": "t2"},
                    {"template_id": "t3"},
                ],
            },
        )
        return httpx.Response(200, json=payload)

    client = _make_client(handler)
    catalog = client.fine_tunes.compress.templates.list()
    assert captured["params"] == {}
    assert [t.template_id for t in catalog.templates] == [
        "quant-flash-nvfp4-mlp-nomtp",
        "t2",
        "t3",
    ]


# --- jobs.create ----------------------------------------------------------


def test_create_builds_wire_body():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_job_payload("PENDING"))

    client = _make_client(handler)
    job = client.fine_tunes.compress.jobs.create(
        model="qwen3.5-flash-ft",
        template_id="tpl-1",
        job_name="quant-run",
        job_description="int4 quantization",
        hyper_parameters={"bits": 4},
        custom_calibration_file_ids="file-abc",
        output_model_suffix="q4",
    )
    assert captured["method"] == "POST"
    assert captured["path"] == JOBS
    body = captured["body"]
    assert body["model"] == "qwen3.5-flash-ft"
    assert body["template_id"] == "tpl-1"
    assert body["job_name"] == "quant-run"
    assert body["job_description"] == "int4 quantization"
    assert body["hyper_parameters"] == {"bits": 4}
    # A bare string is normalized to a one-element list.
    assert body["custom_calibration_file_ids"] == ["file-abc"]
    assert body["output_model_suffix"] == "q4"
    assert job.job_id == "cj-1"
    assert job.status == "PENDING"
    assert job.request_id == "req-1"
    assert job.quantized_output == "qwen3.5-flash-ft-q4-cj-1"
    assert job.compress_type == "quantization"


def test_create_minimal_body_omits_unset_optionals():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_job_payload("PENDING"))

    client = _make_client(handler)
    client.fine_tunes.compress.jobs.create(
        model="qwen3.5-flash-ft",
        template_id="tpl-1",
    )
    assert captured["body"] == {
        "model": "qwen3.5-flash-ft",
        "template_id": "tpl-1",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_model_suffix": "BAD!"},
        {"output_model_suffix": "toolongsuffix"},
        {"output_model_suffix": "Upper"},
        {"job_name": "x" * 51},
        {"job_description": "x" * 201},
    ],
)
def test_create_rejects_invalid_inputs(kwargs):
    called = []

    def handler(request):
        called.append(request)
        return httpx.Response(200, json=_job_payload("PENDING"))

    client = _make_client(handler)
    with pytest.raises(DashScopeError):
        client.fine_tunes.compress.jobs.create(
            model="qwen3.5-flash-ft",
            template_id="tpl-1",
            **kwargs,
        )
    # Validation happens before any request is sent.
    assert called == []


def test_create_accepts_valid_suffix_boundary():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_job_payload("PENDING"))

    client = _make_client(handler)
    client.fine_tunes.compress.jobs.create(
        model="m",
        template_id="tpl-1",
        output_model_suffix="abcd1234",  # exactly 8, lowercase + digits
    )
    assert captured["body"]["output_model_suffix"] == "abcd1234"


# --- jobs.get / list / cancel / delete / logs paths -----------------------


def test_job_lifecycle_paths():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path.endswith("/logs"):
            assert dict(request.url.params) == {"offset": "2", "line": "50"}
            return httpx.Response(
                200,
                json={"request_id": "req-l", "output": {"logs": ["a", "b"]}},
            )
        if request.method == "GET" and request.url.path.endswith("cj-1"):
            return httpx.Response(200, json=_job_payload("RUNNING"))
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "request_id": "req-2",
                    "output": {
                        "page_no": 1,
                        "page_size": 10,
                        "total": 1,
                        "jobs": [_job_payload("SUCCEEDED")["output"]],
                    },
                },
            )
        return httpx.Response(
            200,
            json={"request_id": "req-3", "output": {"status": "done"}},
        )

    client = _make_client(handler)
    assert client.fine_tunes.compress.jobs.get("cj-1").status == "RUNNING"
    listed = client.fine_tunes.compress.jobs.list()
    assert isinstance(listed, CompressJobList)
    assert listed.total_count == 1
    assert listed.jobs[0].job_id == "cj-1"
    logs = client.fine_tunes.compress.jobs.logs("cj-1", offset=2, lines=50)
    assert logs.content == "a\nb"
    assert logs.job_id == "cj-1"
    assert client.fine_tunes.compress.jobs.cancel("cj-1").status == "done"
    assert client.fine_tunes.compress.jobs.delete("cj-1").status == "done"
    assert seen == [
        ("GET", f"{JOBS}/cj-1"),
        ("GET", JOBS),
        ("GET", f"{JOBS}/cj-1/logs"),
        ("POST", f"{JOBS}/cj-1/cancel"),
        ("DELETE", f"{JOBS}/cj-1"),
    ]


def test_jobs_list_filters_pass_through():
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={"request_id": "r", "output": {"jobs": [], "total": 0}},
        )

    client = _make_client(handler)
    client.fine_tunes.compress.jobs.list(
        status="SUCCEEDED",
        model="qwen3.5-flash-ft",
        template_id="tpl-1",
        quant_spec="int4",
        quant_method="gptq",
        start_time="2026-11-01",
        end_time="2026-11-17",
        job_name="quant-run",
        job_id="cj-1",
        search_key="kw",
        select_key="sel",
        sort_by="create_time",
        sort_order="desc",
        page_no=2,
        page_size=50,
    )
    assert captured["params"] == {
        "page_no": "2",
        "page_size": "50",
        "status": "SUCCEEDED",
        "model": "qwen3.5-flash-ft",
        "template_id": "tpl-1",
        "quant_spec": "int4",
        "quant_method": "gptq",
        "start_time": "2026-11-01",
        "end_time": "2026-11-17",
        "job_name": "quant-run",
        "job_id": "cj-1",
        "search_key": "kw",
        "select_key": "sel",
        "sort_by": "create_time",
        "sort_order": "desc",
    }


def test_jobs_list_pagination_preserves_filters():
    catalog = [{"job_id": f"cj{i}"} for i in range(5)]
    seen = []

    def handler(request):
        params = request.url.params
        seen.append((int(params.get("page_no", 1)), params.get("status")))
        page_no = int(params.get("page_no", 1))
        page_size = int(params.get("page_size", 10))
        start = (page_no - 1) * page_size
        return httpx.Response(
            200,
            json={
                "request_id": "r",
                "output": {
                    "jobs": catalog[start : start + page_size],
                    "page_no": page_no,
                    "page_size": page_size,
                    "total": len(catalog),
                },
            },
        )

    client = _make_client(handler)
    page = client.fine_tunes.compress.jobs.list(
        page_no=1,
        page_size=2,
        status="SUCCEEDED",
    )
    collected = [j.job_id for p in page.iter_pages() for j in p]
    assert collected == ["cj0", "cj1", "cj2", "cj3", "cj4"]
    assert seen == [(1, "SUCCEEDED"), (2, "SUCCEEDED"), (3, "SUCCEEDED")]


def test_job_parses_nested_error_and_usage():
    payload = _job_payload("FAILED")
    payload["output"]["error"] = {"code": "InternalError", "message": "boom"}
    payload["output"]["usage"] = 120
    client = _make_client(lambda r: httpx.Response(200, json=payload))
    job = client.fine_tunes.compress.jobs.get("cj-1")
    assert job.error.code == "InternalError"
    assert job.error.message == "boom"
    assert job.usage == 120
    assert job.finished is True
    assert job.succeeded is False


# --- jobs.wait ------------------------------------------------------------


def test_wait_polls_until_terminal():
    statuses = iter(["QUEUING", "RUNNING", "SUCCEEDED"])
    delays = []

    def handler(request):
        return httpx.Response(200, json=_job_payload(next(statuses)))

    client = _make_client(handler)
    job = client.fine_tunes.compress.jobs.wait("cj-1", _sleep=delays.append)
    assert job.status == "SUCCEEDED"
    assert delays == [1.0, 2.0]


def test_wait_failed_job_raises_with_nested_error():
    payload = _job_payload("FAILED")
    payload["output"]["error"] = {
        "code": "InternalError",
        "message": "quant crashed",
    }

    def handler(request):
        return httpx.Response(
            200,
            json=payload,
            headers={"x-request-id": "req-h"},
        )

    client = _make_client(handler)
    with pytest.raises(DashScopeError) as exc:
        client.fine_tunes.compress.jobs.wait("cj-1", _sleep=lambda s: None)
    error = exc.value
    assert error.code == "InternalError"
    assert error.message == "quant crashed"
    assert error.headers["x-request-id"] == "req-h"
    assert "headers" not in error.body
    assert error.body["status"] == "FAILED"


def test_wait_canceled_job_raises():
    client = _make_client(
        lambda r: httpx.Response(200, json=_job_payload("CANCELED")),
    )
    with pytest.raises(DashScopeError, match="canceled"):
        client.fine_tunes.compress.jobs.wait("cj-1", _sleep=lambda s: None)


def test_wait_returns_error_envelope_in_compat_mode():
    def handler(request):
        return httpx.Response(
            401,
            json={
                "request_id": "req-x",
                "code": "InvalidApiKey",
                "message": "bad key",
            },
        )

    client = _make_client(handler, raise_on_error=False)
    job = client.fine_tunes.compress.jobs.wait("cj-1", _sleep=lambda s: None)
    assert job.is_error is True
    assert job.code == "InvalidApiKey"
    assert job.request_id == "req-x"


def test_wait_accepts_job_object():
    client = _make_client(
        lambda r: httpx.Response(200, json=_job_payload("SUCCEEDED")),
    )
    job = CompressJob(job_id="cj-1", status="RUNNING")
    done = client.fine_tunes.compress.jobs.wait(job, _sleep=lambda s: None)
    assert done.status == "SUCCEEDED"


# --- raw escape hatch -----------------------------------------------------


def test_raw_returns_api_response():
    client = _make_client(
        lambda r: httpx.Response(200, json=_job_payload("RUNNING")),
    )
    resp = client.fine_tunes.compress.jobs.get("cj-1", raw=True)
    assert resp.http_response.status_code == 200
    assert resp.json()["output"]["job_id"] == "cj-1"


# --- async symmetry -------------------------------------------------------


@pytest.mark.asyncio
async def test_async_create_get_and_templates():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.url.path == TEMPLATES:
            return httpx.Response(200, json=_templates_payload())
        if request.method == "POST":
            return httpx.Response(200, json=_job_payload("PENDING"))
        return httpx.Response(200, json=_job_payload("SUCCEEDED"))

    client = _make_async_client(handler)
    job = await client.fine_tunes.compress.jobs.create(
        model="qwen3.5-flash-ft",
        template_id="tpl-1",
        output_model_suffix="q4",
    )
    assert job.job_id == "cj-1"
    fetched = await client.fine_tunes.compress.jobs.get("cj-1")
    assert fetched.status == "SUCCEEDED"
    catalog = await client.fine_tunes.compress.templates.list()
    assert catalog.custom_models[0].templates[0].template_id == (
        "quant-flash-nvfp4-mlp-nomtp"
    )
    assert seen == [
        ("POST", JOBS),
        ("GET", f"{JOBS}/cj-1"),
        ("GET", TEMPLATES),
    ]
    await client.aclose()


@pytest.mark.asyncio
async def test_async_list_pagination_preserves_filters():
    catalog = [{"job_id": f"cj{i}"} for i in range(5)]
    seen = []

    def handler(request):
        params = request.url.params
        seen.append((int(params.get("page_no", 1)), params.get("status")))
        page_no = int(params.get("page_no", 1))
        page_size = int(params.get("page_size", 10))
        start = (page_no - 1) * page_size
        return httpx.Response(
            200,
            json={
                "request_id": "r",
                "output": {
                    "jobs": catalog[start : start + page_size],
                    "page_no": page_no,
                    "page_size": page_size,
                    "total": len(catalog),
                },
            },
        )

    client = _make_async_client(handler)
    page = await client.fine_tunes.compress.jobs.list(
        page_no=1,
        page_size=2,
        status="SUCCEEDED",
    )
    collected = [j.job_id async for p in page.iter_pages_async() for j in p]
    assert collected == ["cj0", "cj1", "cj2", "cj3", "cj4"]
    assert seen == [(1, "SUCCEEDED"), (2, "SUCCEEDED"), (3, "SUCCEEDED")]
    await client.aclose()


@pytest.mark.asyncio
async def test_async_wait_polls_until_terminal():
    statuses = iter(["RUNNING", "SUCCEEDED"])
    delays = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    def handler(request):
        return httpx.Response(200, json=_job_payload(next(statuses)))

    client = _make_async_client(handler)
    job = await client.fine_tunes.compress.jobs.wait(
        "cj-1",
        _sleep=fake_sleep,
    )
    assert job.status == "SUCCEEDED"
    assert delays == [1.0]
    await client.aclose()


# --- retry policy + job id guards ------------------------------------------


def test_create_default_no_retry():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(503, json={"message": "busy"})

    client = _make_client(handler)
    with pytest.raises(DashScopeError):
        client.fine_tunes.compress.jobs.create(
            model="qwen3.5-flash-ft",
            template_id="tpl-1",
        )
    # No idempotency key on the wire -> create must not auto-retry.
    assert len(attempts) == 1
    client.close()


@pytest.mark.asyncio
async def test_async_create_default_no_retry():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(503, json={"message": "busy"})

    client = _make_async_client(handler)
    with pytest.raises(DashScopeError):
        await client.fine_tunes.compress.jobs.create(
            model="qwen3.5-flash-ft",
            template_id="tpl-1",
        )
    assert len(attempts) == 1
    await client.aclose()


@pytest.mark.parametrize("op", ["get", "cancel", "delete", "logs"])
def test_empty_job_id_rejected_before_wire(op):
    def handler(request):
        raise AssertionError("empty job id must not reach the wire")

    client = _make_client(handler)
    with pytest.raises(DashScopeError, match="job id required"):
        getattr(client.fine_tunes.compress.jobs, op)("")
    client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["get", "cancel", "delete", "logs"])
async def test_async_empty_job_id_rejected_before_wire(op):
    def handler(request):
        raise AssertionError("empty job id must not reach the wire")

    client = _make_async_client(handler)
    with pytest.raises(DashScopeError, match="job id required"):
        await getattr(client.fine_tunes.compress.jobs, op)("")
    await client.aclose()


@pytest.mark.parametrize("op", ["get", "cancel", "delete", "logs"])
def test_whitespace_job_id_rejected_before_wire(op):
    def handler(request):
        raise AssertionError("whitespace job id must not reach the wire")

    client = _make_client(handler)
    with pytest.raises(DashScopeError, match="job id required"):
        getattr(client.fine_tunes.compress.jobs, op)("   ")
    client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["get", "cancel", "delete", "logs"])
async def test_async_whitespace_job_id_rejected_before_wire(op):
    def handler(request):
        raise AssertionError("whitespace job id must not reach the wire")

    client = _make_async_client(handler)
    with pytest.raises(DashScopeError, match="job id required"):
        await getattr(client.fine_tunes.compress.jobs, op)("   ")
    await client.aclose()


_TRICKY_JOB = "cj/a?b#c"
_ENCODED_JOB = b"/api/v1/fine-tunes/compress/jobs/cj%2Fa%3Fb%23c"
_JOB_SUFFIX = {
    "get": b"",
    "cancel": b"/cancel",
    "delete": b"",
    "logs": b"/logs?offset=0&line=100",
}


@pytest.mark.parametrize("op", ["get", "cancel", "delete", "logs"])
def test_job_id_percent_encoded_in_path(op):
    seen = []

    def handler(request):
        seen.append(request.url.raw_path)
        return httpx.Response(200, json=_job_payload("RUNNING"))

    client = _make_client(handler)
    getattr(client.fine_tunes.compress.jobs, op)(_TRICKY_JOB)
    # '/', '?' and '#' stay inside one path segment instead of retargeting
    # the request to another path or query.
    assert seen == [_ENCODED_JOB + _JOB_SUFFIX[op]]
    client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["get", "cancel", "delete", "logs"])
async def test_async_job_id_percent_encoded_in_path(op):
    seen = []

    def handler(request):
        seen.append(request.url.raw_path)
        return httpx.Response(200, json=_job_payload("RUNNING"))

    client = _make_async_client(handler)
    await getattr(client.fine_tunes.compress.jobs, op)(_TRICKY_JOB)
    assert seen == [_ENCODED_JOB + _JOB_SUFFIX[op]]
    await client.aclose()


def test_wait_does_not_double_encode_job_id():
    seen = []
    statuses = iter(["RUNNING", "SUCCEEDED"])

    def handler(request):
        seen.append(request.url.raw_path)
        return httpx.Response(200, json=_job_payload(next(statuses)))

    client = _make_client(handler)
    job = client.fine_tunes.compress.jobs.wait(
        _TRICKY_JOB,
        _sleep=lambda _: None,
    )
    assert job.status == "SUCCEEDED"
    # wait polls through get(), which encodes once per call.
    assert seen == [_ENCODED_JOB, _ENCODED_JOB]
    client.close()


@pytest.mark.asyncio
async def test_async_wait_does_not_double_encode_job_id():
    seen = []
    statuses = iter(["RUNNING", "SUCCEEDED"])

    def handler(request):
        seen.append(request.url.raw_path)
        return httpx.Response(200, json=_job_payload(next(statuses)))

    async def no_sleep(_):
        return None

    client = _make_async_client(handler)
    job = await client.fine_tunes.compress.jobs.wait(
        _TRICKY_JOB,
        _sleep=no_sleep,
    )
    assert job.status == "SUCCEEDED"
    assert seen == [_ENCODED_JOB, _ENCODED_JOB]
    await client.aclose()


def test_logs_keeps_raw_job_id_in_result():
    def handler(request):
        return httpx.Response(
            200,
            json={"request_id": "req-l", "output": {"logs": ["a", "b"]}},
        )

    client = _make_client(handler)
    logs = client.fine_tunes.compress.jobs.logs(_TRICKY_JOB)
    # The parsed envelope echoes the caller's id, not the encoded form.
    assert logs.job_id == _TRICKY_JOB
    assert logs.content == "a\nb"
    client.close()


@pytest.mark.asyncio
async def test_async_logs_keeps_raw_job_id_in_result():
    def handler(request):
        return httpx.Response(
            200,
            json={"request_id": "req-l", "output": {"logs": ["a", "b"]}},
        )

    client = _make_async_client(handler)
    logs = await client.fine_tunes.compress.jobs.logs(_TRICKY_JOB)
    assert logs.job_id == _TRICKY_JOB
    assert logs.content == "a\nb"
    await client.aclose()
