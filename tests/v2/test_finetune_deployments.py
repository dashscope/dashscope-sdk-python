# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.fine_tunes and client.deployments (M3 3.6)."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client
from dashscope.v2.errors import (
    BadRequestError,
    DashScopeError,
    RateLimitError,
)
from dashscope.v2.resources.fine_tunes import FineTune

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


def _job_payload(status, job_id="ft-1"):
    return {
        "request_id": "req-1",
        "output": {
            "job_id": job_id,
            "status": status,
            "model": "qwen-turbo",
            "base_model": "qwen-turbo",
            "training_file_ids": ["file-1"],
            "validation_file_ids": [],
            "hyper_parameters": {"n_epochs": 2},
            "training_type": "sft",
        },
    }


def test_create_builds_wire_body():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_job_payload("PENDING"))

    client = _make_client(handler)
    job = client.fine_tunes.create(
        model="qwen-turbo",
        training_file_ids="file-1",
        mode="sft",
        hyper_parameters={"n_epochs": 2},
        finetuned_model="my-model",
    )
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v1/fine-tunes"
    body = captured["body"]
    assert body["model"] == "qwen-turbo"
    assert body["training_file_ids"] == ["file-1"]
    assert body["validation_file_ids"] is None
    assert body["training_type"] == "sft"
    assert body["hyper_parameters"] == {"n_epochs": 2}
    assert body["model_name"] == "my-model"
    # ``finetuned_output`` is a response-only field; sending it on create
    # makes the server reject the request with 400 InvalidParameter.
    assert "finetuned_output" not in body
    assert job.job_id == "ft-1"
    assert job.status == "PENDING"
    assert job.request_id == "req-1"


def test_get_list_cancel_delete_paths():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path.endswith("ft-1"):
            return httpx.Response(200, json=_job_payload("RUNNING"))
        if request.method == "GET":
            assert dict(request.url.params) == {
                "page_no": "2",
                "page_size": "5",
            }
            payload = _job_payload("SUCCEEDED")
            return httpx.Response(
                200,
                json={
                    "request_id": "req-2",
                    "output": {
                        "page_no": 2,
                        "page_size": 5,
                        "total": 11,
                        "jobs": [payload["output"]],
                    },
                },
            )
        return httpx.Response(
            200,
            json={"request_id": "req-3", "output": {"status": "done"}},
        )

    client = _make_client(handler)
    job = client.fine_tunes.get("ft-1")
    assert job.status == "RUNNING"
    listed = client.fine_tunes.list(page_no=2, page_size=5)
    assert listed.total_count == 11
    assert listed.jobs[0].job_id == "ft-1"
    canceled = client.fine_tunes.cancel("ft-1")
    assert canceled.status == "done"
    deleted = client.fine_tunes.delete("ft-1")
    assert deleted.status == "done"
    assert seen == [
        ("GET", "/api/v1/fine-tunes/ft-1"),
        ("GET", "/api/v1/fine-tunes"),
        ("POST", "/api/v1/fine-tunes/ft-1/cancel"),
        ("DELETE", "/api/v1/fine-tunes/ft-1"),
    ]


def test_wait_polls_until_terminal():
    statuses = iter(["RUNNING", "RUNNING", "SUCCEEDED"])
    delays = []

    def handler(request):
        return httpx.Response(200, json=_job_payload(next(statuses)))

    client = _make_client(handler)
    job = client.fine_tunes.wait(
        "ft-1",
        _sleep=delays.append,
    )
    assert job.status == "SUCCEEDED"
    assert delays == [1.0, 2.0]


def test_unknown_status_is_not_terminal():
    """The fine-tune plane deliberately excludes ``UNKNOWN``.

    1.x lists ``TaskStatus.UNKNOWN`` as terminal only in
    ``client/base_api.py`` (async-task plane); ``finetune/finetunes.py``
    waits on FAILED/CANCELED/SUCCEEDED alone, and ``fine_tunes._TERMINAL``
    mirrors that narrower rule — the opposite boundary of
    ``_jobs._TERMINAL``, which does treat UNKNOWN as terminal.
    """
    job = FineTune(job_id="ft-1", status="UNKNOWN")
    assert job.finished is False
    assert job.succeeded is False


def test_wait_does_not_stop_on_unknown():
    """An UNKNOWN fine-tune snapshot keeps polling; it never ends the wait."""
    statuses = iter(["UNKNOWN", "SUCCEEDED"])
    delays = []

    def handler(request):
        return httpx.Response(200, json=_job_payload(next(statuses)))

    client = _make_client(handler)
    job = client.fine_tunes.wait(
        "ft-1",
        _sleep=delays.append,
    )
    assert job.status == "SUCCEEDED"
    assert delays == [1.0]


def test_wait_failed_job_raises():
    def handler(request):
        payload = _job_payload("FAILED")
        payload["output"]["code"] = "InternalError"
        payload["output"]["message"] = "training crashed"
        return httpx.Response(
            200,
            json=payload,
            headers={"x-request-id": "req-h", "x-gw-region": "cn-hz"},
        )

    client = _make_client(handler)
    with pytest.raises(DashScopeError) as exc:
        client.fine_tunes.wait("ft-1", _sleep=lambda s: None)
    error = exc.value
    assert error.code == "InternalError"
    # The server said why the job failed, so that reason reaches the
    # caller rather than a generic local sentence.
    assert error.message == "training crashed"
    assert error.headers["x-request-id"] == "req-h"
    assert error.headers["x-gw-region"] == "cn-hz"
    # ``body`` stays the job snapshot: ``headers`` is transport metadata
    # and must not be smuggled into it, since callers log ``err.body``.
    assert "headers" not in error.body
    assert error.body["status"] == "FAILED"
    assert error.body["message"] == "training crashed"


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
    job = client.fine_tunes.wait("ft-1", _sleep=lambda s: None)
    # Compat mode: the first failed poll comes back as the envelope
    # instead of spinning to the timeout and raising there.
    assert job.is_error is True
    assert job.code == "InvalidApiKey"
    assert job.request_id == "req-x"


def test_stream_events_yields_frames_and_raises_on_error():
    sse = (
        'id: 1\ndata: {"status": "RUNNING"}\n\n'
        "id: 2\nevent:error\nstatus: 400\n"
        'data: {"code": "InvalidParameter", "message": "boom"}\n\n'
    )

    def handler(request):
        assert request.url.path == "/api/v1/fine-tunes/ft-1/stream"
        return httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    events = client.fine_tunes.stream_events("ft-1")
    first = next(events)
    assert first.output == {"status": "RUNNING"}
    assert first.is_error is False
    with pytest.raises(BadRequestError):
        next(events)


def test_stream_events_error_without_status_is_business_error():
    # `status:` is optional on the wire. Without it nothing reported a
    # server 500, so the frame is a business failure carried on the HTTP
    # 200 stream and the body's own code picks the class.
    sse = (
        "id: 1\nevent:error\n"
        'data: {"code": "Throttling", "message": "slow down"}\n\n'
    )

    def handler(request):
        return httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    events = client.fine_tunes.stream_events("ft-1")
    with pytest.raises(RateLimitError, match="slow down") as exc:
        next(events)
    assert exc.value.status_code == 200


def test_deployments_crud_and_scale():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        body = json.loads(request.content) if request.content else {}
        if request.method == "POST" and request.url.path.endswith(
            "deployments",
        ):
            assert body == {
                "model_name": "ft-model",
                "capacity": 2,
                "suffix": "alpha",
            }
        if request.method == "PUT":
            assert body == {"deployed_model": "dep-1", "capacity": 4}
        output = {
            "deployed_model": "dep-1",
            "model_name": "ft-model",
            "status": "RUNNING",
            "capacity": 2,
        }
        if request.method == "GET" and not request.url.path.endswith(
            "deployments",
        ):
            return httpx.Response(
                200,
                json={"request_id": "req-1", "output": output},
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "request_id": "req-2",
                    "output": {
                        "page_no": 1,
                        "page_size": 10,
                        "total": 1,
                        "deployments": [output],
                    },
                },
            )
        if request.method == "DELETE":
            return httpx.Response(
                200,
                json={
                    "request_id": "req-3",
                    "output": {"status": "deleted"},
                },
            )
        return httpx.Response(
            200,
            json={"request_id": "req-1", "output": output},
        )

    client = _make_client(handler)
    created = client.deployments.create(
        model="ft-model",
        capacity=2,
        suffix="alpha",
    )
    assert created.deployed_model == "dep-1"
    fetched = client.deployments.get("dep-1")
    assert fetched.model == "ft-model"
    listed = client.deployments.list()
    assert listed.total_count == 1
    scaled = client.deployments.scale("dep-1", capacity=4)
    assert scaled.deployed_model == "dep-1"
    deleted = client.deployments.delete("dep-1")
    assert deleted.status == "deleted"
    assert seen == [
        ("POST", "/api/v1/deployments"),
        ("GET", "/api/v1/deployments/dep-1"),
        ("GET", "/api/v1/deployments"),
        ("PUT", "/api/v1/deployments/dep-1/scale"),
        ("DELETE", "/api/v1/deployments/dep-1"),
    ]


def test_deployments_create_sends_plan_and_mu_fields():
    # The server rejects creates without ``plan`` (400 "plan is blank"),
    # and MU deployments additionally require ``deploy_spec`` and
    # ``billing_method``; all of them must reach the wire verbatim.
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "output": {"deployed_model": "dep-1", "status": "PENDING"},
            },
        )

    client = _make_client(handler)
    created = client.deployments.create(
        model="ft-model",
        capacity=2,
        plan="plan-xx",
        template_id="tpl-1",
        deploy_spec={"gpu_type": "a10", "replicas": 1},
        billing_method="postpaid",
    )
    assert created.deployed_model == "dep-1"
    assert captured["body"] == {
        "model_name": "ft-model",
        "capacity": 2,
        "plan": "plan-xx",
        "template_id": "tpl-1",
        "deploy_spec": {"gpu_type": "a10", "replicas": 1},
        "billing_method": "postpaid",
    }


def test_deployments_stream_events_yields_frames():
    sse = (
        'id: 1\ndata: {"status": "RUNNING"}\n\n'
        'id: 2\ndata: {"status": "SUCCEEDED"}\n\n'
    )

    def handler(request):
        assert request.url.path == "/api/v1/deployments/dep-1/stream"
        return httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    events = list(client.deployments.stream_events("dep-1"))
    assert [event.output for event in events] == [
        {"status": "RUNNING"},
        {"status": "SUCCEEDED"},
    ]
    assert all(not event.is_error for event in events)


def test_deployments_stream_events_error_raises():
    sse = (
        "id: 1\nevent:error\nstatus: 400\n"
        'data: {"code": "InvalidParameter", "message": "boom"}\n\n'
    )

    def handler(request):
        return httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    with pytest.raises(BadRequestError, match="boom"):
        next(client.deployments.stream_events("dep-1"))


def test_deployments_stream_events_error_event_when_not_raising():
    sse = (
        'id: 1\ndata: {"status": "RUNNING"}\n\n'
        "id: 2\nevent:error\nstatus: 400\n"
        'data: {"code": "InvalidParameter", "message": "boom"}\n\n'
    )

    def handler(request):
        return httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler, raise_on_error=False)
    events = list(client.deployments.stream_events("dep-1"))
    assert len(events) == 2
    assert events[0].output == {"status": "RUNNING"}
    assert events[1].is_error is True
    assert events[1].output == {
        "code": "InvalidParameter",
        "message": "boom",
    }


def test_fine_tune_parses_identity_fields():
    def handler(request):
        payload = _job_payload("SUCCEEDED")
        payload["output"].update(
            {
                "user_identity": "uid-1",
                "modifier": "bob",
                "creator": "alice",
                "group": "grp-1",
            },
        )
        return httpx.Response(200, json=payload)

    client = _make_client(handler)
    job = client.fine_tunes.get("ft-1")
    assert job.user_identity == "uid-1"
    assert job.modifier == "bob"
    assert job.creator == "alice"
    assert job.group == "grp-1"


def test_deployment_parses_modifier_and_creator():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "output": {
                    "deployed_model": "dep-1",
                    "status": "RUNNING",
                    "modifier": "bob",
                    "creator": "alice",
                },
            },
        )

    client = _make_client(handler)
    deployment = client.deployments.get("dep-1")
    assert deployment.modifier == "bob"
    assert deployment.creator == "alice"


@pytest.mark.asyncio
async def test_async_fine_tunes_create_and_wait():
    statuses = iter(["RUNNING", "SUCCEEDED"])
    delays = []

    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json=_job_payload("PENDING"))
        return httpx.Response(200, json=_job_payload(next(statuses)))

    client = _make_async_client(handler)

    async def fake_sleep(seconds):
        delays.append(seconds)

    job = await client.fine_tunes.create(
        model="qwen-turbo",
        training_file_ids=["file-1"],
    )
    assert job.status == "PENDING"
    done = await client.fine_tunes.wait(job, _sleep=fake_sleep)
    assert done.status == "SUCCEEDED"
    assert delays == [1.0, 2.0]


@pytest.mark.asyncio
async def test_async_deployments_scale():
    def handler(request):
        assert request.method == "PUT"
        assert request.url.path == "/api/v1/deployments/dep-1/scale"
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "output": {
                    "deployed_model": "dep-1",
                    "status": "SCALING",
                    "capacity": 4,
                },
            },
        )

    client = _make_async_client(handler)
    scaled = await client.deployments.scale("dep-1", capacity=4)
    assert scaled.capacity == 4


@pytest.mark.asyncio
async def test_async_deployments_create_sends_plan():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "output": {"deployed_model": "dep-1", "status": "PENDING"},
            },
        )

    client = _make_async_client(handler)
    created = await client.deployments.create(
        model="ft-model",
        capacity=1,
        plan="plan-xx",
    )
    assert created.deployed_model == "dep-1"
    assert captured["body"] == {
        "model_name": "ft-model",
        "capacity": 1,
        "plan": "plan-xx",
    }


@pytest.mark.asyncio
async def test_async_deployments_stream_events():
    sse = 'id: 1\ndata: {"status": "RUNNING"}\n\n'

    def handler(request):
        assert request.url.path == "/api/v1/deployments/dep-1/stream"
        return httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )

    client = _make_async_client(handler)
    events = [
        event async for event in client.deployments.stream_events("dep-1")
    ]
    assert len(events) == 1
    assert events[0].output == {"status": "RUNNING"}
    assert events[0].is_error is False


@pytest.mark.asyncio
async def test_async_stream_events():
    sse = 'id: 1\ndata: {"status": "RUNNING"}\n\n'

    def handler(request):
        return httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )

    client = _make_async_client(handler)
    events = [event async for event in client.fine_tunes.stream_events("ft-1")]
    assert len(events) == 1
    assert events[0].output == {"status": "RUNNING"}


@pytest.mark.asyncio
async def test_async_stream_events_error_without_status():
    # The async stream used to carry its own copy of the fabricated 500
    # default, so it gets the same coverage as the sync path.
    sse = (
        "id: 1\nevent:error\n"
        'data: {"code": "Throttling", "message": "slow down"}\n\n'
    )

    def handler(request):
        return httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )

    client = _make_async_client(handler)
    with pytest.raises(RateLimitError, match="slow down") as exc:
        async for _ in client.fine_tunes.stream_events("ft-1"):
            pass
    assert exc.value.status_code == 200


def test_business_code_reaches_fine_tune_and_deployment():
    # Both types flatten the envelope, so a failure reported with HTTP 200
    # surfaces as is_error instead of leaving the caller to compare status
    # against the terminal values it has to know about.
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "request_id": "req-e",
                "code": "Job.Failed",
                "message": "training data invalid",
                "output": {"status": "FAILED"},
            },
        )

    client = _make_client(handler, raise_on_error=False)
    job = client.fine_tunes.get("ft-1")
    assert job.is_error
    assert job.code == "Job.Failed"
    assert job.request_id == "req-e"

    dep = client.deployments.get("dep-1")
    assert dep.is_error
    assert dep.code == "Job.Failed"
    assert seen == [
        "/api/v1/fine-tunes/ft-1",
        "/api/v1/deployments/dep-1",
    ]


def test_logs_joins_array_of_lines():
    # The server returns training logs as an array of strings; the SDK
    # normalizes them into a newline-joined ``content`` string.
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-logs",
                "output": {"logs": ["step 1", "step 2", "done"]},
            },
        )

    client = _make_client(handler)
    logs = client.fine_tunes.logs("ft-1")
    assert logs.job_id == "ft-1"
    assert logs.content == "step 1\nstep 2\ndone"


def test_logs_accepts_plain_string():
    # The string form of the response must keep working unchanged.
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-logs",
                "output": {"logs": "step 1\nstep 2"},
            },
        )

    client = _make_client(handler)
    logs = client.fine_tunes.logs("ft-1")
    assert logs.content == "step 1\nstep 2"


@pytest.mark.asyncio
async def test_async_logs_joins_array_of_lines():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-logs",
                "output": {"logs": ["step 1", "step 2"]},
            },
        )

    client = _make_async_client(handler)
    logs = await client.fine_tunes.logs("ft-1")
    assert logs.content == "step 1\nstep 2"
    await client.aclose()


def test_business_code_reaches_list_and_logs_responses():
    # The list/logs parsers re-pick their fields from the flattened body,
    # so the envelope needs threading through just like the single-item
    # responses above.
    envelope = {
        "request_id": "req-x",
        "code": "Throttling",
        "message": "slow down",
    }

    def handler(request):
        if request.url.path.endswith("/logs"):
            return httpx.Response(
                200,
                json={**envelope, "output": {"logs": ""}},
            )
        if request.url.path == "/api/v1/fine-tunes":
            return httpx.Response(
                200,
                json={**envelope, "output": {"jobs": []}},
            )
        return httpx.Response(
            200,
            json={**envelope, "output": {"deployments": []}},
        )

    client = _make_client(handler, raise_on_error=False)
    jobs = client.fine_tunes.list()
    assert jobs.is_error
    assert jobs.code == "Throttling"
    assert jobs.request_id == "req-x"

    logs = client.fine_tunes.logs("ft-1")
    assert logs.is_error
    assert logs.code == "Throttling"

    deployments = client.deployments.list()
    assert deployments.is_error
    assert deployments.code == "Throttling"


@pytest.mark.asyncio
async def test_async_business_code_reaches_list_and_logs_responses():
    envelope = {
        "request_id": "req-x",
        "code": "Throttling",
        "message": "slow down",
    }

    def handler(request):
        if request.url.path.endswith("/logs"):
            return httpx.Response(
                200,
                json={**envelope, "output": {"logs": ""}},
            )
        if request.url.path == "/api/v1/fine-tunes":
            return httpx.Response(
                200,
                json={**envelope, "output": {"jobs": []}},
            )
        return httpx.Response(
            200,
            json={**envelope, "output": {"deployments": []}},
        )

    client = _make_async_client(handler, raise_on_error=False)
    jobs = await client.fine_tunes.list()
    assert jobs.is_error and jobs.code == "Throttling"
    logs = await client.fine_tunes.logs("ft-1")
    assert logs.is_error and logs.code == "Throttling"
    deployments = await client.deployments.list()
    assert deployments.is_error and deployments.code == "Throttling"
    await client.aclose()
