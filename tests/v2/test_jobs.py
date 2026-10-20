# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the unified Job infrastructure (M3 task 3.2)."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client
from dashscope.v2.errors import DashScopeError
from dashscope.v2.resources._jobs import AsyncJobs, Job, JobList, Jobs

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


def _job_body(status="PENDING", task_id="task-1", **extra):
    output = {"task_id": task_id, "task_status": status}
    output.update(extra)
    return {"request_id": "req-1", "output": output}


@pytest.fixture
def wait_clock():
    now = [10.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    async def async_sleep(seconds):
        sleep(seconds)

    # Replace only the resource's time binding, not asyncio's global clock.
    return SimpleNamespace(
        monotonic=lambda: now[0],
        sleep=sleep,
        async_sleep=async_sleep,
        sleeps=sleeps,
    )


def test_job_from_payload():
    job = Job.from_payload(
        {
            "request_id": "req-1",
            "output": {
                "task_id": "task-1",
                "task_status": "SUCCEEDED",
                "submit_time": "2026-08-25 10:00:00",
                "results": [{"url": "https://cdn.example.com/a.png"}],
            },
            "usage": {"image_count": 1},
        },
    )
    assert job.task_id == "task-1"
    assert job.succeeded is True
    assert job.finished is True
    assert job.request_id == "req-1"
    assert isinstance(job.output, dict)
    output = job.output
    # pylint: disable-next=unsubscriptable-object
    assert output["results"][0]["url"] == "https://cdn.example.com/a.png"
    assert job.usage.image_count == 1


def test_create_submits_async_task():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_job_body("PENDING"))

    client = _make_client(handler)
    jobs = Jobs(client, "aigc")
    job = jobs.create(
        model="wanx2.1-t2i",
        input={"prompt": "a cat"},
        parameters={"n": 1, "size": "1024*1024"},
    )
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v1/services/aigc"
    assert captured["headers"]["x-dashscope-async"] == "enable"
    assert captured["body"] == {
        "model": "wanx2.1-t2i",
        "input": {"prompt": "a cat"},
        "parameters": {"n": 1, "size": "1024*1024"},
    }
    assert job.task_id == "task-1"
    assert job.status == "PENDING"
    assert job.finished is False


@pytest.mark.parametrize(
    "parameter_kwargs, expected_fields",
    [
        pytest.param({}, {}, id="omitted"),
        pytest.param({"parameters": None}, {}, id="none"),
        pytest.param({"parameters": {}}, {"parameters": {}}, id="empty"),
    ],
)
def test_create_distinguishes_none_and_empty_parameters(
    parameter_kwargs,
    expected_fields,
):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_job_body())

    with _make_client(handler) as client:
        Jobs(client, "aigc").create(
            model="wanx2.1-t2i",
            input={"prompt": "a cat"},
            **parameter_kwargs,
        )
    assert bodies == [
        {
            "model": "wanx2.1-t2i",
            "input": {"prompt": "a cat"},
            **expected_fields,
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parameter_kwargs, expected_fields",
    [
        pytest.param({}, {}, id="omitted"),
        pytest.param({"parameters": None}, {}, id="none"),
        pytest.param({"parameters": {}}, {"parameters": {}}, id="empty"),
    ],
)
async def test_async_create_distinguishes_none_and_empty_parameters(
    parameter_kwargs,
    expected_fields,
):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_job_body())

    async with _make_async_client(handler) as client:
        await AsyncJobs(client, "aigc").create(
            model="wanx2.1-t2i",
            input={"prompt": "a cat"},
            **parameter_kwargs,
        )
    assert bodies == [
        {
            "model": "wanx2.1-t2i",
            "input": {"prompt": "a cat"},
            **expected_fields,
        },
    ]


def test_wait_polls_with_exponential_backoff():
    statuses = iter(["RUNNING", "RUNNING", "SUCCEEDED"])
    calls = []

    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/tasks/task-1"
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json=_job_body(
                next(statuses),
                results=[{"url": "https://cdn.example.com/a.png"}],
            ),
        )

    client = _make_client(handler)
    jobs = Jobs(client, "aigc")
    delays = []
    final = jobs.wait(
        "task-1",
        poll_interval=1.0,
        max_interval=8.0,
        _sleep=delays.append,
    )
    assert final.succeeded is True
    assert len(calls) == 3
    assert delays == [1.0, 2.0]


def test_wait_accepts_job_and_caps_interval():
    statuses = iter(["RUNNING", "RUNNING", "SUCCEEDED"])

    def handler(request):
        return httpx.Response(200, json=_job_body(next(statuses)))

    client = _make_client(handler)
    jobs = Jobs(client, "aigc")
    delays = []
    pending = Job(task_id="task-9", status="PENDING")
    final = jobs.wait(
        pending,
        poll_interval=1.0,
        max_interval=2.0,
        _sleep=delays.append,
    )
    assert final.succeeded is True
    assert delays == [1.0, 2.0, 2.0]


def test_wait_times_out(monkeypatch, wait_clock):
    def handler(request):
        return httpx.Response(200, json=_job_body("RUNNING"))

    monkeypatch.setattr("dashscope.v2.resources._jobs.time", wait_clock)
    client = _make_client(handler)
    jobs = Jobs(client, "aigc")
    with pytest.raises(DashScopeError, match="timed out"):
        jobs.wait("task-1", timeout=0.5, _sleep=wait_clock.sleep)


def test_fetch_failed_raises_mapped_error():
    def handler(request):
        return httpx.Response(
            200,
            json=_job_body(
                "FAILED",
                code="InvalidParameter",
                message="bad prompt",
            ),
            headers={"x-request-id": "req-h", "x-gw-region": "cn-hz"},
        )

    client = _make_client(handler)
    jobs = Jobs(client, "aigc")
    with pytest.raises(DashScopeError, match="bad prompt") as exc_info:
        jobs.fetch("task-1")
    assert exc_info.value.code == "InvalidParameter"
    # A task failure reported with HTTP 200 still carries the gateway
    # headers of the response that reported it, matching what
    # ``errors.response_error`` puts on a transport-level exception.
    assert exc_info.value.headers["x-request-id"] == "req-h"
    assert exc_info.value.headers["x-gw-region"] == "cn-hz"


def test_fetch_failed_returns_job_when_not_raising():
    def handler(request):
        return httpx.Response(
            200,
            json=_job_body("FAILED", code="InvalidParameter"),
        )

    client = _make_client(handler, raise_on_error=False)
    jobs = Jobs(client, "aigc")
    job = jobs.fetch("task-1")
    assert job.status == "FAILED"


def test_wait_raises_on_canceled():
    def handler(request):
        return httpx.Response(200, json=_job_body("CANCELED"))

    client = _make_client(handler)
    jobs = Jobs(client, "aigc")
    with pytest.raises(DashScopeError, match="canceled"):
        jobs.wait("task-1", _sleep=lambda seconds: None)


def _list_body(**extra):
    body = {
        "request_id": "req-1",
        "total": 11,
        "page_no": 2,
        "page_size": 5,
        "tasks": [
            {
                "task_id": "task-9",
                "task_status": "SUCCEEDED",
                "submit_time": "2026-08-25 10:00:00",
            },
        ],
    }
    body.update(extra)
    return body


def test_cancel_posts_to_task_cancel():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(200, json=_job_body("CANCELED"))

    client = _make_client(handler)
    jobs = Jobs(client, "aigc")
    job = jobs.cancel("task-1")
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v1/tasks/task-1/cancel"
    assert job.task_id == "task-1"
    assert job.status == "CANCELED"


def test_list_gets_tasks_with_pagination():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        return httpx.Response(200, json=_list_body())

    client = _make_client(handler)
    jobs = Jobs(client, "aigc")
    result = jobs.list(
        model_name="wanx2.1-t2i",
        status="SUCCEEDED",
        page_no=2,
        page_size=5,
    )
    assert captured["method"] == "GET"
    assert captured["path"] == "/api/v1/tasks"
    assert captured["query"] == {
        "page_no": "2",
        "page_size": "5",
        "model_name": "wanx2.1-t2i",
        "status": "SUCCEEDED",
    }
    assert isinstance(result, JobList)
    assert result.total == 11
    assert result.page_no == 2
    assert result.page_size == 5
    assert result.request_id == "req-1"
    assert result.tasks[0].task_id == "task-9"
    assert result.tasks[0].succeeded is True
    assert result.tasks[0].submitted_at == "2026-08-25 10:00:00"


@pytest.mark.asyncio
async def test_async_cancel_and_list():
    captured = []

    def handler(request):
        captured.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(200, json=_job_body("CANCELED"))
        return httpx.Response(200, json=_list_body())

    client = _make_async_client(handler)
    jobs = AsyncJobs(client, "aigc")
    canceled = await jobs.cancel("task-1")
    assert canceled.status == "CANCELED"
    result = await jobs.list(status="FAILED")
    assert isinstance(result, JobList)
    assert result.tasks[0].task_id == "task-9"
    assert captured == [
        ("POST", "/api/v1/tasks/task-1/cancel"),
        ("GET", "/api/v1/tasks"),
    ]


@pytest.mark.asyncio
async def test_async_create_fetch_wait():
    statuses = iter(["PENDING", "RUNNING", "SUCCEEDED"])
    captured = {}

    def handler(request):
        if request.method == "POST":
            captured["header"] = request.headers.get("x-dashscope-async")
            captured["path"] = request.url.path
            return httpx.Response(200, json=_job_body(next(statuses)))
        return httpx.Response(200, json=_job_body(next(statuses)))

    client = _make_async_client(handler)
    jobs = AsyncJobs(client, "videosynthesis")
    pending = await jobs.create(
        model="wanx2.1-t2v",
        input={"prompt": "a running cat"},
    )
    assert captured["path"] == "/api/v1/services/videosynthesis"
    assert captured["header"] == "enable"
    assert pending.status == "PENDING"

    delays = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    final = await jobs.wait(pending, _sleep=fake_sleep)
    assert final.succeeded is True
    assert delays == [1.0, 2.0]


@pytest.mark.asyncio
async def test_async_fetch_failed_carries_response_headers():
    """The async jobs plane threads headers exactly like the sync one."""

    def handler(request):
        return httpx.Response(
            200,
            json=_job_body(
                "FAILED",
                code="InvalidParameter",
                message="bad prompt",
            ),
            headers={"x-request-id": "req-h", "x-gw-region": "cn-hz"},
        )

    client = _make_async_client(handler)
    jobs = AsyncJobs(client, "aigc")
    with pytest.raises(DashScopeError, match="bad prompt") as exc_info:
        await jobs.fetch("task-1")
    assert exc_info.value.headers["x-request-id"] == "req-h"
    assert exc_info.value.headers["x-gw-region"] == "cn-hz"


def test_wait_returns_failed_job_when_not_raising():
    def handler(request):
        return httpx.Response(
            200,
            json=_job_body(
                "FAILED",
                code="InvalidParameter",
                message="bad prompt",
            ),
        )

    client = _make_client(handler, raise_on_error=False)
    jobs = Jobs(client, "aigc")
    job = jobs.wait("task-1", _sleep=lambda seconds: None)
    assert job.status == "FAILED"
    assert job.code == "InvalidParameter"


def test_wait_returns_canceled_job_when_not_raising():
    def handler(request):
        return httpx.Response(200, json=_job_body("CANCELED"))

    client = _make_client(handler, raise_on_error=False)
    jobs = Jobs(client, "aigc")
    job = jobs.wait("task-1", _sleep=lambda seconds: None)
    assert job.status == "CANCELED"


def test_wait_returns_error_envelope_when_not_raising():
    def handler(request):
        return httpx.Response(
            401,
            json={"code": "InvalidApiKey", "message": "bad key"},
        )

    client = _make_client(handler, raise_on_error=False)
    jobs = Jobs(client, "aigc")
    sleeps = []
    job = jobs.wait(
        "task-1",
        timeout=60.0,
        _sleep=lambda seconds: sleeps.append(seconds),
    )
    # The error envelope has no task status; wait must return it
    # instead of polling until the timeout.
    assert job.is_error is True
    assert job.code == "InvalidApiKey"
    assert sleeps == []


@pytest.mark.asyncio
async def test_async_wait_returns_error_envelope_when_not_raising():
    def handler(request):
        return httpx.Response(
            401,
            json={"code": "InvalidApiKey", "message": "bad key"},
        )

    client = _make_async_client(handler, raise_on_error=False)
    jobs = AsyncJobs(client, "aigc")
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    job = await jobs.wait("task-1", timeout=60.0, _sleep=fake_sleep)
    assert job.is_error is True
    assert job.code == "InvalidApiKey"
    assert sleeps == []


def _fail_sleep(seconds):
    """``_sleep`` seam that fails the test if wait() ever pauses.

    Raising rather than recording keeps a regression fast: without the
    UNKNOWN terminal state the poll loop would spin against the 60s
    deadline issuing requests, instead of stopping on the first one.
    """
    raise AssertionError("wait() must not re-poll an UNKNOWN task")


async def _fail_sleep_async(seconds):
    raise AssertionError("wait() must not re-poll an UNKNOWN task")


def test_unknown_status_is_terminal():
    """``UNKNOWN`` is the gateway's answer for a task id that is not there.

    1.x lists ``TaskStatus.UNKNOWN`` alongside SUCCEEDED/FAILED/CANCELED
    in ``client/base_api.py``, so ``finished`` must agree.
    """
    job = Job(task_id="task-missing", status="UNKNOWN")
    assert job.finished is True
    assert job.succeeded is False


def test_wait_returns_unknown_job_without_polling():
    """A 200 + UNKNOWN must end the wait, not be read as "still running"."""
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=_job_body("UNKNOWN"))

    client = _make_client(handler)
    jobs = Jobs(client, "aigc")
    job = jobs.wait("task-missing", timeout=60.0, _sleep=_fail_sleep)
    assert job.status == "UNKNOWN"
    assert job.finished is True
    # No business ``code`` arrives with it, so it is data, not an error.
    assert job.is_error is False
    assert calls == ["/api/v1/tasks/task-missing"]


@pytest.mark.asyncio
async def test_async_wait_returns_unknown_job_without_polling():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=_job_body("UNKNOWN"))

    client = _make_async_client(handler)
    jobs = AsyncJobs(client, "aigc")
    job = await jobs.wait(
        "task-missing",
        timeout=60.0,
        _sleep=_fail_sleep_async,
    )
    assert job.status == "UNKNOWN"
    assert job.finished is True
    assert job.is_error is False
    assert calls == ["/api/v1/tasks/task-missing"]


def test_wait_still_polls_past_running_to_succeed():
    """Guard the fix's blast radius: non-terminal states keep polling."""
    statuses = iter(["PENDING", "RUNNING", "SUCCEEDED"])
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=_job_body(next(statuses)))

    client = _make_client(handler)
    jobs = Jobs(client, "aigc")
    delays = []
    job = jobs.wait(
        "task-1",
        timeout=60.0,
        _sleep=delays.append,
    )
    assert job.succeeded is True
    assert len(calls) == 3
    assert delays == [1.0, 2.0]


@pytest.mark.parametrize("modality", ["images", "videos", "embeddings"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("raise_on_error", [False, True])
@pytest.mark.parametrize(
    "status",
    ["SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"],
)
async def test_non_asr_wait_terminal_snapshot_contract(
    modality,
    asynchronous,
    raise_on_error,
    status,
):
    def handler(request):
        pytest.fail("terminal snapshots must not trigger another request")

    client = (
        _make_async_client(handler, raise_on_error=raise_on_error)
        if asynchronous
        else _make_client(handler, raise_on_error=raise_on_error)
    )
    job = Job.from_payload(
        _job_body(
            status,
            **(
                {"code": "InvalidParameter", "message": "invalid input"}
                if status == "FAILED"
                else {}
            ),
        ),
    )
    try:
        jobs = getattr(client, modality).jobs
        if raise_on_error and status == "CANCELED":
            with pytest.raises(DashScopeError, match="canceled") as caught:
                result = jobs.wait(job)
                if asynchronous:
                    await result
            assert type(caught.value) is DashScopeError
            assert caught.value.request_id == "req-1"
        else:
            result = jobs.wait(job)
            if asynchronous:
                result = await result
            assert result is job
    finally:
        if asynchronous:
            await client.aclose()
        else:
            client.close()


@pytest.mark.parametrize("modality", ["images", "videos", "embeddings"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("raise_on_error", [False, True])
async def test_non_asr_wait_allows_final_fetch_at_deadline(
    monkeypatch,
    wait_clock,
    modality,
    asynchronous,
    raise_on_error,
):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        assert request.url.path == "/api/v1/tasks/task-1"
        assert wait_clock.monotonic() == 10.5
        return httpx.Response(200, json=_job_body("SUCCEEDED"))

    monkeypatch.setattr("dashscope.v2.resources._jobs.time", wait_clock)
    client = (
        _make_async_client(handler, raise_on_error=raise_on_error)
        if asynchronous
        else _make_client(handler, raise_on_error=raise_on_error)
    )
    try:
        result = getattr(client, modality).jobs.wait(
            Job(task_id="task-1", status="RUNNING"),
            timeout=0.5,
            poll_interval=1,
            _sleep=wait_clock.async_sleep
            if asynchronous
            else wait_clock.sleep,
        )
        if asynchronous:
            result = await result
        assert result.succeeded is True
        assert len(requests) == 1
        assert wait_clock.sleeps == [0.5]
    finally:
        if asynchronous:
            await client.aclose()
        else:
            client.close()


@pytest.mark.parametrize("modality", ["images", "videos", "embeddings", "asr"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("raise_on_error", [False, True])
@pytest.mark.parametrize("status_code", [None, 0])
async def test_public_wait_polls_running_until_succeeded(
    monkeypatch,
    wait_clock,
    modality,
    asynchronous,
    raise_on_error,
    status_code,
):
    statuses = iter(["RUNNING", "RUNNING", "RUNNING", "SUCCEEDED"])
    requests = []
    output_extra = {} if status_code is None else {"status_code": status_code}

    def handler(request):
        requests.append(request)
        assert request.method == "GET"
        assert request.url.path == "/api/v1/tasks/task-1"
        assert request.headers["Authorization"] == "Bearer sk-override"
        assert request.headers["X-DashScope-WorkSpace"] == "ws-9"
        return httpx.Response(
            200,
            json=_job_body(next(statuses), **output_extra),
        )

    clock_module = "audio.asr" if modality == "asr" else "_jobs"
    monkeypatch.setattr(
        f"dashscope.v2.resources.{clock_module}.time",
        wait_clock,
    )
    client = (
        _make_async_client(handler, raise_on_error=raise_on_error)
        if asynchronous
        else _make_client(handler, raise_on_error=raise_on_error)
    )
    try:
        resource = (
            client.audio.asr
            if modality == "asr"
            else getattr(client, modality)
        )
        result = resource.jobs.wait(
            "task-1",
            timeout=60,
            poll_interval=1,
            max_interval=2,
            api_key="sk-override",
            workspace="ws-9",
            _sleep=wait_clock.async_sleep
            if asynchronous
            else wait_clock.sleep,
        )
        if asynchronous:
            result = await result
        assert result.succeeded is True
        assert result.is_error is False
        assert result.request_id == "req-1"
        if status_code is not None:
            assert result.output["status_code"] == status_code
        assert len(requests) == 4
        assert wait_clock.sleeps == [1, 2, 2]
    finally:
        if asynchronous:
            await client.aclose()
        else:
            client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("raise_on_error", [False, True])
async def test_asr_wait_does_not_fetch_after_budget_expires(
    monkeypatch,
    wait_clock,
    asynchronous,
    raise_on_error,
):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=_job_body("SUCCEEDED"))

    monkeypatch.setattr("dashscope.v2.resources.audio.asr.time", wait_clock)
    client = (
        _make_async_client(handler, raise_on_error=raise_on_error)
        if asynchronous
        else _make_client(handler, raise_on_error=raise_on_error)
    )
    try:
        with pytest.raises(DashScopeError, match="timed out"):
            result = client.audio.asr.jobs.wait(
                Job(task_id="task-1", status="RUNNING"),
                timeout=0.5,
                poll_interval=1,
                _sleep=(
                    wait_clock.async_sleep
                    if asynchronous
                    else wait_clock.sleep
                ),
            )
            if asynchronous:
                await result
        assert requests == []
        assert wait_clock.sleeps == [0.5]
    finally:
        if asynchronous:
            await client.aclose()
        else:
            client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("raise_on_error", [False, True])
@pytest.mark.parametrize(
    "status",
    ["SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"],
)
async def test_asr_wait_terminal_snapshot_contract(
    asynchronous,
    raise_on_error,
    status,
):
    def handler(request):
        pytest.fail("terminal snapshots must not trigger another request")

    client = (
        _make_async_client(handler, raise_on_error=raise_on_error)
        if asynchronous
        else _make_client(handler, raise_on_error=raise_on_error)
    )
    job = Job.from_payload(
        _job_body(
            status,
            **(
                {"code": "InvalidParameter", "message": "invalid audio"}
                if status == "FAILED"
                else {}
            ),
        ),
    )
    job.headers = {"x-request-id": "header-id", "x-gateway": "test"}
    try:
        if raise_on_error and status in ("FAILED", "CANCELED"):
            with pytest.raises(DashScopeError) as caught:
                result = client.audio.asr.jobs.wait(job)
                if asynchronous:
                    await result
            assert caught.value.request_id == "req-1"
            assert dict(caught.value.headers) == job.headers
            if status == "FAILED":
                assert type(caught.value).__name__ == "BadRequestError"
                assert caught.value.code == "InvalidParameter"
                assert caught.value.message == "invalid audio"
        else:
            result = client.audio.asr.jobs.wait(job)
            if asynchronous:
                result = await result
            assert result is job
    finally:
        if asynchronous:
            await client.aclose()
        else:
            client.close()


async def test_async_asr_wait_cancellation_stops_polling():
    import asyncio

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=_job_body("RUNNING"))

    async def cancel_sleep(seconds):
        raise asyncio.CancelledError()

    async with _make_async_client(handler) as client:
        with pytest.raises(asyncio.CancelledError):
            await client.audio.asr.jobs.wait(
                "task-1",
                _sleep=cancel_sleep,
            )
    assert len(requests) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("entry", ["transcribe", "fetch"])
@pytest.mark.parametrize(
    "status, code, expected_error",
    [
        (400, "InvalidParameter", "BadRequestError"),
        (401, "InvalidApiKey", "AuthenticationError"),
        (403, "AccessDenied", "PermissionDeniedError"),
        (404, "NotFound", "NotFoundError"),
        (429, "Throttling", "RateLimitError"),
        (500, "InternalError", "InternalServerError"),
        (200, "InvalidParameter", "BadRequestError"),
    ],
)
async def test_asr_http_errors_preserve_request_id(
    asynchronous,
    entry,
    status,
    code,
    expected_error,
):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            status,
            headers={"x-request-id": "header-request"},
            json={"code": code, "message": "audio request rejected"},
        )

    client = (
        _make_async_client(handler, max_retries=0)
        if asynchronous
        else _make_client(handler, max_retries=0)
    )
    try:
        with pytest.raises(DashScopeError) as caught:
            if entry == "transcribe":
                result = client.audio.asr.transcribe(
                    ["https://example.com/a.wav"],
                )
            else:
                result = client.audio.asr.jobs.fetch("task-1")
            if asynchronous:
                await result
        assert type(caught.value).__name__ == expected_error
        assert caught.value.code == code
        assert caught.value.request_id == "header-request"
        assert caught.value.status_code == status
        assert len(requests) == 1
    finally:
        if asynchronous:
            await client.aclose()
        else:
            client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "transport_error, expected_error",
    [
        (httpx.ReadTimeout, "APITimeoutError"),
        (httpx.ConnectError, "APIConnectionError"),
    ],
)
async def test_asr_transport_errors_are_not_suppressed(
    asynchronous,
    transport_error,
    expected_error,
):
    calls = []

    def handler(request):
        calls.append(request)
        raise transport_error("network fault", request=request)

    client = (
        _make_async_client(handler, raise_on_error=False, max_retries=0)
        if asynchronous
        else _make_client(handler, raise_on_error=False, max_retries=0)
    )
    try:
        with pytest.raises(DashScopeError) as caught:
            result = client.audio.asr.transcribe(["https://example.com/a.wav"])
            if asynchronous:
                await result
        assert type(caught.value).__name__ == expected_error
        assert len(calls) == 1
    finally:
        if asynchronous:
            await client.aclose()
        else:
            client.close()
