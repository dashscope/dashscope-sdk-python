# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.rl thin mount (M3 3.7)."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import httpx
import pytest

from dashscope.finetune import agentic_rl as agentic_rl_module
from dashscope.finetune.reinforcement.common.errors import (
    AgenticRLError,
    InstanceQueryError,
    OSSUploadError,
    RuntimeErrorWithCode,
)
from dashscope.v2 import AsyncClient, Client
from dashscope.v2.errors import DashScopeError, SdkError
from dashscope.v2.types.common import JobStatus

BASE_URL = "https://api.test.local/api/v1"

# RL job verbs use the v2 client's own base_url like every other v2
# resource; 1.x's forced management-plane host is a 1.x config quirk.
BASE_HOST = httpx.URL(BASE_URL).host


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


class FakeTuning:
    def __init__(self):
        self.functions = None
        self.added = []

    def add_function_components(self, functype, classpaths, workspace_dir):
        self.added.append((functype, classpaths))


class FakeFineTuneOutput:
    """Stands in for the 1.x ``FineTuneOutput`` nested under ``output``."""

    def __init__(self):
        self.job_id = "ft-run-1"
        self.status = "RUNNING"
        self.create_time = "2026-01-02 03:04:05"


class FakeFineTune:
    """Stands in for the 1.x ``FineTune`` dataclass ``rl.run`` returns."""

    def __init__(self):
        self.request_id = "req-run"
        self.status_code = 200
        self.output = FakeFineTuneOutput()


class FakeAgenticRL:
    calls = {}

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.tuning = FakeTuning()
        self.inited = None
        FakeAgenticRL.instances.append(self)

    def init(self, config_path=None, **kwargs):
        self.inited = (config_path, kwargs)
        return self

    async def register_functions(self, lazy_load=True):
        return (["ro-e"], ["rw-e"], [], ["ro-i"], ["rw-i"], [])

    async def upload_datasets(
        self,
        training_files=None,
        validation_files=None,
    ):
        return ["train-1"], ["val-1"]

    async def run(self):
        return FakeFineTune()

    @classmethod
    async def test_functions(
        cls,
        instance_id,
        functype,
        input_data,
        api_key=None,
        pull_logs=False,
        log_page_size=100,
        start_time=None,
        end_time=None,
    ):
        cls.calls["test_functions"] = {
            "instance_id": instance_id,
            "functype": functype,
            "input_data": input_data,
            "api_key": api_key,
            "pull_logs": pull_logs,
            "log_page_size": log_page_size,
            "start_time": start_time,
            "end_time": end_time,
        }
        return {"ok": True, "instance_id": instance_id}

    @classmethod
    async def query_function_instance_logs(
        cls,
        function_instance_id,
        page_number=1,
        page_size=100,
        start_time=None,
        end_time=None,
        keywords=None,
        api_key=None,
    ):
        cls.calls["query_function_instance_logs"] = {
            "function_instance_id": function_instance_id,
            "page_number": page_number,
            "page_size": page_size,
            "start_time": start_time,
            "end_time": end_time,
            "keywords": keywords,
            "api_key": api_key,
        }
        return {"logs": ["line-1"], "page_number": page_number}

    @classmethod
    async def query_all_function_instance_logs(
        cls,
        function_instance_id,
        page_size=100,
        start_time=None,
        end_time=None,
        keywords=None,
        api_key=None,
    ):
        cls.calls["query_all_function_instance_logs"] = {
            "function_instance_id": function_instance_id,
            "page_size": page_size,
            "start_time": start_time,
            "end_time": end_time,
            "keywords": keywords,
            "api_key": api_key,
        }
        return ["line-1", "line-2"]

    @classmethod
    async def delete_function_instance(
        cls,
        function_instance_id,
        api_key=None,
    ):
        cls.calls["delete_function_instance"] = {
            "function_instance_id": function_instance_id,
            "api_key": api_key,
        }
        return {"data": {"status": "deleted"}}


@pytest.fixture(name="fake_agentic_rl")
def fixture_fake_agentic_rl(monkeypatch):
    FakeAgenticRL.instances = []
    FakeAgenticRL.calls = {}
    monkeypatch.setattr(
        agentic_rl_module,
        "AgenticRL",
        FakeAgenticRL,
    )
    return FakeAgenticRL


def test_rl_mounted_on_clients():
    client = _make_client(lambda request: httpx.Response(200, json={}))
    async_client = _make_async_client(
        lambda request: httpx.Response(200, json={}),
    )
    assert hasattr(client, "rl")
    assert hasattr(async_client, "rl")


def test_register_functions_delegates_to_1x(fake_agentic_rl):
    client = _make_client(lambda request: httpx.Response(200, json={}))
    result = client.rl.register_functions(
        rollout_classpaths=["rollout.py:MyRollout"],
        reward_classpaths=["reward.py:MyReward"],
    )
    assert result.rollout_entity_ids == ["ro-e"]
    assert result.reward_entity_ids == ["rw-e"]
    assert result.group_reward_entity_ids == []
    assert result.rollout_instance_ids == ["ro-i"]
    assert result.reward_instance_ids == ["rw-i"]
    assert result.group_reward_instance_ids == []


def test_upload_data_delegates_to_1x(fake_agentic_rl):
    client = _make_client(lambda request: httpx.Response(200, json={}))
    result = client.rl.upload_data(
        training_files=["train.jsonl"],
        validation_files=["val.jsonl"],
    )
    assert result.training_file_ids == ["train-1"]
    assert result.validation_file_ids == ["val-1"]


def test_jobs_create_uses_v2_api_key_and_init(fake_agentic_rl):
    client = _make_client(lambda request: httpx.Response(200, json={}))
    job = client.rl.jobs.create(config_path="init.yaml", job_name="my-job")
    created = fake_agentic_rl.instances[0]
    assert created.api_key == "sk-test"
    assert created.inited == ("init.yaml", {"job_name": "my-job"})
    assert job.job_id == "ft-run-1"
    assert job.status == "RUNNING"
    assert job.created_at == "2026-01-02 03:04:05"
    assert job.request_id == "req-run"
    assert job.status_code == 200
    assert job.finished is False


def test_jobs_wait_polls_until_terminal():
    statuses = ["RUNNING", "SUCCEEDED"]

    def handler(request):
        # wait polls through fetch, same client endpoint as the verbs.
        assert request.url.host == BASE_HOST
        assert request.url.path == "/api/v1/fine-tunes/ft-1"
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "output": {"job_id": "ft-1", "status": statuses.pop(0)},
            },
        )

    client = _make_client(handler)
    job = client.rl.jobs.wait("ft-1", _sleep=lambda _: None)
    assert job.succeeded is True
    assert not statuses


def test_jobs_wait_does_not_stop_on_unknown():
    # RL reuses ``FineTune.finished``, whose terminal set deliberately
    # excludes UNKNOWN (the 1.x finetune-plane rule), so an UNKNOWN
    # snapshot keeps polling rather than ending the wait early.
    statuses = ["UNKNOWN", "SUCCEEDED"]

    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "output": {"job_id": "ft-1", "status": statuses.pop(0)},
            },
        )

    client = _make_client(handler)
    job = client.rl.jobs.wait("ft-1", _sleep=lambda _: None)
    assert job.succeeded is True
    assert not statuses


def test_jobs_wait_failed_raises_with_headers_and_clean_body():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "output": {
                    "job_id": "ft-1",
                    "status": "FAILED",
                    "code": "InternalError",
                    "message": "rollout crashed",
                },
            },
            headers={"x-request-id": "req-h", "x-gw-region": "cn-hz"},
        )

    client = _make_client(handler)
    with pytest.raises(DashScopeError) as exc:
        client.rl.jobs.wait("ft-1", _sleep=lambda _: None)
    error = exc.value
    assert error.code == "InternalError"
    assert error.message == "rollout crashed"
    assert error.headers["x-request-id"] == "req-h"
    assert error.headers["x-gw-region"] == "cn-hz"
    # ``headers`` is transport metadata: it rides on the exception, not
    # inside ``body``, which callers log verbatim.
    assert "headers" not in error.body
    assert error.body["status"] == "FAILED"


def test_jobs_wait_returns_error_envelope_in_compat_mode():
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
    job = client.rl.jobs.wait("ft-1", _sleep=lambda _: None)
    # Compat mode: the first failed poll comes back as the envelope
    # instead of spinning to the timeout and raising there.
    assert job.is_error is True
    assert job.code == "InvalidApiKey"
    assert job.request_id == "req-x"


@pytest.mark.asyncio
async def test_async_jobs_wait_returns_error_envelope_in_compat_mode():
    def handler(request):
        return httpx.Response(
            401,
            json={
                "request_id": "req-x",
                "code": "InvalidApiKey",
                "message": "bad key",
            },
        )

    async def _no_sleep(_):
        return None

    client = _make_async_client(handler, raise_on_error=False)
    job = await client.rl.jobs.wait("ft-1", _sleep=_no_sleep)
    assert job.is_error is True
    assert job.code == "InvalidApiKey"
    assert job.request_id == "req-x"


def test_test_functions_resolves_string_functype(fake_agentic_rl):
    client = _make_client(lambda request: httpx.Response(200, json={}))
    result = client.rl.test_functions(
        instance_id="ro-ins-1",
        func_type="rollout",
        input_data={"prompt": "hi"},
    )
    assert result["ok"] is True
    assert result["instance_id"] == "ro-ins-1"


def test_test_functions_forwards_log_params(fake_agentic_rl):
    client = _make_client(lambda request: httpx.Response(200, json={}))
    client.rl.test_functions(
        instance_id="ro-ins-1",
        func_type="rollout",
        input_data={"prompt": "hi"},
        pull_logs=True,
        log_page_size=20,
        start_time=10,
        end_time=20,
    )
    call = fake_agentic_rl.calls["test_functions"]
    assert call["pull_logs"] is True
    assert call["log_page_size"] == 20
    assert call["start_time"] == 10
    assert call["end_time"] == 20
    assert call["api_key"] == "sk-test"


def test_query_function_instance_logs_delegates_to_1x(fake_agentic_rl):
    client = _make_client(lambda request: httpx.Response(200, json={}))
    result = client.rl.query_function_instance_logs(
        "ro-ins-1",
        page_number=2,
        page_size=50,
        start_time=100,
        end_time=200,
        keywords=["err"],
    )
    assert result["logs"] == ["line-1"]
    assert result["page_number"] == 2
    assert fake_agentic_rl.calls["query_function_instance_logs"] == {
        "function_instance_id": "ro-ins-1",
        "page_number": 2,
        "page_size": 50,
        "start_time": 100,
        "end_time": 200,
        "keywords": ["err"],
        "api_key": "sk-test",
    }


def test_query_all_function_instance_logs_delegates_to_1x(fake_agentic_rl):
    client = _make_client(lambda request: httpx.Response(200, json={}))
    result = client.rl.query_all_function_instance_logs(
        "ro-ins-1",
        page_size=25,
        start_time=100,
        end_time=200,
        keywords=["warn"],
    )
    assert result == ["line-1", "line-2"]
    assert fake_agentic_rl.calls["query_all_function_instance_logs"] == {
        "function_instance_id": "ro-ins-1",
        "page_size": 25,
        "start_time": 100,
        "end_time": 200,
        "keywords": ["warn"],
        "api_key": "sk-test",
    }


def test_delete_function_instance_delegates_to_1x(fake_agentic_rl):
    client = _make_client(lambda request: httpx.Response(200, json={}))
    result = client.rl.delete_function_instance("ro-ins-1")
    assert result["data"]["status"] == "deleted"
    assert fake_agentic_rl.calls["delete_function_instance"] == {
        "function_instance_id": "ro-ins-1",
        "api_key": "sk-test",
    }


def test_job_management_uses_client_base_url():
    # RL job verbs ride the client endpoint like every other v2 resource.
    seen = []

    def handler(request):
        assert request.url.host == BASE_HOST
        seen.append((request.method, request.url.path))
        if request.url.path.endswith("/logs"):
            assert dict(request.url.params) == {"offset": "3", "line": "50"}
            return httpx.Response(
                200,
                json={
                    "request_id": "req-1",
                    "output": {"logs": "line1\nline2"},
                },
            )
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "request_id": "req-2",
                    "output": {"status": "canceled"},
                },
            )
        if request.method == "DELETE":
            return httpx.Response(
                200,
                json={
                    "request_id": "req-4",
                    "output": {"status": "deleted"},
                },
            )
        if request.url.path.endswith("/fine-tunes/ft-1"):
            return httpx.Response(
                200,
                json={
                    "request_id": "req-3",
                    "output": {"job_id": "ft-1", "status": "RUNNING"},
                },
            )
        return httpx.Response(
            200,
            json={
                "request_id": "req-3",
                "output": {
                    "page_no": 1,
                    "page_size": 10,
                    "total": 1,
                    "jobs": [{"job_id": "ft-1", "status": "RUNNING"}],
                },
            },
        )

    client = _make_client(handler)
    job = client.rl.jobs.fetch("ft-1")
    assert job.status == "RUNNING"
    listed = client.rl.jobs.list()
    assert listed.jobs[0].job_id == "ft-1"
    canceled = client.rl.jobs.cancel("ft-1")
    assert canceled.status == "canceled"
    logs = client.rl.jobs.logs("ft-1", offset=3, lines=50)
    assert logs.content == "line1\nline2"
    deleted = client.rl.jobs.delete("ft-1")
    assert isinstance(deleted, JobStatus)
    assert deleted.status == "deleted"
    assert seen == [
        ("GET", "/api/v1/fine-tunes/ft-1"),
        ("GET", "/api/v1/fine-tunes"),
        ("POST", "/api/v1/fine-tunes/ft-1/cancel"),
        ("GET", "/api/v1/fine-tunes/ft-1/logs"),
        ("DELETE", "/api/v1/fine-tunes/ft-1"),
    ]


def test_jobs_verbs_thread_api_key_and_workspace():
    def handler(request):
        assert request.url.host == BASE_HOST
        assert request.headers["Authorization"] == "Bearer sk-override"
        assert request.headers["X-DashScope-WorkSpace"] == "ws-9"
        return httpx.Response(
            200,
            json={
                "request_id": "req-9",
                "output": {"job_id": "ft-1", "status": "RUNNING"},
            },
        )

    client = _make_client(handler)
    job = client.rl.jobs.fetch(
        "ft-1",
        api_key="sk-override",
        workspace="ws-9",
    )
    assert job.job_id == "ft-1"


@pytest.mark.asyncio
async def test_async_rl_upload_and_jobs(fake_agentic_rl):
    def handler(request):
        assert request.url.host == BASE_HOST
        assert request.url.path == "/api/v1/fine-tunes/ft-1/logs"
        return httpx.Response(
            200,
            json={"request_id": "req-1", "output": {"logs": "ok"}},
        )

    client = _make_async_client(handler)
    uploaded = await client.rl.upload_data(training_files=["train.jsonl"])
    assert uploaded.training_file_ids == ["train-1"]
    job = await client.rl.jobs.create(config_path="init.yaml")
    assert job.job_id == "ft-run-1"
    assert job.status == "RUNNING"
    logs = await client.rl.jobs.logs("ft-1")
    assert logs.content == "ok"


@pytest.mark.asyncio
async def test_async_jobs_hit_rl_management_host():
    seen = []

    def handler(request):
        assert request.url.host == BASE_HOST
        seen.append((request.method, request.url.path))
        if request.method == "DELETE":
            return httpx.Response(
                200,
                json={
                    "request_id": "req-4",
                    "output": {"status": "deleted"},
                },
            )
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "request_id": "req-2",
                    "output": {"status": "canceled"},
                },
            )
        return httpx.Response(
            200,
            json={
                "request_id": "req-3",
                "output": {"job_id": "ft-1", "status": "RUNNING"},
            },
        )

    client = _make_async_client(handler)
    job = await client.rl.jobs.fetch("ft-1")
    assert job.status == "RUNNING"
    canceled = await client.rl.jobs.cancel("ft-1")
    assert isinstance(canceled, JobStatus)
    assert canceled.status == "canceled"
    deleted = await client.rl.jobs.delete("ft-1")
    assert isinstance(deleted, JobStatus)
    assert deleted.status == "deleted"
    assert seen == [
        ("GET", "/api/v1/fine-tunes/ft-1"),
        ("POST", "/api/v1/fine-tunes/ft-1/cancel"),
        ("DELETE", "/api/v1/fine-tunes/ft-1"),
    ]


@pytest.mark.asyncio
async def test_async_test_functions_forwards_log_params(fake_agentic_rl):
    client = _make_async_client(lambda request: httpx.Response(200, json={}))
    await client.rl.test_functions(
        instance_id="ro-ins-1",
        func_type="reward",
        input_data={"x": 1},
        pull_logs=True,
        log_page_size=5,
        start_time=1,
        end_time=2,
    )
    call = fake_agentic_rl.calls["test_functions"]
    assert call["pull_logs"] is True
    assert call["log_page_size"] == 5
    assert call["start_time"] == 1
    assert call["end_time"] == 2


@pytest.mark.asyncio
async def test_async_function_instance_ops_delegate(fake_agentic_rl):
    client = _make_async_client(lambda request: httpx.Response(200, json={}))
    page = await client.rl.query_function_instance_logs(
        "ro-ins-1",
        page_number=3,
    )
    assert page["page_number"] == 3
    all_logs = await client.rl.query_all_function_instance_logs(
        "ro-ins-1",
        page_size=25,
    )
    assert all_logs == ["line-1", "line-2"]
    deleted = await client.rl.delete_function_instance("ro-ins-1")
    assert deleted["data"]["status"] == "deleted"
    calls = fake_agentic_rl.calls
    assert calls["query_function_instance_logs"]["page_number"] == 3
    assert calls["query_function_instance_logs"]["api_key"] == "sk-test"
    assert calls["query_all_function_instance_logs"]["page_size"] == 25
    assert calls["delete_function_instance"] == {
        "function_instance_id": "ro-ins-1",
        "api_key": "sk-test",
    }


def test_rl_long_form_alias():
    client = _make_client(lambda r: httpx.Response(200))
    assert client.reinforcement_learning is client.rl
    async_client = _make_async_client(lambda r: httpx.Response(200))
    assert async_client.reinforcement_learning is async_client.rl


# -- 1.x exceptions translated into the v2 error family -----------------


@pytest.fixture(name="fail_step")
def fixture_fail_step(fake_agentic_rl, monkeypatch):
    """Make one named step of the 1.x fake raise ``exc``."""

    # Steps the mount layer invokes as ``AgenticRL`` classmethods.
    classmethod_steps = {
        "test_functions",
        "query_function_instance_logs",
        "query_all_function_instance_logs",
        "delete_function_instance",
    }

    def _set(step, exc):
        async def _afail(first, **kwargs):
            raise exc

        def _sfail(first, **kwargs):
            raise exc

        if step in classmethod_steps:
            replacement = classmethod(_afail)
        elif step == "init":
            replacement = _sfail
        else:
            replacement = _afail
        monkeypatch.setattr(fake_agentic_rl, step, replacement)

    return _set


def test_jobs_create_translates_1x_failure(fail_step):
    fail_step("run", RuntimeErrorWithCode("submit blew up", error_code=3002))
    client = _make_client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(SdkError) as ei:
        client.rl.jobs.create(config_path="init.yaml")
    err = ei.value
    assert err.code == "sdk.agentic_rl.RuntimeErrorWithCode"
    # SDK-side failures have no HTTP semantics, so no invented status.
    assert err.status_code is None
    assert "submit blew up" in err.message
    assert isinstance(err.__cause__, RuntimeErrorWithCode)


def test_translated_error_is_catchable_as_dashscope_error(fail_step):
    fail_step("upload_datasets", OSSUploadError("bucket denied"))
    client = _make_client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(DashScopeError) as ei:
        client.rl.upload_data(training_files=["t.jsonl"])
    assert isinstance(ei.value, SdkError)


def test_exact_catalog_hit_carries_its_own_solution(fail_step):
    fail_step("upload_datasets", OSSUploadError("bucket denied"))
    client = _make_client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(SdkError) as ei:
        client.rl.upload_data(training_files=["t.jsonl"])
    err = ei.value
    assert err.code == "sdk.agentic_rl.OSSUploadError"
    assert err.allow_retry is True
    assert "OSS bucket permissions" in err.solution


def test_uncatalogued_rl_class_falls_back_to_the_family_entry(fail_step):
    # AgenticRLError is the 1.x base but has no catalog entry of its own,
    # so it must land on the family fallback rather than escape raw.
    fail_step("register_functions", AgenticRLError("something odd"))
    client = _make_client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(SdkError) as ei:
        client.rl.register_functions(rollout_classpaths=["r.py:R"])
    assert ei.value.code == "sdk.agentic_rl.Error"


def test_test_functions_translates(fail_step):
    fail_step("test_functions", InstanceQueryError("no instance"))
    client = _make_client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(SdkError) as ei:
        client.rl.test_functions(
            instance_id="ro-ins-1",
            func_type="rollout",
            input_data={"prompt": "hi"},
        )
    assert ei.value.code == "sdk.agentic_rl.InstanceQueryError"


def test_function_instance_delete_translates(fail_step):
    fail_step("delete_function_instance", InstanceQueryError("no instance"))
    client = _make_client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(SdkError) as ei:
        client.rl.delete_function_instance("ro-ins-1")
    assert ei.value.code == "sdk.agentic_rl.InstanceQueryError"


def test_non_rl_exceptions_pass_through(fail_step):
    # A bare ValueError is not part of the rl family, so relabelling it
    # as an SDK error would hide bugs in the mount layer itself.
    fail_step("init", ValueError("plain python error"))
    client = _make_client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(ValueError) as ei:
        client.rl.jobs.create(config_path="init.yaml")
    assert not isinstance(ei.value, SdkError)


@pytest.mark.asyncio
async def test_async_jobs_create_translates(fail_step):
    fail_step("run", RuntimeErrorWithCode("submit blew up", error_code=3002))
    client = _make_async_client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(SdkError) as ei:
        await client.rl.jobs.create(config_path="init.yaml")
    assert ei.value.code == "sdk.agentic_rl.RuntimeErrorWithCode"


@pytest.mark.asyncio
async def test_async_register_functions_translates(fail_step):
    fail_step("register_functions", OSSUploadError("layer upload failed"))
    client = _make_async_client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(SdkError) as ei:
        await client.rl.register_functions(rollout_classpaths=["r.py:R"])
    assert ei.value.code == "sdk.agentic_rl.OSSUploadError"
    assert isinstance(ei.value, DashScopeError)


def test_workflow_steps_forward_the_per_call_api_key(fake_agentic_rl):
    client = _make_client(lambda request: httpx.Response(200, json={}))
    client.rl.jobs.create(api_key="sk-override")
    client.rl.register_functions(api_key="sk-override")
    client.rl.upload_data(
        training_files=["train.jsonl"],
        api_key="sk-override",
    )
    assert [instance.api_key for instance in fake_agentic_rl.instances] == [
        "sk-override",
        "sk-override",
        "sk-override",
    ]
    client.rl.test_functions(
        instance_id="inst-1",
        func_type="rollout",
        input_data={},
        api_key="sk-override",
    )
    assert fake_agentic_rl.calls["test_functions"]["api_key"] == "sk-override"
    client.rl.query_function_instance_logs("inst-1", api_key="sk-override")
    assert (
        fake_agentic_rl.calls["query_function_instance_logs"]["api_key"]
        == "sk-override"
    )
    client.rl.query_all_function_instance_logs("inst-1", api_key="sk-override")
    assert (
        fake_agentic_rl.calls["query_all_function_instance_logs"]["api_key"]
        == "sk-override"
    )
    client.rl.delete_function_instance("inst-1", api_key="sk-override")
    assert (
        fake_agentic_rl.calls["delete_function_instance"]["api_key"]
        == "sk-override"
    )
