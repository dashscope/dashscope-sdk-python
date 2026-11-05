# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope rl`` -- the jobs and functions subgroups."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from dashscope import errors as sdk_errors
from dashscope.cli.app import build_app
from dashscope.resources.fine_tunes import FineTune, FineTuneList, JobLogs
from dashscope.resources.reinforcement_learning import (
    RegisteredFunctions,
    UploadedData,
)
from dashscope.types.common import JobStatus


def _job(**overrides: Any) -> FineTune:
    data: Dict[str, Any] = {
        "job_id": "rl-job-1",
        "job_name": "my-rl-run",
        "status": "SUCCEEDED",
        "model": "qwen-plus",
        "created_at": "2026-01-01 12:00:00",
    }
    data.update(overrides)
    return FineTune(**data)


class _RlJobs:
    def __init__(self, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results

    def _record(self, name: str, *args, **kwargs):
        self.calls.append({"method": name, "args": args, "kwargs": kwargs})
        result = self._results.get(name)
        if isinstance(result, Exception):
            raise result
        return result() if callable(result) else result

    def create(self, **kwargs):
        return self._record("create", **kwargs)

    def fetch(self, job_id, **kwargs):
        return self._record("fetch", job_id, **kwargs)

    def list(self, **kwargs):
        return self._record("list", **kwargs)

    def logs(self, job_id, **kwargs):
        return self._record("logs", job_id, **kwargs)

    def cancel(self, job_id, **kwargs):
        return self._record("cancel", job_id, **kwargs)

    def delete(self, job_id, **kwargs):
        return self._record("delete", job_id, **kwargs)

    def wait(self, job, **kwargs):
        self._record("wait", job, **kwargs)
        return self._results.get("wait", job)


class _Rl:
    """Stand-in for ``client.rl`` recording every call."""

    def __init__(self, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results
        self.jobs = _RlJobs(**results.get("jobs", {}))

    def _record(self, name: str, *args, **kwargs):
        self.calls.append({"method": name, "args": args, "kwargs": kwargs})
        result = self._results.get(name)
        if isinstance(result, Exception):
            raise result
        return result() if callable(result) else result

    def register_functions(self, **kwargs):
        return self._record("register_functions", **kwargs)

    def test_functions(self, **kwargs):
        return self._record("test_functions", **kwargs)

    def upload_data(self, *args, **kwargs):
        return self._record("upload_data", *args, **kwargs)

    def query_function_instance_logs(self, instance_id, **kwargs):
        return self._record(
            "query_function_instance_logs",
            instance_id,
            **kwargs,
        )

    def query_all_function_instance_logs(self, instance_id, **kwargs):
        return self._record(
            "query_all_function_instance_logs",
            instance_id,
            **kwargs,
        )

    def delete_function_instance(self, instance_id, **kwargs):
        return self._record("delete_function_instance", instance_id, **kwargs)


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def rl(make_client):
    def install(**results: Any) -> _Rl:
        resource = _Rl(**results)
        make_client(rl=resource)
        return resource

    return install


def _as_terminal_tty(monkeypatch, answer: bool):
    monkeypatch.setattr("dashscope.cli.confirm.stdin_is_tty", lambda: True)
    monkeypatch.setattr(
        "dashscope.cli.confirm.typer.confirm",
        lambda *a, **k: answer,
    )


class TestJobsCreate:
    def test_create_submits_without_waiting(self, invoke, rl):
        resource = rl(jobs={"create": _job(status="PENDING")})
        result = invoke(["rl", "jobs", "create", "--config", "rl.yaml"])
        assert result.exit_code == 0
        assert resource.jobs.calls[0]["kwargs"]["config_path"] == "rl.yaml"
        assert [c["method"] for c in resource.jobs.calls] == ["create"]

    def test_create_forwards_config_overrides(self, invoke, rl):
        resource = rl(jobs={"create": _job()})
        invoke(
            [
                "rl",
                "jobs",
                "create",
                "--config",
                "rl.yaml",
                "--job-name",
                "run-7",
                "--set",
                "epochs=3",
                "--set",
                'files=["a.jsonl","b.jsonl"]',
                "--set",
                "note=plain string",
            ],
        )
        kwargs = resource.jobs.calls[0]["kwargs"]
        assert kwargs["job_name"] == "run-7"
        assert kwargs["epochs"] == 3
        assert kwargs["files"] == ["a.jsonl", "b.jsonl"]
        assert kwargs["note"] == "plain string"

    def test_create_rejects_a_malformed_override(self, invoke, rl):
        rl(jobs={"create": _job()})
        result = invoke(["rl", "jobs", "create", "--set", "epochs"])
        assert result.exit_code == 2
        assert "key=value" in result.stderr

    def test_create_wait_blocks_on_the_rl_waiter(self, invoke, rl):
        resource = rl(
            jobs={"create": _job(status="PENDING"), "wait": _job()},
        )
        invoke(["rl", "jobs", "create", "--wait", "--timeout", "30"])
        methods = [c["method"] for c in resource.jobs.calls]
        assert methods == ["create", "wait"]
        assert resource.jobs.calls[1]["kwargs"]["timeout"] == 30.0

    def test_create_json_output(self, invoke, rl):
        rl(jobs={"create": _job()})
        result = invoke(
            ["rl", "jobs", "create"],
            global_args=["-o", "json"],
        )
        doc = json.loads(result.stdout)
        assert doc["job_id"] == "rl-job-1"


class TestJobsRead:
    def test_get(self, invoke, rl):
        resource = rl(jobs={"fetch": _job()})
        result = invoke(["rl", "jobs", "get", "rl-job-1"])
        assert result.exit_code == 0
        assert resource.jobs.calls[0]["args"] == ("rl-job-1",)
        assert "SUCCEEDED" in result.stdout

    def test_list(self, invoke, rl):
        rl(
            jobs={
                "list": FineTuneList(
                    jobs=[_job(), _job(job_id="rl-job-2")],
                    page_no=1,
                    page_size=10,
                    total_count=2,
                ),
            },
        )
        result = invoke(["rl", "jobs", "list"])
        assert result.exit_code == 0
        assert "rl-job-1" in result.stdout
        assert "rl-job-2" in result.stdout

    def test_list_pagination_flags(self, invoke, rl):
        resource = rl(
            jobs={"list": FineTuneList(jobs=[], page_no=2, page_size=5)},
        )
        invoke(["rl", "jobs", "list", "--page", "2", "--size", "5"])
        assert resource.jobs.calls[0]["kwargs"] == {
            "page_no": 2,
            "page_size": 5,
        }

    def test_logs_print_the_content(self, invoke, rl):
        resource = rl(
            jobs={
                "logs": JobLogs(
                    job_id="rl-job-1",
                    content="line one\nline two",
                ),
            },
        )
        result = invoke(["rl", "jobs", "logs", "rl-job-1", "--lines", "50"])
        assert result.exit_code == 0
        assert resource.jobs.calls[0]["kwargs"] == {"offset": 1, "lines": 50}
        assert "line one" in result.stdout

    def test_logs_json_output(self, invoke, rl):
        rl(jobs={"logs": JobLogs(job_id="rl-job-1", content="hello")})
        result = invoke(
            ["rl", "jobs", "logs", "rl-job-1"],
            global_args=["-o", "json"],
        )
        assert json.loads(result.stdout)["content"] == "hello"

    def test_wait_polls_the_id_and_nothing_else(self, invoke, rl):
        resource = rl(jobs={"wait": _job()})
        result = invoke(["rl", "jobs", "wait", "rl-job-1"])
        assert result.exit_code == 0
        assert [c["method"] for c in resource.jobs.calls] == ["wait"]
        assert resource.jobs.calls[0]["args"] == ("rl-job-1",)

    def test_wait_defaults_to_an_hour(self, invoke, rl):
        resource = rl(jobs={"wait": _job()})
        invoke(["rl", "jobs", "wait", "rl-job-1"])
        assert resource.jobs.calls[0]["kwargs"] == {"timeout": 3600.0}

    def test_wait_table_shows_the_settled_job(self, invoke, rl):
        rl(jobs={"wait": _job(status="SUCCEEDED")})
        result = invoke(["rl", "jobs", "wait", "rl-job-1"])
        assert "SUCCEEDED" in result.stdout

    def test_wait_json_output_is_the_whole_job(self, invoke, rl):
        rl(jobs={"wait": _job()})
        result = invoke(
            ["rl", "jobs", "wait", "rl-job-1"],
            global_args=["-o", "json"],
        )
        doc = json.loads(result.stdout)
        assert doc["job_id"] == "rl-job-1"
        assert doc["status"] == "SUCCEEDED"

    def test_wait_failure_exits_nonzero(self, invoke, rl):
        rl(jobs={"wait": sdk_errors.InternalServerError("training diverged")})
        result = invoke(["rl", "jobs", "wait", "rl-job-1"])
        assert result.exit_code == 1
        assert "Error [server]: training diverged" in result.stderr

    def test_wait_requires_the_id(self, invoke, rl):
        resource = rl(jobs={"wait": _job()})
        assert invoke(["rl", "jobs", "wait"]).exit_code == 2
        assert resource.jobs.calls == []

    @pytest.mark.parametrize("timeout", ["0", "nan"])
    def test_wait_rejects_a_timeout_it_could_not_honour(
        self,
        invoke,
        rl,
        timeout,
    ):
        """``nan`` is false against every comparison, so the poll never ends."""
        resource = rl(jobs={"wait": _job()})
        result = invoke(
            ["rl", "jobs", "wait", "rl-job-1", "--timeout", timeout],
        )
        assert result.exit_code == 2
        assert "--timeout" in result.stderr
        assert resource.jobs.calls == []


class TestJobsDestructive:
    def test_delete_with_yes(self, invoke, rl):
        resource = rl(jobs={"delete": JobStatus()})
        result = invoke(["rl", "jobs", "delete", "rl-job-1", "--yes"])
        assert result.exit_code == 0
        assert resource.jobs.calls[0]["args"] == ("rl-job-1",)

    def test_delete_refuses_without_a_tty(self, invoke, rl):
        resource = rl(jobs={"delete": JobStatus()})
        result = invoke(["rl", "jobs", "delete", "rl-job-1"])
        assert result.exit_code == 2
        assert "--yes" in result.stderr
        assert resource.jobs.calls == []

    def test_delete_confirms_in_a_tty(self, invoke, rl, monkeypatch):
        _as_terminal_tty(monkeypatch, answer=True)
        resource = rl(jobs={"delete": JobStatus()})
        result = invoke(["rl", "jobs", "delete", "rl-job-1"])
        assert result.exit_code == 0
        assert len(resource.jobs.calls) == 1

    def test_delete_aborts_on_no(self, invoke, rl, monkeypatch):
        _as_terminal_tty(monkeypatch, answer=False)
        resource = rl(jobs={"delete": JobStatus()})
        result = invoke(["rl", "jobs", "delete", "rl-job-1"])
        assert result.exit_code != 0
        assert resource.jobs.calls == []

    def test_cancel_with_yes(self, invoke, rl):
        resource = rl(jobs={"cancel": JobStatus()})
        result = invoke(["rl", "jobs", "cancel", "rl-job-1", "--yes"])
        assert result.exit_code == 0
        assert resource.jobs.calls[0]["args"] == ("rl-job-1",)

    def test_cancel_refuses_without_a_tty(self, invoke, rl):
        resource = rl(jobs={"cancel": JobStatus()})
        result = invoke(["rl", "jobs", "cancel", "rl-job-1"])
        assert result.exit_code == 2
        assert resource.jobs.calls == []


class TestFunctions:
    def test_register_passes_the_classpaths(self, invoke, rl):
        resource = rl(
            register_functions=RegisteredFunctions(
                rollout_entity_ids=["e1"],
                rollout_instance_ids=["i1"],
            ),
        )
        result = invoke(
            [
                "rl",
                "functions",
                "register",
                "--rollout",
                "mypkg.rollout:MyRollout",
                "--reward",
                "mypkg.reward:MyReward",
                "--workspace-dir",
                "/tmp/ws",
                "--no-lazy-load",
            ],
        )
        assert result.exit_code == 0
        kwargs = resource.calls[0]["kwargs"]
        assert kwargs["rollout_classpaths"] == ["mypkg.rollout:MyRollout"]
        assert kwargs["reward_classpaths"] == ["mypkg.reward:MyReward"]
        assert kwargs["group_reward_classpaths"] == []
        assert kwargs["workspace_dir"] == "/tmp/ws"
        assert kwargs["lazy_load"] is False
        assert "i1" in result.stdout

    def test_register_requires_a_function(self, invoke, rl):
        resource = rl(register_functions=RegisteredFunctions())
        result = invoke(["rl", "functions", "register"])
        assert result.exit_code == 2
        assert "--rollout" in result.stderr
        assert resource.calls == []

    def test_test_passes_the_payload(self, invoke, rl):
        resource = rl(test_functions={"score": 1.0})
        result = invoke(
            [
                "rl",
                "functions",
                "test",
                "--instance-id",
                "i1",
                "--type",
                "reward",
                "--input",
                '{"answer": "42"}',
            ],
        )
        assert result.exit_code == 0
        kwargs = resource.calls[0]["kwargs"]
        assert kwargs["instance_id"] == "i1"
        assert kwargs["func_type"] == "reward"
        assert kwargs["input_data"] == {"answer": "42"}

    def test_test_rejects_a_non_object_input(self, invoke, rl):
        rl(test_functions={})
        result = invoke(
            [
                "rl",
                "functions",
                "test",
                "--instance-id",
                "i1",
                "--type",
                "reward",
                "--input",
                "[1,2]",
            ],
        )
        assert result.exit_code == 2

    def test_upload_data_passes_both_file_lists(self, invoke, rl):
        resource = rl(
            upload_data=UploadedData(
                training_file_ids=["f1"],
                validation_file_ids=["f2"],
            ),
        )
        result = invoke(
            [
                "rl",
                "functions",
                "upload-data",
                "--training-file",
                "train.jsonl",
                "--validation-file",
                "val.jsonl",
            ],
        )
        assert result.exit_code == 0
        assert resource.calls[0]["args"] == (["train.jsonl"], ["val.jsonl"])
        assert "f1" in result.stdout

    def test_upload_data_requires_a_training_file(self, invoke, rl):
        rl(upload_data=UploadedData())
        result = invoke(["rl", "functions", "upload-data"])
        assert result.exit_code == 2

    def test_logs_reads_one_page(self, invoke, rl):
        resource = rl(
            query_function_instance_logs={"logs": ["a"], "total": 1},
        )
        result = invoke(
            ["rl", "functions", "logs", "i1", "--page", "2", "--size", "5"],
        )
        assert result.exit_code == 0
        assert resource.calls[0]["args"] == ("i1",)
        assert resource.calls[0]["kwargs"]["page_number"] == 2
        assert resource.calls[0]["kwargs"]["page_size"] == 5

    def test_logs_all_pages_prints_one_line_each(self, invoke, rl):
        resource = rl(
            query_all_function_instance_logs=["first line", "second line"],
        )
        result = invoke(["rl", "functions", "logs", "i1", "--all"])
        assert result.exit_code == 0
        assert (
            resource.calls[0]["method"] == "query_all_function_instance_logs"
        )
        assert result.stdout.splitlines() == ["first line", "second line"]

    def test_logs_all_pages_is_json_for_machines(self, invoke, rl):
        rl(query_all_function_instance_logs=["first line", "second line"])
        result = invoke(
            ["rl", "functions", "logs", "i1", "--all"],
            global_args=["-o", "json"],
        )
        assert json.loads(result.stdout) == ["first line", "second line"]

    def test_logs_passes_keywords(self, invoke, rl):
        resource = rl(query_function_instance_logs={})
        invoke(
            [
                "rl",
                "functions",
                "logs",
                "i1",
                "--keyword",
                "error",
                "--keyword",
                "traceback",
            ],
        )
        assert resource.calls[0]["kwargs"]["keywords"] == [
            "error",
            "traceback",
        ]

    def test_delete_with_yes(self, invoke, rl):
        resource = rl(delete_function_instance={"deleted": True})
        result = invoke(["rl", "functions", "delete", "i1", "--yes"])
        assert result.exit_code == 0
        assert resource.calls[0]["args"] == ("i1",)

    def test_delete_refuses_without_a_tty(self, invoke, rl):
        resource = rl(delete_function_instance={"deleted": True})
        result = invoke(["rl", "functions", "delete", "i1"])
        assert result.exit_code == 2
        assert "--yes" in result.stderr
        assert resource.calls == []
