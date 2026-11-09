# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope embeddings`` -- create and jobs subgroup."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from dashscope import errors as sdk_errors
from dashscope.cli.app import build_app
from dashscope.resources._jobs import Job, JobList
from dashscope.resources.embeddings import Embedding, EmbeddingResponse
from dashscope.types.common import Usage


def _embedding_response(**overrides: Any) -> EmbeddingResponse:
    data: Dict[str, Any] = {
        "embeddings": [
            Embedding(index=0, embedding=[0.1, 0.2, 0.3]),
        ],
        "usage": Usage(input_tokens=5, total_tokens=5),
    }
    data.update(overrides)
    return EmbeddingResponse(**data)


def _job(**overrides: Any) -> Job:
    data: Dict[str, Any] = {
        "task_id": "task-abc123",
        "status": "SUCCEEDED",
        "model": "text-embedding-v3",
        "created_at": "2026-01-01 12:00:00",
    }
    data.update(overrides)
    return Job(**data)


class _EmbeddingJobs:
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

    def fetch(self, task_id, **kwargs):
        return self._record("fetch", task_id, **kwargs)

    def list(self, **kwargs):
        return self._record("list", **kwargs)

    def cancel(self, task_id, **kwargs):
        return self._record("cancel", task_id, **kwargs)

    def wait(self, job, **kwargs):
        self._record("wait", job, **kwargs)
        return self._results.get("wait", job)


class _Embeddings:
    def __init__(self, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results
        self.jobs = _EmbeddingJobs(**results.get("jobs", {}))

    def create(self, **kwargs):
        self.calls.append({"method": "create", "kwargs": kwargs})
        result = self._results.get("create")
        if isinstance(result, Exception):
            raise result
        return result() if callable(result) else result


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def embeddings(make_client):
    def install(**results: Any) -> _Embeddings:
        resource = _Embeddings(**results)
        make_client(embeddings=resource)
        return resource

    return install


class TestCreate:
    def test_create_single_text(self, invoke, embeddings):
        resource = embeddings(create=_embedding_response())
        result = invoke(
            [
                "embeddings",
                "create",
                "-m",
                "text-embedding-v3",
                "--input",
                "hello",
            ],
        )
        assert result.exit_code == 0
        assert resource.calls[0]["kwargs"]["input"] == "hello"

    def test_create_table_output_summarises(self, invoke, embeddings):
        embeddings(create=_embedding_response())
        result = invoke(
            [
                "embeddings",
                "create",
                "-m",
                "text-embedding-v3",
                "--input",
                "hello",
            ],
        )
        assert result.stdout.strip() == "1 embedding(s), 3 dimensions"
        # The counters go to stderr, so stdout stays a one-line summary.
        assert "'input_tokens': 5" in result.stderr

    def test_create_json_array_input(self, invoke, embeddings):
        resource = embeddings(create=_embedding_response())
        invoke(
            [
                "embeddings",
                "create",
                "-m",
                "text-embedding-v3",
                "--input",
                '["hello", "world"]',
            ],
        )
        assert resource.calls[0]["kwargs"]["input"] == ["hello", "world"]

    def test_create_with_options(self, invoke, embeddings):
        resource = embeddings(create=_embedding_response())
        invoke(
            [
                "embeddings",
                "create",
                "-m",
                "text-embedding-v3",
                "--input",
                "hello",
                "--text-type",
                "query",
                "--dimension",
                "512",
            ],
        )
        kwargs = resource.calls[0]["kwargs"]
        assert kwargs["text_type"] == "query"
        assert kwargs["dimension"] == 512

    def test_create_json_output(self, invoke, embeddings):
        embeddings(create=_embedding_response())
        result = invoke(
            [
                "embeddings",
                "create",
                "-m",
                "text-embedding-v3",
                "--input",
                "hi",
            ],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        doc = json.loads(result.stdout)
        assert len(doc["embeddings"]) == 1

    def test_create_requires_input(self, invoke, embeddings):
        embeddings(create=_embedding_response())
        result = invoke(["embeddings", "create", "-m", "text-embedding-v3"])
        assert result.exit_code == 2


class TestJobs:
    def test_jobs_create(self, invoke, embeddings):
        resource = embeddings(jobs={"create": _job()})
        result = invoke(
            [
                "embeddings",
                "jobs",
                "create",
                "-m",
                "text-embedding-v3",
                "--file-url",
                "https://example.com/texts.txt",
            ],
        )
        assert result.exit_code == 0
        kwargs = resource.jobs.calls[0]["kwargs"]
        assert kwargs["file_url"] == "https://example.com/texts.txt"

    def test_jobs_get(self, invoke, embeddings):
        resource = embeddings(jobs={"fetch": _job()})
        invoke(["embeddings", "jobs", "get", "task-abc123"])
        assert resource.jobs.calls[0]["args"] == ("task-abc123",)

    def test_jobs_list(self, invoke, embeddings):
        embeddings(
            jobs={
                "list": JobList(
                    tasks=[_job()],
                    page_no=1,
                    page_size=10,
                    total=1,
                ),
            },
        )
        result = invoke(["embeddings", "jobs", "list"])
        assert result.exit_code == 0
        assert "task-abc123" in result.stdout

    def test_jobs_list_reads_the_real_envelope_fields(
        self,
        invoke,
        embeddings,
    ):
        """``JobList`` spells them ``tasks``/``total``, not ``jobs``/``total_count``."""
        embeddings(jobs={"list": JobList(tasks=[_job()], total=1)})
        result = invoke(["embeddings", "jobs", "list"])
        assert result.exit_code == 0
        assert "total 1" in result.stderr

    def test_jobs_cancel(self, invoke, embeddings):
        resource = embeddings(jobs={"cancel": _job(status="CANCELED")})
        invoke(["embeddings", "jobs", "cancel", "task-abc123", "--yes"])
        assert resource.jobs.calls[0]["args"] == ("task-abc123",)

    def test_jobs_cancel_refuses_without_a_tty(self, invoke, embeddings):
        embeddings(jobs={"cancel": _job(status="CANCELED")})
        result = invoke(["embeddings", "jobs", "cancel", "task-abc123"])
        assert result.exit_code == 2
        assert "--yes" in result.stderr

    def test_jobs_wait_polls_the_id_and_nothing_else(self, invoke, embeddings):
        resource = embeddings(jobs={"wait": _job()})
        result = invoke(["embeddings", "jobs", "wait", "task-abc123"])
        assert result.exit_code == 0
        assert [c["method"] for c in resource.jobs.calls] == ["wait"]
        assert resource.jobs.calls[0]["args"] == ("task-abc123",)

    def test_jobs_wait_defaults_to_an_hour(self, invoke, embeddings):
        resource = embeddings(jobs={"wait": _job()})
        invoke(["embeddings", "jobs", "wait", "task-abc123"])
        assert resource.jobs.calls[0]["kwargs"] == {"timeout": 3600.0}

    def test_jobs_wait_table_shows_the_settled_task(self, invoke, embeddings):
        embeddings(jobs={"wait": _job(status="SUCCEEDED")})
        result = invoke(["embeddings", "jobs", "wait", "task-abc123"])
        assert "SUCCEEDED" in result.stdout

    def test_jobs_wait_json_output_is_the_whole_job(self, invoke, embeddings):
        embeddings(jobs={"wait": _job()})
        result = invoke(
            ["embeddings", "jobs", "wait", "task-abc123"],
            global_args=["-o", "json"],
        )
        doc = json.loads(result.stdout)
        assert doc["task_id"] == "task-abc123"
        assert doc["status"] == "SUCCEEDED"

    def test_jobs_wait_failure_exits_nonzero(self, invoke, embeddings):
        embeddings(
            jobs={"wait": sdk_errors.InternalServerError("task exploded")},
        )
        result = invoke(["embeddings", "jobs", "wait", "task-abc123"])
        assert result.exit_code == 1
        assert "Error [server]: task exploded" in result.stderr

    def test_jobs_wait_requires_the_id(self, invoke, embeddings):
        resource = embeddings(jobs={"wait": _job()})
        assert invoke(["embeddings", "jobs", "wait"]).exit_code == 2
        assert resource.jobs.calls == []

    @pytest.mark.parametrize("timeout", ["0", "nan"])
    def test_jobs_wait_rejects_a_timeout_it_could_not_honour(
        self,
        invoke,
        embeddings,
        timeout,
    ):
        """``nan`` is false against every comparison, so the poll never ends."""
        resource = embeddings(jobs={"wait": _job()})
        result = invoke(
            [
                "embeddings",
                "jobs",
                "wait",
                "task-abc123",
                "--timeout",
                timeout,
            ],
        )
        assert result.exit_code == 2
        assert "--timeout" in result.stderr
        assert resource.jobs.calls == []
