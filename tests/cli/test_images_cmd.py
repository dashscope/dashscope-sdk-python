# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope images`` -- generate and the jobs subgroup."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from dashscope import errors as sdk_errors
from dashscope.cli.app import build_app
from dashscope.resources._jobs import Job, JobList


def _job(**overrides: Any) -> Job:
    data: Dict[str, Any] = {
        "task_id": "task-abc123",
        "status": "SUCCEEDED",
        "submitted_at": "2026-01-01 12:00:00",
        "output": {"results": [{"url": "https://example.com/cat.png"}]},
    }
    data.update(overrides)
    return Job(**data)


class _ImageJobs:
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


class _Images:
    """Stand-in for ``client.images`` recording every call."""

    def __init__(self, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results
        self.jobs = _ImageJobs(**results.get("jobs", {}))

    def generate(self, **kwargs):
        self.calls.append({"method": "generate", "kwargs": kwargs})
        result = self._results.get("generate")
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
def images(make_client):
    def install(**results: Any) -> _Images:
        resource = _Images(**results)
        make_client(images=resource)
        return resource

    return install


class TestGenerate:
    def test_generate_prints_one_url_per_line(self, invoke, images):
        images(
            generate=_job(
                output={
                    "results": [
                        {"url": "https://example.com/a.png"},
                        {"url": "https://example.com/b.png"},
                    ],
                },
            ),
        )
        result = invoke(
            ["images", "generate", "-m", "wan2.2-t2i-flash", "-p", "a cat"],
        )
        assert result.exit_code == 0
        assert result.stdout.split() == [
            "https://example.com/a.png",
            "https://example.com/b.png",
        ]

    def test_generate_passes_the_payload(self, invoke, images):
        resource = images(generate=_job())
        invoke(
            [
                "images",
                "generate",
                "-m",
                "wan2.2-t2i-flash",
                "-p",
                "a cat",
                "--negative-prompt",
                "blurry",
                "-n",
                "2",
                "--size",
                "1024x1024",
                "--seed",
                "7",
                "--no-watermark",
            ],
        )
        kwargs = resource.calls[0]["kwargs"]
        assert kwargs["model"] == "wan2.2-t2i-flash"
        assert kwargs["prompt"] == "a cat"
        assert kwargs["negative_prompt"] == "blurry"
        assert kwargs["n"] == 2
        assert kwargs["size"] == "1024x1024"
        assert kwargs["seed"] == 7
        assert kwargs["watermark"] is False

    def test_generate_collects_repeatable_reference_images(
        self,
        invoke,
        images,
    ):
        resource = images(generate=_job())
        invoke(
            [
                "images",
                "generate",
                "-m",
                "wan2.6-image",
                "-p",
                "a cat",
                "--image-url",
                "https://example.com/a.png",
                "--image-url",
                "b.png",
            ],
        )
        assert resource.calls[0]["kwargs"]["image_urls"] == [
            "https://example.com/a.png",
            "b.png",
        ]

    def test_generate_reads_a_prompt_from_a_file(
        self,
        invoke,
        images,
        tmp_path,
    ):
        payload = tmp_path / "prompt.txt"
        payload.write_text("a very long prompt\n", encoding="utf-8")
        resource = images(generate=_job())
        invoke(
            [
                "images",
                "generate",
                "-m",
                "wan2.2-t2i-flash",
                "-p",
                f"@{payload}",
            ],
        )
        assert resource.calls[0]["kwargs"]["prompt"] == "a very long prompt\n"

    def test_generate_json_output_keeps_the_whole_job(self, invoke, images):
        images(generate=_job())
        result = invoke(
            ["images", "generate", "-m", "wan2.2-t2i-flash", "-p", "a cat"],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        doc = json.loads(result.stdout)
        assert doc["task_id"] == "task-abc123"
        assert (
            doc["output"]["results"][0]["url"] == "https://example.com/cat.png"
        )

    def test_generate_requires_a_prompt(self, invoke, images):
        images(generate=_job())
        result = invoke(["images", "generate", "-m", "wan2.2-t2i-flash"])
        assert result.exit_code == 2

    def test_generate_refuses_an_empty_prompt(self, invoke, images):
        images(generate=_job())
        result = invoke(
            ["images", "generate", "-m", "wan2.2-t2i-flash", "-p", "   "],
        )
        assert result.exit_code == 2
        assert "--prompt" in result.stderr


class TestJobs:
    def test_jobs_create_submits_without_waiting(self, invoke, images):
        resource = images(
            jobs={"create": _job(status="PENDING", output=None)},
        )
        result = invoke(
            [
                "images",
                "jobs",
                "create",
                "-m",
                "wan2.2-t2i-flash",
                "-p",
                "a cat",
            ],
        )
        assert result.exit_code == 0
        assert "task-abc123" in result.stdout
        assert [c["method"] for c in resource.jobs.calls] == ["create"]

    def test_jobs_create_wait_blocks(self, invoke, images):
        resource = images(
            jobs={
                "create": _job(status="PENDING", output=None),
                "wait": _job(),
            },
        )
        invoke(
            [
                "images",
                "jobs",
                "create",
                "-m",
                "wan2.2-t2i-flash",
                "-p",
                "a cat",
                "--wait",
                "--timeout",
                "30",
            ],
        )
        methods = [c["method"] for c in resource.jobs.calls]
        assert methods == ["create", "wait"]
        assert resource.jobs.calls[1]["kwargs"]["timeout"] == 30.0

    def test_jobs_get(self, invoke, images):
        resource = images(jobs={"fetch": _job()})
        invoke(["images", "jobs", "get", "task-abc123"])
        assert resource.jobs.calls[0]["args"] == ("task-abc123",)

    def test_jobs_list(self, invoke, images):
        images(
            jobs={
                "list": JobList(
                    tasks=[_job(), _job(task_id="task-def456")],
                    page_no=1,
                    page_size=10,
                    total=2,
                ),
            },
        )
        result = invoke(["images", "jobs", "list"])
        assert result.exit_code == 0
        assert "task-abc123" in result.stdout
        assert "task-def456" in result.stdout

    def test_jobs_list_pagination_and_filters(self, invoke, images):
        resource = images(jobs={"list": JobList(tasks=[], total=0)})
        invoke(
            [
                "images",
                "jobs",
                "list",
                "--page",
                "2",
                "--size",
                "5",
                "--status",
                "SUCCEEDED",
                "--model",
                "wan2.2-t2i-flash",
                "--api-key-id",
                "key-1",
                "--region",
                "cn-beijing",
            ],
        )
        assert resource.jobs.calls[0]["kwargs"] == {
            "page_no": 2,
            "page_size": 5,
            "status": "SUCCEEDED",
            "model_name": "wan2.2-t2i-flash",
            "api_key_id": "key-1",
            "region": "cn-beijing",
            "start_time": None,
            "end_time": None,
        }

    def test_jobs_list_json_output_is_the_envelope(self, invoke, images):
        images(jobs={"list": JobList(tasks=[_job()], total=1)})
        result = invoke(["images", "jobs", "list"], global_args=["-o", "json"])
        doc = json.loads(result.stdout)
        assert doc["total_count"] == 1
        assert len(doc["tasks"]) == 1

    def test_jobs_cancel_with_yes(self, invoke, images):
        resource = images(jobs={"cancel": _job(status="CANCELED")})
        result = invoke(["images", "jobs", "cancel", "task-abc123", "--yes"])
        assert result.exit_code == 0
        assert resource.jobs.calls[0]["args"] == ("task-abc123",)

    def test_jobs_cancel_refuses_without_a_tty(self, invoke, images):
        resource = images(jobs={"cancel": _job(status="CANCELED")})
        result = invoke(["images", "jobs", "cancel", "task-abc123"])
        assert result.exit_code == 2
        assert "--yes" in result.stderr
        assert resource.jobs.calls == []

    def test_jobs_wait_polls_the_id_and_nothing_else(self, invoke, images):
        resource = images(jobs={"wait": _job()})
        assert invoke(["images", "jobs", "wait", "task-abc123"]).exit_code == 0
        assert [c["method"] for c in resource.jobs.calls] == ["wait"]
        assert resource.jobs.calls[0]["args"] == ("task-abc123",)

    def test_jobs_wait_defaults_to_an_hour(self, invoke, images):
        resource = images(jobs={"wait": _job()})
        invoke(["images", "jobs", "wait", "task-abc123"])
        assert resource.jobs.calls[0]["kwargs"] == {"timeout": 3600.0}

    def test_jobs_wait_prints_the_urls_like_generate_does(
        self,
        invoke,
        images,
    ):
        """The point of waiting separately is usually to collect the URL."""
        images(jobs={"wait": _job()})
        result = invoke(["images", "jobs", "wait", "task-abc123"])
        assert result.stdout.strip() == "https://example.com/cat.png"

    def test_jobs_wait_json_output_is_the_whole_job(self, invoke, images):
        images(jobs={"wait": _job()})
        result = invoke(
            ["images", "jobs", "wait", "task-abc123"],
            global_args=["-o", "json"],
        )
        doc = json.loads(result.stdout)
        assert doc["task_id"] == "task-abc123"
        assert doc["status"] == "SUCCEEDED"

    def test_jobs_wait_failure_exits_nonzero(self, invoke, images):
        images(
            jobs={"wait": sdk_errors.InternalServerError("task exploded")},
        )
        result = invoke(["images", "jobs", "wait", "task-abc123"])
        assert result.exit_code == 1
        assert "Error [server]: task exploded" in result.stderr

    def test_jobs_wait_requires_the_id(self, invoke, images):
        resource = images(jobs={"wait": _job()})
        assert invoke(["images", "jobs", "wait"]).exit_code == 2
        assert resource.jobs.calls == []

    @pytest.mark.parametrize("timeout", ["0", "nan"])
    def test_jobs_wait_rejects_a_timeout_it_could_not_honour(
        self,
        invoke,
        images,
        timeout,
    ):
        """``nan`` is false against every comparison, so the poll never ends."""
        resource = images(jobs={"wait": _job()})
        result = invoke(
            ["images", "jobs", "wait", "task-abc123", "--timeout", timeout],
        )
        assert result.exit_code == 2
        assert "--timeout" in result.stderr
        assert resource.jobs.calls == []
