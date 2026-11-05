# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope videos`` -- generate and the jobs subgroup."""

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
        "output": {"video_url": "https://example.com/cat.mp4"},
    }
    data.update(overrides)
    return Job(**data)


class _VideoJobs:
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


class _Videos:
    """Stand-in for ``client.videos`` recording every call."""

    def __init__(self, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results
        self.jobs = _VideoJobs(**results.get("jobs", {}))

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
def videos(make_client):
    def install(**results: Any) -> _Videos:
        resource = _Videos(**results)
        make_client(videos=resource)
        return resource

    return install


class TestGenerate:
    def test_generate_prints_the_video_url(self, invoke, videos):
        videos(generate=_job())
        result = invoke(
            ["videos", "generate", "-m", "wan2.2-t2v-plus", "-p", "a cat"],
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "https://example.com/cat.mp4"

    def test_generate_falls_back_to_result_url(self, invoke, videos):
        videos(
            generate=_job(
                output={"results": [{"url": "https://example.com/a.mp4"}]},
            ),
        )
        result = invoke(
            ["videos", "generate", "-m", "wan2.2-t2v-plus", "-p", "a cat"],
        )
        assert result.stdout.strip() == "https://example.com/a.mp4"

    def test_generate_passes_the_payload(self, invoke, videos):
        resource = videos(generate=_job())
        invoke(
            [
                "videos",
                "generate",
                "-m",
                "wan2.2-t2v-plus",
                "-p",
                "a cat",
                "--duration",
                "5",
                "--resolution",
                "1080P",
                "--ratio",
                "16:9",
                "--seed",
                "7",
                "--no-watermark",
                "--prompt-extend",
            ],
        )
        kwargs = resource.calls[0]["kwargs"]
        assert kwargs["model"] == "wan2.2-t2v-plus"
        assert kwargs["prompt"] == "a cat"
        assert kwargs["duration"] == 5
        assert kwargs["resolution"] == "1080P"
        assert kwargs["ratio"] == "16:9"
        assert kwargs["seed"] == 7
        assert kwargs["watermark"] is False
        assert kwargs["prompt_extend"] is True

    def test_generate_defaults_to_an_hour_of_waiting(self, invoke, videos):
        resource = videos(generate=_job())
        invoke(["videos", "generate", "-m", "wan2.2-t2v-plus", "-p", "a cat"])
        assert resource.calls[0]["kwargs"]["timeout"] == 3600.0

    def test_generate_accepts_an_image_url_alone(self, invoke, videos):
        resource = videos(generate=_job())
        result = invoke(
            [
                "videos",
                "generate",
                "-m",
                "wan2.2-i2v",
                "--image-url",
                "https://example.com/a.png",
            ],
        )
        assert result.exit_code == 0
        assert resource.calls[0]["kwargs"]["image_url"] == (
            "https://example.com/a.png"
        )

    def test_generate_passes_media_through(self, invoke, videos):
        resource = videos(generate=_job())
        invoke(
            [
                "videos",
                "generate",
                "-m",
                "wan2.7-i2v",
                "--media",
                '[{"first_frame": "a.png"}, {"last_frame": "b.png"}]',
            ],
        )
        assert resource.calls[0]["kwargs"]["media"] == [
            {"first_frame": "a.png"},
            {"last_frame": "b.png"},
        ]

    def test_generate_rejects_media_that_is_not_an_array(self, invoke, videos):
        videos(generate=_job())
        result = invoke(
            [
                "videos",
                "generate",
                "-m",
                "wan2.7-i2v",
                "--media",
                '{"first_frame": "a.png"}',
            ],
        )
        assert result.exit_code == 2
        assert "--media" in result.stderr

    def test_generate_rejects_a_media_entry_that_is_not_an_object(
        self,
        invoke,
        videos,
    ):
        videos(generate=_job())
        result = invoke(
            ["videos", "generate", "-m", "wan2.7-i2v", "--media", '["a.png"]'],
        )
        assert result.exit_code == 2
        assert "entry 0" in result.stderr

    def test_generate_needs_something_to_generate_from(self, invoke, videos):
        resource = videos(generate=_job())
        result = invoke(["videos", "generate", "-m", "wan2.2-t2v-plus"])
        assert result.exit_code == 2
        assert "--prompt" in result.stderr
        assert resource.calls == []

    def test_generate_json_output_keeps_the_whole_job(self, invoke, videos):
        videos(generate=_job())
        result = invoke(
            ["videos", "generate", "-m", "wan2.2-t2v-plus", "-p", "a cat"],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        doc = json.loads(result.stdout)
        assert doc["task_id"] == "task-abc123"
        assert doc["output"]["video_url"] == "https://example.com/cat.mp4"

    def test_generate_reports_a_still_running_job_by_its_state(
        self,
        invoke,
        videos,
    ):
        videos(generate=_job(status="RUNNING", output=None))
        result = invoke(
            ["videos", "generate", "-m", "wan2.2-t2v-plus", "-p", "a cat"],
        )
        assert result.exit_code == 0
        assert "RUNNING" in result.stdout


class TestJobs:
    def test_jobs_create_submits_without_waiting(self, invoke, videos):
        resource = videos(
            jobs={"create": _job(status="PENDING", output=None)},
        )
        result = invoke(
            [
                "videos",
                "jobs",
                "create",
                "-m",
                "wan2.2-t2v-plus",
                "-p",
                "a cat",
            ],
        )
        assert result.exit_code == 0
        assert "task-abc123" in result.stdout
        assert [c["method"] for c in resource.jobs.calls] == ["create"]

    def test_jobs_create_wait_blocks(self, invoke, videos):
        resource = videos(
            jobs={
                "create": _job(status="PENDING", output=None),
                "wait": _job(),
            },
        )
        result = invoke(
            [
                "videos",
                "jobs",
                "create",
                "-m",
                "wan2.2-t2v-plus",
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
        assert result.stdout.strip() == "https://example.com/cat.mp4"

    def test_jobs_create_needs_something_to_generate_from(
        self,
        invoke,
        videos,
    ):
        resource = videos(jobs={"create": _job()})
        result = invoke(["videos", "jobs", "create", "-m", "wan2.2-t2v-plus"])
        assert result.exit_code == 2
        assert resource.jobs.calls == []

    def test_jobs_get(self, invoke, videos):
        resource = videos(jobs={"fetch": _job()})
        invoke(["videos", "jobs", "get", "task-abc123"])
        assert resource.jobs.calls[0]["args"] == ("task-abc123",)

    def test_jobs_list(self, invoke, videos):
        videos(
            jobs={
                "list": JobList(
                    tasks=[_job(), _job(task_id="task-def456")],
                    page_no=1,
                    page_size=10,
                    total=2,
                ),
            },
        )
        result = invoke(["videos", "jobs", "list"])
        assert result.exit_code == 0
        assert "task-abc123" in result.stdout
        assert "task-def456" in result.stdout
        assert "total 2" in result.stderr

    def test_jobs_list_reads_the_real_envelope_fields(self, invoke, videos):
        videos(jobs={"list": JobList(tasks=[_job()], total=1, page_size=10)})
        result = invoke(["videos", "jobs", "list"])
        assert "total 1" in result.stderr

    def test_jobs_list_pagination_and_filters(self, invoke, videos):
        resource = videos(jobs={"list": JobList(tasks=[], total=0)})
        invoke(
            [
                "videos",
                "jobs",
                "list",
                "--page",
                "2",
                "--size",
                "5",
                "--status",
                "SUCCEEDED",
                "--model",
                "wan2.2-t2v-plus",
                "--start-time",
                "2026-01-01",
                "--end-time",
                "2026-02-01",
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
            "model_name": "wan2.2-t2v-plus",
            "api_key_id": "key-1",
            "region": "cn-beijing",
            "start_time": "2026-01-01",
            "end_time": "2026-02-01",
        }

    def test_jobs_list_json_output_is_the_envelope(self, invoke, videos):
        videos(jobs={"list": JobList(tasks=[_job()], total=1)})
        result = invoke(["videos", "jobs", "list"], global_args=["-o", "json"])
        doc = json.loads(result.stdout)
        assert doc["total"] == 1
        assert len(doc["tasks"]) == 1

    def test_jobs_cancel_with_yes(self, invoke, videos):
        resource = videos(jobs={"cancel": _job(status="CANCELED")})
        result = invoke(["videos", "jobs", "cancel", "task-abc123", "--yes"])
        assert result.exit_code == 0
        assert resource.jobs.calls[0]["args"] == ("task-abc123",)

    def test_jobs_cancel_refuses_without_a_tty(self, invoke, videos):
        resource = videos(jobs={"cancel": _job(status="CANCELED")})
        result = invoke(["videos", "jobs", "cancel", "task-abc123"])
        assert result.exit_code == 2
        assert "--yes" in result.stderr
        assert resource.jobs.calls == []

    def test_jobs_wait_polls_the_id_and_nothing_else(self, invoke, videos):
        resource = videos(jobs={"wait": _job()})
        assert invoke(["videos", "jobs", "wait", "task-abc123"]).exit_code == 0
        assert [c["method"] for c in resource.jobs.calls] == ["wait"]
        assert resource.jobs.calls[0]["args"] == ("task-abc123",)

    def test_jobs_wait_defaults_to_an_hour(self, invoke, videos):
        resource = videos(jobs={"wait": _job()})
        invoke(["videos", "jobs", "wait", "task-abc123"])
        assert resource.jobs.calls[0]["kwargs"] == {"timeout": 3600.0}

    def test_jobs_wait_prints_the_url_like_generate_does(self, invoke, videos):
        """`url=$(dashscope videos jobs wait ...)` is the whole point."""
        videos(jobs={"wait": _job()})
        result = invoke(["videos", "jobs", "wait", "task-abc123"])
        assert result.stdout.strip() == "https://example.com/cat.mp4"

    def test_jobs_wait_json_output_is_the_whole_job(self, invoke, videos):
        videos(jobs={"wait": _job()})
        result = invoke(
            ["videos", "jobs", "wait", "task-abc123"],
            global_args=["-o", "json"],
        )
        doc = json.loads(result.stdout)
        assert doc["task_id"] == "task-abc123"
        assert doc["output"]["video_url"] == "https://example.com/cat.mp4"

    def test_jobs_wait_failure_exits_nonzero(self, invoke, videos):
        videos(jobs={"wait": sdk_errors.InternalServerError("task exploded")})
        result = invoke(["videos", "jobs", "wait", "task-abc123"])
        assert result.exit_code == 1
        assert "Error [server]: task exploded" in result.stderr

    def test_jobs_wait_requires_the_id(self, invoke, videos):
        resource = videos(jobs={"wait": _job()})
        assert invoke(["videos", "jobs", "wait"]).exit_code == 2
        assert resource.jobs.calls == []

    @pytest.mark.parametrize("timeout", ["0", "nan"])
    def test_jobs_wait_rejects_a_timeout_it_could_not_honour(
        self,
        invoke,
        videos,
        timeout,
    ):
        """``nan`` is false against every comparison, so the poll never ends."""
        resource = videos(jobs={"wait": _job()})
        result = invoke(
            ["videos", "jobs", "wait", "task-abc123", "--timeout", timeout],
        )
        assert result.exit_code == 2
        assert "--timeout" in result.stderr
        assert resource.jobs.calls == []
