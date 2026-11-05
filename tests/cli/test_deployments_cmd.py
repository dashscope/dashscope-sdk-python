# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope deployments`` -- create, inspect, scale, delete, stream."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from dashscope.cli.app import build_app
from dashscope.resources.deployments import (
    Deployment,
    DeploymentEvent,
    DeploymentList,
)
from dashscope.types.common import JobStatus


def _deployment(**overrides: Any) -> Deployment:
    data: Dict[str, Any] = {
        "deployed_model": "qwen-plus-ft-202601",
        "model": "qwen-plus-ft",
        "status": "RUNNING",
        "capacity": 2,
        "ready_capacity": 2,
        "created_at": "2026-01-01 12:00:00",
    }
    data.update(overrides)
    return Deployment(**data)


class _Deployments:
    """Stand-in for ``client.deployments`` recording every call."""

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

    def get(self, deployed_model, **kwargs):
        return self._record("get", deployed_model, **kwargs)

    def list(self, **kwargs):
        return self._record("list", **kwargs)

    def scale(self, deployed_model, **kwargs):
        return self._record("scale", deployed_model, **kwargs)

    def delete(self, deployed_model, **kwargs):
        return self._record("delete", deployed_model, **kwargs)

    def stream_events(self, deployed_model, **kwargs):
        self.calls.append(
            {
                "method": "stream_events",
                "args": (deployed_model,),
                "kwargs": kwargs,
            },
        )
        return iter(self._results.get("stream_events", []))


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def deployments(make_client):
    def install(**results: Any) -> _Deployments:
        resource = _Deployments(**results)
        make_client(deployments=resource)
        return resource

    return install


def _as_terminal_tty(monkeypatch, answer: bool):
    monkeypatch.setattr("dashscope.cli.confirm.stdin_is_tty", lambda: True)
    monkeypatch.setattr(
        "dashscope.cli.confirm.typer.confirm",
        lambda *a, **k: answer,
    )


class TestCreate:
    def test_create_passes_model_and_capacity(self, invoke, deployments):
        resource = deployments(create=_deployment())
        result = invoke(
            [
                "deployments",
                "create",
                "-m",
                "qwen-plus-ft",
                "--capacity",
                "2",
            ],
        )
        assert result.exit_code == 0
        kwargs = resource.calls[0]["kwargs"]
        assert kwargs["model"] == "qwen-plus-ft"
        assert kwargs["capacity"] == 2
        assert "qwen-plus-ft-202601" in result.stdout

    def test_create_accepts_a_deploy_spec(self, invoke, deployments):
        resource = deployments(create=_deployment())
        invoke(
            [
                "deployments",
                "create",
                "-m",
                "qwen-plus-ft",
                "--capacity",
                "2",
                "--deploy-spec",
                '{"gpu": "a10"}',
                "--billing-method",
                "monthly",
            ],
        )
        kwargs = resource.calls[0]["kwargs"]
        assert kwargs["deploy_spec"] == {"gpu": "a10"}
        assert kwargs["billing_method"] == "monthly"

    def test_create_rejects_a_non_object_deploy_spec(
        self, invoke, deployments
    ):
        deployments(create=_deployment())
        result = invoke(
            [
                "deployments",
                "create",
                "-m",
                "qwen-plus-ft",
                "--capacity",
                "2",
                "--deploy-spec",
                "[1,2]",
            ],
        )
        assert result.exit_code == 2

    def test_create_requires_capacity(self, invoke, deployments):
        deployments(create=_deployment())
        result = invoke(["deployments", "create", "-m", "qwen-plus-ft"])
        assert result.exit_code == 2


class TestRead:
    def test_get(self, invoke, deployments):
        resource = deployments(get=_deployment())
        result = invoke(["deployments", "get", "qwen-plus-ft-202601"])
        assert result.exit_code == 0
        assert resource.calls[0]["args"] == ("qwen-plus-ft-202601",)
        assert "RUNNING" in result.stdout

    def test_list_renders_rows(self, invoke, deployments):
        deployments(
            list=DeploymentList(
                deployments=[
                    _deployment(),
                    _deployment(deployed_model="other-202602"),
                ],
                page_no=1,
                page_size=10,
                total_count=2,
            ),
        )
        result = invoke(["deployments", "list"])
        assert result.exit_code == 0
        assert "qwen-plus-ft-202601" in result.stdout
        assert "other-202602" in result.stdout

    def test_list_pagination_flags(self, invoke, deployments):
        resource = deployments(
            list=DeploymentList(deployments=[], page_no=2, page_size=5),
        )
        invoke(["deployments", "list", "--page", "2", "--size", "5"])
        assert resource.calls[0]["kwargs"] == {"page_no": 2, "page_size": 5}

    def test_list_json_output_is_the_envelope(self, invoke, deployments):
        deployments(
            list=DeploymentList(
                deployments=[_deployment()],
                page_no=1,
                page_size=10,
                total_count=1,
            ),
        )
        result = invoke(["deployments", "list"], global_args=["-o", "json"])
        doc = json.loads(result.stdout)
        assert doc["total_count"] == 1
        assert len(doc["deployments"]) == 1


class TestScale:
    def test_scale_with_yes(self, invoke, deployments):
        resource = deployments(scale=_deployment(capacity=4))
        result = invoke(
            [
                "deployments",
                "scale",
                "qwen-plus-ft-202601",
                "--capacity",
                "4",
                "--yes",
            ],
        )
        assert result.exit_code == 0
        assert resource.calls[0]["args"] == ("qwen-plus-ft-202601",)
        assert resource.calls[0]["kwargs"] == {"capacity": 4}

    def test_scale_refuses_without_a_tty(self, invoke, deployments):
        resource = deployments(scale=_deployment(capacity=4))
        result = invoke(
            [
                "deployments",
                "scale",
                "qwen-plus-ft-202601",
                "--capacity",
                "4",
            ],
        )
        assert result.exit_code == 2
        assert "--yes" in result.stderr
        assert resource.calls == []

    def test_scale_confirms_in_a_tty(self, invoke, deployments, monkeypatch):
        _as_terminal_tty(monkeypatch, answer=True)
        resource = deployments(scale=_deployment(capacity=4))
        result = invoke(
            [
                "deployments",
                "scale",
                "qwen-plus-ft-202601",
                "--capacity",
                "4",
            ],
        )
        assert result.exit_code == 0
        assert len(resource.calls) == 1


class TestDelete:
    def test_delete_with_yes(self, invoke, deployments):
        resource = deployments(delete=JobStatus())
        result = invoke(
            ["deployments", "delete", "qwen-plus-ft-202601", "--yes"],
        )
        assert result.exit_code == 0
        assert resource.calls[0]["args"] == ("qwen-plus-ft-202601",)

    def test_delete_refuses_without_a_tty(self, invoke, deployments):
        resource = deployments(delete=JobStatus())
        result = invoke(["deployments", "delete", "qwen-plus-ft-202601"])
        assert result.exit_code == 2
        assert "--yes" in result.stderr
        assert resource.calls == []

    def test_delete_aborts_on_no(self, invoke, deployments, monkeypatch):
        _as_terminal_tty(monkeypatch, answer=False)
        resource = deployments(delete=JobStatus())
        result = invoke(["deployments", "delete", "qwen-plus-ft-202601"])
        assert result.exit_code != 0
        assert resource.calls == []


class TestStream:
    def test_stream_echoes_each_event(self, invoke, deployments):
        resource = deployments(
            stream_events=[
                DeploymentEvent(output={"status": "DEPLOYING"}),
                DeploymentEvent(output={"status": "RUNNING"}),
            ],
        )
        result = invoke(["deployments", "stream", "qwen-plus-ft-202601"])
        assert result.exit_code == 0
        assert resource.calls[0]["args"] == ("qwen-plus-ft-202601",)
        lines = [line for line in result.stdout.splitlines() if line]
        assert lines == ['{"status":"DEPLOYING"}', '{"status":"RUNNING"}']

    def test_stream_emits_ndjson_for_machines(self, invoke, deployments):
        deployments(
            stream_events=[
                DeploymentEvent(output={"status": "DEPLOYING"}),
                DeploymentEvent(output={"status": "RUNNING"}),
            ],
        )
        result = invoke(
            ["deployments", "stream", "qwen-plus-ft-202601"],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        docs = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        assert [doc["output"]["status"] for doc in docs] == [
            "DEPLOYING",
            "RUNNING",
        ]
