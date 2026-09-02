# -*- coding: utf-8 -*-
"""Managed Agent deployment resource tests."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from dashscope.agentstudio import AsyncClient, Client, user_message
from dashscope.agentstudio.transport import APIResponse
from dashscope.agentstudio.types import (
    Agent,
    DeploymentAgentReference,
    DeploymentError,
    DeploymentPausedReason,
    DeploymentResource,
    DeploymentSchedule,
)


def _deployment_payload() -> Dict[str, Any]:
    return {
        "id": "depl_01",
        "type": "deployment",
        "name": "daily-summary",
        "description": "Summarize orders",
        "agent": {
            "id": "agent_01",
            "type": "agent",
            "version": 12,
            "name": "data-analyst",
            "model": {"id": "qwen3-max"},
            "system": "Analyze data.",
        },
        "environment_id": "env_01",
        "schedule": {
            "type": "cron",
            "expression": "0 9 * * 1-5",
            "timezone": "Asia/Shanghai",
            "next_run_at": "2026-07-28T01:00:00Z",
        },
        "initial_events": [user_message("Summarize yesterday's orders")],
        "resources": [
            {
                "type": "file",
                "file_id": "file_01",
                "mount_path": "/mnt/data",
            },
        ],
        "vault_ids": ["vault_01"],
        "metadata": {"biz": "summary"},
        "status": "active",
        "paused_reason": {
            "type": "error",
            "error": {"code": "RUN_FAILED", "message": "failed"},
        },
        "created_at": "2026-07-27T01:00:00Z",
        "updated_at": "2026-07-27T01:00:00Z",
        "request_id": "req_01",
    }


def _run_payload() -> Dict[str, Any]:
    return {
        "id": "drun_01",
        "type": "deployment_run",
        "deployment_id": "depl_01",
        "agent": {"id": "agent_01", "version": 12},
        "session_id": "session_01",
        "trigger_source": "manual",
        "status": "failed",
        "error": {"code": "RUN_FAILED", "message": "failed"},
        "started_at": "2026-07-27T01:00:00.123Z",
        "finished_at": "2026-07-27T01:01:00.456Z",
        "request_id": "req_02",
    }


def _record_response(
    calls: List[Dict[str, Any]],
    method: str,
    path: str,
    kwargs: Dict[str, Any],
) -> APIResponse:
    calls.append({"method": method, "path": path, **kwargs})
    params = kwargs.get("params") or {}
    if path == "/deployments" and method == "GET":
        return APIResponse(
            data={
                "data": [_deployment_payload()],
                "next_page": None if params.get("page") else "next-depl",
            },
            request_id="req_list",
        )
    if path.endswith("/runs") or path == "/deployment_runs":
        return APIResponse(
            data={
                "data": [_run_payload()],
                "next_page": None if params.get("page") else "next-run",
            },
            request_id="req_runs",
        )
    if path.startswith("/deployment_runs/") or path.endswith("/run"):
        return APIResponse(data=_run_payload(), request_id="req_02")
    return APIResponse(data=_deployment_payload(), request_id="req_01")


class _RecordingTransport:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def request(self, method: str, path: str, **kwargs: Any) -> APIResponse:
        return _record_response(self.calls, method, path, kwargs)

    def close(self) -> None:
        pass


class _AsyncRecordingTransport:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> APIResponse:
        return _record_response(self.calls, method, path, kwargs)

    async def aclose(self) -> None:
        pass


@pytest.fixture(name="client")
def _client_fixture() -> Client:
    client = Client(api_key="test-key", base_url="http://test")
    client.transport.close()
    client.transport = _RecordingTransport()
    return client


def test_create_serializes_contract_and_hydrates_nested_models(client: Client):
    deployment = client.deployments.create(
        name="daily-summary",
        description="Summarize orders",
        agent={"id": "agent_01", "version": 12},
        environment_id="env_01",
        schedule={
            "type": "cron",
            "expression": "0 9 * * 1-5",
            "timezone": "Asia/Shanghai",
        },
        initial_events=[user_message("Summarize yesterday's orders")],
        resources=[
            {
                "type": "file",
                "file_id": "file_01",
                "mount_path": "/mnt/data",
            },
        ],
        vault_ids=["vault_01"],
        metadata={"biz": "summary"},
    )

    call = client.transport.calls[-1]
    assert call["method"] == "POST"
    assert call["path"] == "/deployments"
    assert call["json"]["agent"] == {"id": "agent_01", "version": 12}
    assert call["json"]["initial_events"][0]["type"] == "message"
    assert call["json"]["schedule"]["timezone"] == "Asia/Shanghai"
    assert call["json"]["metadata"] == {"biz": "summary"}
    assert isinstance(deployment.agent, Agent)
    assert deployment.agent.version == 12
    assert isinstance(deployment.schedule, DeploymentSchedule)
    assert isinstance(deployment.resources[0], DeploymentResource)
    assert isinstance(deployment.paused_reason, DeploymentPausedReason)
    assert isinstance(deployment.paused_reason.error, DeploymentError)
    assert deployment.metadata == {"biz": "summary"}
    assert deployment.request_id == "req_01"


def test_update_distinguishes_omitted_and_explicit_null(client: Client):
    client.deployments.update("depl_01", name="renamed")
    omitted_body = client.transport.calls[-1]["json"]
    assert "environment_id" not in omitted_body
    assert "schedule" not in omitted_body

    client.deployments.update(
        "depl_01",
        environment_id=None,
        schedule=None,
        resources=[],
        metadata={},
    )
    clear_body = client.transport.calls[-1]["json"]
    assert clear_body["environment_id"] is None
    assert clear_body["schedule"] is None
    assert clear_body["resources"] == []
    assert clear_body["metadata"] == {}


def test_list_filters_and_cursor_pagination(client: Client):
    page = client.deployments.list(
        agent_id="agent_01",
        keyword="daily",
        status="active",
        include_archived=True,
        limit=10,
        created_at_gte="2026-07-01T00:00:00Z",
        created_at_lte="2026-07-31T23:59:59Z",
    )

    params = client.transport.calls[-1]["params"]
    assert params["agent_id"] == "agent_01"
    assert params["include_archived"] == "true"
    assert params["created_at[gte]"] == "2026-07-01T00:00:00Z"
    assert page.next_page == "next-depl"

    next_page = page.get_next()
    assert next_page is not None
    assert client.transport.calls[-1]["params"]["page"] == "next-depl"


def test_lifecycle_run_and_run_history_routes(client: Client):
    client.deployments.retrieve("depl_01")
    client.deployments.pause("depl_01")
    client.deployments.unpause("depl_01")
    client.deployments.archive("depl_01")
    run = client.deployments.run("depl_01")
    history = client.deployments.list_runs("depl_01", limit=20)
    all_runs = client.deployment_runs.list(limit=20)
    retrieved = client.deployment_runs.retrieve("drun_01")

    routes = [
        (call["method"], call["path"]) for call in client.transport.calls
    ]
    assert ("GET", "/deployments/depl_01") in routes
    assert ("POST", "/deployments/depl_01/pause") in routes
    assert ("POST", "/deployments/depl_01/unpause") in routes
    assert ("POST", "/deployments/depl_01/archive") in routes
    assert ("POST", "/deployments/depl_01/run") in routes
    assert ("GET", "/deployments/depl_01/runs") in routes
    assert ("GET", "/deployment_runs") in routes
    assert ("GET", "/deployment_runs/drun_01") in routes
    assert isinstance(run.agent, DeploymentAgentReference)
    assert isinstance(run.error, DeploymentError)
    assert run.request_id == "req_02"
    assert history.data[0].deployment_id == "depl_01"
    assert all_runs.data[0].id == "drun_01"
    assert retrieved.session_id == "session_01"


def test_async_resources_use_same_contract():
    async def exercise() -> None:
        client = AsyncClient(api_key="test-key", base_url="http://test")
        await client.transport.aclose()
        client.transport = _AsyncRecordingTransport()

        deployment = await client.deployments.create(
            name="daily-summary",
            agent={"id": "agent_01"},
            initial_events=[user_message("Summarize")],
            metadata={"biz": "summary"},
        )
        run = await client.deployments.run("depl_01")
        runs = await client.deployment_runs.list(limit=5)

        assert deployment.id == "depl_01"
        assert client.transport.calls[0]["json"]["metadata"] == {
            "biz": "summary",
        }
        assert run.id == "drun_01"
        assert runs.data[0].id == "drun_01"
        await client.aclose()

    asyncio.run(exercise())
