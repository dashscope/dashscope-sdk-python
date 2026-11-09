# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope agentstudio`` -- agents, sessions and session events."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from dashscope.applications.agentstudio.pagination import CursorPage
from dashscope.applications.agentstudio.resources._helpers import _coerce_event
from dashscope.applications.agentstudio.types import (
    Agent,
    AgentVersion,
    DeleteResponse,
    Session,
)
from dashscope.cli.app import build_app


def _agent(**overrides: Any) -> Agent:
    data: Dict[str, Any] = {
        "id": "agent-1",
        "type": "agent",
        "version": 3,
        "name": "helper",
        "model": {"id": "qwen3.8-max"},
        "created_at": "2026-11-01T00:00:00Z",
        "request_id": "req-1",
    }
    data.update(overrides)
    return Agent(**data)


def _version(**overrides: Any) -> AgentVersion:
    data: Dict[str, Any] = {
        "agent_id": "agent-1",
        "version": 3,
        "created_at": "2026-11-01T00:00:00Z",
    }
    data.update(overrides)
    return AgentVersion(**data)


def _session(**overrides: Any) -> Session:
    data: Dict[str, Any] = {
        "id": "sess-1",
        "type": "session",
        "title": "a run",
        "status": "idle",
        "agent": {"id": "agent-1", "version": 3},
        "created_at": "2026-11-01T00:00:00Z",
        "request_id": "req-1",
    }
    data.update(overrides)
    return Session(**data)


def _event_payload(**overrides: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "type": "message",
        "role": "assistant",
        "sequence_number": 2,
        "content": [{"type": "text", "text": "hello"}],
        "created_at": "2026-11-01T00:00:01Z",
    }
    data.update(overrides)
    return data


def _page(
    items: List[Any],
    *,
    next_page: Optional[str] = None,
    **extra: Any,
) -> CursorPage:
    return CursorPage(
        data=items,
        next_page=next_page,
        request_id="req-1",
        extra=extra,
    )


class _Agents:
    """Stand-in for ``client.agentstudio.agents``."""

    def __init__(self, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results

    def _go(self, method: str, args: tuple = (), **kwargs: Any) -> Any:
        self.calls.append(
            {"method": method, "args": args, "kwargs": kwargs},
        )
        if method in self._results:
            return self._results[method]
        if method == "list":
            return _page([_agent()])
        if method == "list_versions":
            return _page([_version()])
        return _agent()

    def create(self, **kwargs):
        return self._go("create", **kwargs)

    def retrieve(self, agent_id, **kwargs):
        return self._go("retrieve", (agent_id,), **kwargs)

    def list(self, **kwargs):
        return self._go("list", **kwargs)

    def list_versions(self, agent_id, **kwargs):
        return self._go("list_versions", (agent_id,), **kwargs)

    def update(self, agent_id, **kwargs):
        return self._go("update", (agent_id,), **kwargs)

    def archive(self, agent_id, **kwargs):
        return self._go("archive", (agent_id,), **kwargs)


class _SessionEvents:
    """Stand-in for ``client.agentstudio.sessions.events``."""

    def __init__(self, stream: Any = None, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results
        self._stream = stream

    def send(self, session_id, events, **kwargs):
        self.calls.append(
            {"method": "send", "args": (session_id, events), "kwargs": kwargs},
        )
        return self._results.get(
            "send",
            {"accepted": 1, "request_id": "req-1"},
        )

    def list(self, session_id, **kwargs):
        self.calls.append(
            {"method": "list", "args": (session_id,), "kwargs": kwargs},
        )
        return self._results.get(
            "list",
            _page([_coerce_event(_event_payload())]),
        )

    def stream(self, session_id, **kwargs):
        self.calls.append(
            {"method": "stream", "args": (session_id,), "kwargs": kwargs},
        )
        return self._stream if self._stream is not None else _Stream()


class _Sessions:
    """Stand-in for ``client.agentstudio.sessions``."""

    def __init__(self, events: Any = None, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results
        self.events = events if events is not None else _SessionEvents()

    def _go(self, method: str, args: tuple = (), **kwargs: Any) -> Any:
        self.calls.append(
            {"method": method, "args": args, "kwargs": kwargs},
        )
        if method in self._results:
            return self._results[method]
        if method == "list":
            return _page([_session()])
        if method == "delete":
            return DeleteResponse(
                id="sess-1",
                type="session",
                request_id="req-1",
            )
        return _session()

    def create(self, **kwargs):
        return self._go("create", **kwargs)

    def retrieve(self, session_id, **kwargs):
        return self._go("retrieve", (session_id,), **kwargs)

    def list(self, **kwargs):
        return self._go("list", **kwargs)

    def update(self, session_id, **kwargs):
        return self._go("update", (session_id,), **kwargs)

    def archive(self, session_id, **kwargs):
        return self._go("archive", (session_id,), **kwargs)

    def delete(self, session_id, **kwargs):
        return self._go("delete", (session_id,), **kwargs)


class _Stream:
    """Stand-in for the SDK's single-pass typed event stream."""

    def __init__(
        self,
        events: Optional[List[Dict[str, Any]]] = None,
        *,
        texts: Optional[List[str]] = None,
        deltas: Optional[List[str]] = None,
    ) -> None:
        self._events = events if events is not None else [_event_payload()]
        self._texts = texts if texts is not None else ["hello ", "world"]
        self._deltas = deltas if deltas is not None else []
        self.closed = False

    def __iter__(self):
        return iter([_coerce_event(item) for item in self._events])

    @property
    def text_stream(self):
        return iter(self._texts)

    @property
    def text_deltas(self):
        return iter(self._deltas)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def studio(make_client):
    """Install the agents/sessions pair and hand both back."""

    def install(
        *,
        agents: Optional[Dict[str, Any]] = None,
        sessions: Optional[Dict[str, Any]] = None,
        events: Optional[Dict[str, Any]] = None,
        stream: Any = None,
    ) -> SimpleNamespace:
        agent_resource = _Agents(**(agents or {}))
        event_resource = _SessionEvents(stream, **(events or {}))
        session_resource = _Sessions(event_resource, **(sessions or {}))
        make_client(
            agentstudio=SimpleNamespace(
                agents=agent_resource,
                sessions=session_resource,
            ),
        )
        return SimpleNamespace(
            agents=agent_resource,
            sessions=session_resource,
            events=event_resource,
        )

    return install


class TestAgents:
    def test_create(self, invoke, studio):
        resource = studio()
        result = invoke(
            [
                "agentstudio",
                "agents",
                "create",
                "--name",
                "helper",
                "-m",
                "qwen3.8-max",
            ],
        )
        assert result.exit_code == 0
        assert "agent-1" in result.stdout
        kwargs = resource.agents.calls[0]["kwargs"]
        assert kwargs["name"] == "helper"
        assert kwargs["model"] == "qwen3.8-max"
        assert kwargs["tools"] is None

    def test_create_takes_the_json_definitions(self, invoke, studio):
        resource = studio()
        invoke(
            [
                "agentstudio",
                "agents",
                "create",
                "--name",
                "helper",
                "-m",
                "qwen3.8-max",
                "--system-prompt",
                "be terse",
                "--tools",
                '[{"type": "function", "name": "lookup"}]',
                "--mcp-servers",
                '[{"url": "https://example.com/mcp"}]',
                "--skills",
                '[{"id": "skill-1"}]',
                "--metadata",
                '{"team": "sdk"}',
            ],
        )
        kwargs = resource.agents.calls[0]["kwargs"]
        assert kwargs["system_prompt"] == "be terse"
        assert kwargs["tools"] == [{"type": "function", "name": "lookup"}]
        assert kwargs["mcp_servers"] == [
            {"url": "https://example.com/mcp"},
        ]
        assert kwargs["skills"] == [{"id": "skill-1"}]
        assert kwargs["metadata"] == {"team": "sdk"}

    def test_a_definition_must_be_an_array(self, invoke, studio):
        resource = studio()
        result = invoke(
            [
                "agentstudio",
                "agents",
                "create",
                "--name",
                "helper",
                "-m",
                "qwen3.8-max",
                "--tools",
                '{"type": "function"}',
            ],
        )
        assert result.exit_code == 2
        assert "--tools: expected a JSON array" in result.stderr
        assert resource.agents.calls == []

    def test_get_reads_the_latest_version(self, invoke, studio):
        resource = studio()
        invoke(["agentstudio", "agents", "get", "agent-1"])
        call = resource.agents.calls[0]
        assert call["method"] == "retrieve"
        assert call["args"] == ("agent-1",)
        assert call["kwargs"] == {"version": None}

    def test_get_can_pin_one_version(self, invoke, studio):
        resource = studio()
        invoke(["agentstudio", "agents", "get", "agent-1", "--version", "2"])
        assert resource.agents.calls[0]["kwargs"] == {"version": 2}

    def test_list_prints_this_page_only(self, invoke, studio):
        """``CursorPage`` auto-paginates when iterated; the CLI must not."""
        resource = studio(
            agents={
                "list": _page(
                    [_agent(), _agent(id="agent-2", name="second")],
                    next_page="cursor-2",
                ),
            },
        )
        result = invoke(["agentstudio", "agents", "list"])
        assert result.exit_code == 0
        assert "agent-1" in result.stdout
        assert "agent-2" in result.stdout
        assert "next page: cursor-2" in result.stderr
        assert resource.agents.calls[0]["kwargs"] == {
            "limit": None,
            "page": None,
            "include_archived": None,
        }

    def test_list_envelope_carries_the_cursor(self, invoke, studio):
        studio(
            agents={"list": _page([_agent()], next_page="cursor-2")},
        )
        result = invoke(
            ["agentstudio", "agents", "list"],
            global_args=["-o", "json"],
        )
        doc = json.loads(result.stdout)
        assert doc["next_page"] == "cursor-2"
        assert doc["request_id"] == "req-1"
        assert doc["data"][0]["id"] == "agent-1"

    def test_versions_lists_the_page(self, invoke, studio):
        """`agents get --version N` needs a way to discover N."""
        resource = studio(
            agents={"list_versions": _page([_version(version=3)])},
        )
        result = invoke(["agentstudio", "agents", "versions", "agent-1"])
        assert result.exit_code == 0
        assert resource.agents.calls[0] == {
            "method": "list_versions",
            "args": ("agent-1",),
            "kwargs": {"limit": None, "page": None},
        }
        assert "agent-1" in result.stdout

    def test_versions_forwards_the_cursor(self, invoke, studio):
        resource = studio(agents={"list_versions": _page([_version()])})
        invoke(
            [
                "agentstudio",
                "agents",
                "versions",
                "agent-1",
                "--limit",
                "5",
                "--page",
                "cursor-2",
            ],
        )
        assert resource.agents.calls[0]["kwargs"] == {
            "limit": 5,
            "page": "cursor-2",
        }

    def test_versions_reports_the_next_cursor(self, invoke, studio):
        studio(
            agents={
                "list_versions": _page([_version()], next_page="cursor-9"),
            },
        )
        result = invoke(["agentstudio", "agents", "versions", "agent-1"])
        assert "next page: cursor-9" in result.stderr

    def test_versions_json_envelope_carries_the_cursor(self, invoke, studio):
        studio(
            agents={
                "list_versions": _page(
                    [_version(version=2)],
                    next_page="cursor-9",
                ),
            },
        )
        result = invoke(
            ["agentstudio", "agents", "versions", "agent-1"],
            global_args=["-o", "json"],
        )
        doc = json.loads(result.stdout)
        assert doc["next_page"] == "cursor-9"
        assert doc["data"][0]["version"] == 2

    def test_update_needs_the_version_it_expects(self, invoke, studio):
        resource = studio()
        result = invoke(["agentstudio", "agents", "update", "agent-1"])
        assert result.exit_code == 2
        assert resource.agents.calls == []

    def test_update_sends_the_new_fields(self, invoke, studio):
        resource = studio(agents={"update": _agent(version=4, name="renamed")})
        result = invoke(
            [
                "agentstudio",
                "agents",
                "update",
                "agent-1",
                "--version",
                "3",
                "--name",
                "renamed",
            ],
        )
        assert result.exit_code == 0
        assert "renamed" in result.stdout
        call = resource.agents.calls[0]
        assert call["args"] == ("agent-1",)
        assert call["kwargs"]["version"] == 3
        assert call["kwargs"]["name"] == "renamed"

    def test_archive_needs_a_confirmation(self, invoke, studio):
        resource = studio()
        result = invoke(["agentstudio", "agents", "archive", "agent-1"])
        assert result.exit_code != 0
        assert resource.agents.calls == []

    def test_archive_with_yes_goes_through(self, invoke, studio):
        resource = studio()
        result = invoke(
            ["agentstudio", "agents", "archive", "agent-1", "--yes"],
        )
        assert result.exit_code == 0
        assert resource.agents.calls[0]["method"] == "archive"


class TestSessions:
    def test_create(self, invoke, studio):
        resource = studio()
        result = invoke(
            ["agentstudio", "sessions", "create", "--agent", "agent-1"],
        )
        assert result.exit_code == 0
        assert "sess-1" in result.stdout
        kwargs = resource.sessions.calls[0]["kwargs"]
        assert kwargs["agent"] == "agent-1"
        assert kwargs["vault_ids"] is None

    def test_create_takes_the_environment_options(self, invoke, studio):
        resource = studio()
        invoke(
            [
                "agentstudio",
                "sessions",
                "create",
                "--agent",
                "agent-1",
                "--environment-id",
                "env-1",
                "--title",
                "a run",
                "--vault-ids",
                '["vault-1"]',
                "--env",
                '{"TOKEN": "t"}',
                "--mcp-configs",
                '[{"url": "https://example.com/mcp"}]',
                "--metadata",
                '{"team": "sdk"}',
            ],
        )
        kwargs = resource.sessions.calls[0]["kwargs"]
        assert kwargs["environment_id"] == "env-1"
        assert kwargs["vault_ids"] == ["vault-1"]
        assert kwargs["environment_variables"] == {"TOKEN": "t"}
        assert kwargs["mcp_configs"] == [
            {"url": "https://example.com/mcp"},
        ]
        assert kwargs["metadata"] == {"team": "sdk"}

    def test_the_agent_is_required(self, invoke, studio):
        studio()
        assert invoke(["agentstudio", "sessions", "create"]).exit_code == 2

    def test_get(self, invoke, studio):
        resource = studio()
        invoke(["agentstudio", "sessions", "get", "sess-1"])
        assert resource.sessions.calls[0]["method"] == "retrieve"
        assert resource.sessions.calls[0]["args"] == ("sess-1",)

    def test_list_filters_by_agent_and_status(self, invoke, studio):
        resource = studio()
        result = invoke(
            [
                "agentstudio",
                "sessions",
                "list",
                "--agent-id",
                "agent-1",
                "--status",
                "idle",
                "--status",
                "running",
                "--limit",
                "5",
            ],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        assert resource.sessions.calls[0]["kwargs"] == {
            "limit": 5,
            "page": None,
            "agent_id": "agent-1",
            "statuses": ["idle", "running"],
        }
        assert json.loads(result.stdout)["data"][0]["status"] == "idle"

    def test_update_sends_only_the_fields_it_was_given(self, invoke, studio):
        """An omitted field must stay absent, not arrive as None."""
        resource = studio(sessions={"update": _session(title="renamed")})
        result = invoke(
            ["agentstudio", "sessions", "update", "sess-1", "--title", "new"],
        )
        assert result.exit_code == 0
        assert resource.sessions.calls[0] == {
            "method": "update",
            "args": ("sess-1",),
            "kwargs": {"title": "new"},
        }

    def test_update_with_nothing_to_change_is_refused(self, invoke, studio):
        """A no-op POST that prints the session back looks like success."""
        resource = studio()
        result = invoke(["agentstudio", "sessions", "update", "sess-1"])
        assert result.exit_code == 2
        assert "nothing to update" in result.stderr
        assert resource.sessions.calls == []

    def test_update_maps_the_json_options(self, invoke, studio):
        resource = studio(sessions={"update": _session()})
        invoke(
            [
                "agentstudio",
                "sessions",
                "update",
                "sess-1",
                "--vault-ids",
                '["vault-1"]',
                "--env",
                '{"TOKEN": "t"}',
                "--mcp-configs",
                '[{"name": "m"}]',
                "--metadata",
                '{"team": "infra"}',
                "--agent-config",
                '{"system": "be brief"}',
            ],
        )
        assert resource.sessions.calls[0]["kwargs"] == {
            "vault_ids": ["vault-1"],
            "environment_variables": {"TOKEN": "t"},
            "mcp_configs": [{"name": "m"}],
            "metadata": {"team": "infra"},
            "agent": {"system": "be brief"},
        }

    def test_update_can_clear_a_list_with_an_empty_array(self, invoke, studio):
        """``[]`` means remove them all; omitting the flag means keep them."""
        resource = studio(sessions={"update": _session()})
        invoke(
            [
                "agentstudio",
                "sessions",
                "update",
                "sess-1",
                "--vault-ids",
                "[]",
            ],
        )
        assert resource.sessions.calls[0]["kwargs"] == {"vault_ids": []}

    def test_agent_config_must_be_an_object(self, invoke, studio):
        """It is a config patch, not the agent id ``create --agent`` takes."""
        resource = studio()
        result = invoke(
            [
                "agentstudio",
                "sessions",
                "update",
                "sess-1",
                "--agent-config",
                "agent-1",
            ],
        )
        assert result.exit_code == 2
        assert "--agent-config" in result.stderr
        assert resource.sessions.calls == []

    def test_update_renders_the_patched_session(self, invoke, studio):
        studio(sessions={"update": _session(title="renamed")})
        result = invoke(
            ["agentstudio", "sessions", "update", "sess-1", "--title", "x"],
        )
        assert "renamed" in result.stdout

    def test_archive_needs_a_confirmation(self, invoke, studio):
        resource = studio()
        result = invoke(["agentstudio", "sessions", "archive", "sess-1"])
        assert result.exit_code != 0
        assert resource.sessions.calls == []

    def test_archive_with_yes_goes_through(self, invoke, studio):
        resource = studio()
        result = invoke(
            ["agentstudio", "sessions", "archive", "sess-1", "-y"],
        )
        assert result.exit_code == 0
        assert resource.sessions.calls[0]["method"] == "archive"

    def test_delete_with_yes_goes_through(self, invoke, studio):
        resource = studio()
        result = invoke(
            ["agentstudio", "sessions", "delete", "sess-1", "-y"],
        )
        assert result.exit_code == 0
        assert "sess-1" in result.stdout
        assert resource.sessions.calls[0]["method"] == "delete"


class TestSessionEvents:
    def test_send_wraps_the_message(self, invoke, studio):
        resource = studio()
        result = invoke(
            [
                "agentstudio",
                "sessions",
                "events",
                "send",
                "sess-1",
                "--message",
                "hi",
            ],
        )
        assert result.exit_code == 0
        assert "accepted" in result.stdout
        call = resource.events.calls[0]
        assert call["args"][0] == "sess-1"
        sent = call["args"][1]
        assert len(sent) == 1
        assert sent[0]["type"] == "message"
        assert sent[0]["role"] == "user"
        assert sent[0]["content"][0]["text"] == "hi"

    def test_send_can_post_into_one_thread(self, invoke, studio):
        resource = studio()
        invoke(
            [
                "agentstudio",
                "sessions",
                "events",
                "send",
                "sess-1",
                "--message",
                "hi",
                "--thread-id",
                "thread-1",
            ],
        )
        sent = resource.events.calls[0]["args"][1]
        assert sent[0]["session_thread_id"] == "thread-1"

    def test_send_takes_raw_events(self, invoke, studio):
        """``--events`` is the escape hatch for frames the CLI has no flag."""
        resource = studio()
        invoke(
            [
                "agentstudio",
                "sessions",
                "events",
                "send",
                "sess-1",
                "--events",
                '[{"type": "message", "role": "user", "content": []}]',
            ],
        )
        assert resource.events.calls[0]["args"][1] == [
            {"type": "message", "role": "user", "content": []},
        ]

    def test_send_needs_one_of_the_two_inputs(self, invoke, studio):
        resource = studio()
        result = invoke(
            ["agentstudio", "sessions", "events", "send", "sess-1"],
        )
        assert result.exit_code == 2
        assert "no input given" in result.stderr
        assert resource.events.calls == []

    def test_list_reads_the_persisted_events(self, invoke, studio):
        resource = studio()
        result = invoke(
            ["agentstudio", "sessions", "events", "list", "sess-1"],
        )
        assert result.exit_code == 0
        assert "message" in result.stdout
        assert resource.events.calls[0]["kwargs"] == {
            "types": None,
            "limit": None,
            "order": None,
            "page": None,
        }

    def test_list_filters_by_type(self, invoke, studio):
        resource = studio()
        invoke(
            [
                "agentstudio",
                "sessions",
                "events",
                "list",
                "sess-1",
                "--type",
                "message",
                "--type",
                "error",
                "--order",
                "desc",
            ],
        )
        kwargs = resource.events.calls[0]["kwargs"]
        assert kwargs["types"] == ["message", "error"]
        assert kwargs["order"] == "desc"

    def test_stream_prints_the_answer_text(self, invoke, studio):
        stream = _Stream(texts=["hello ", "world"])
        resource = studio(stream=stream)
        result = invoke(
            ["agentstudio", "sessions", "events", "stream", "sess-1"],
        )
        assert result.exit_code == 0
        assert result.stdout == "hello world\n"
        assert resource.events.calls[0]["kwargs"] == {
            "event_deltas": None,
            "timeout": None,
        }
        assert stream.closed is True

    def test_stream_opting_into_deltas_reads_the_deltas(self, invoke, studio):
        """``text_deltas`` only yields when the stream asked for frames."""
        stream = _Stream(texts=["ignored"], deltas=["a", "b"])
        resource = studio(stream=stream)
        result = invoke(
            [
                "agentstudio",
                "sessions",
                "events",
                "stream",
                "sess-1",
                "--deltas",
                "message",
            ],
        )
        assert result.stdout == "ab\n"
        assert resource.events.calls[0]["kwargs"]["event_deltas"] == [
            "message",
        ]

    def test_stream_json_is_one_event_per_line(self, invoke, studio):
        stream = _Stream(
            [
                _event_payload(),
                _event_payload(type="error", sequence_number=3),
            ],
        )
        studio(stream=stream)
        result = invoke(
            ["agentstudio", "sessions", "events", "stream", "sess-1"],
            global_args=["-o", "json"],
        )
        lines = [json.loads(line) for line in result.stdout.splitlines()]
        assert [line["type"] for line in lines] == ["message", "error"]
        assert stream.closed is True

    def test_stream_passes_the_timeout(self, invoke, studio):
        resource = studio(stream=_Stream([]))
        invoke(
            [
                "agentstudio",
                "sessions",
                "events",
                "stream",
                "sess-1",
                "--timeout",
                "30",
            ],
        )
        assert resource.events.calls[0]["kwargs"]["timeout"] == 30.0


class TestGroupShape:
    def test_events_is_nested_under_sessions(self, app):
        import typer

        studio = typer.main.get_command(app).commands["agentstudio"]
        assert set(studio.commands) == {"agents", "sessions"}
        assert set(studio.commands["agents"].commands) == {
            "create",
            "get",
            "list",
            "versions",
            "update",
            "archive",
        }
        sessions = studio.commands["sessions"]
        assert set(sessions.commands) - {"events"} == {
            "create",
            "get",
            "list",
            "update",
            "archive",
            "delete",
        }
        assert "events" in sessions.commands
        assert set(sessions.commands["events"].commands) == {
            "send",
            "list",
            "stream",
        }

    def test_the_other_eleven_resources_stay_in_the_sdk(self, app):
        """The core loop only: environments, files, vaults and the rest."""
        import typer

        studio = typer.main.get_command(app).commands["agentstudio"]
        assert "environments" not in studio.commands
        assert "vaults" not in studio.commands

    def test_bare_group_shows_help(self, app, runner):
        result = runner.invoke(app, ["agentstudio"])
        assert "Usage" in result.stdout
