# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope agentstudio`` -- the managed-agent core loop.

Covers the one flow a shell script actually drives end to end: create an
agent, open a session on it, send a message, and follow the events back.
The rest of the AgentStudio surface (environments, files, deployments,
skills, vaults, memory stores, webhooks, security) is management-plane
CRUD that is far easier to read in Python than to spell in argv, so it
stays in the SDK.

Two things here differ from every other v2 command group, both because
AgentStudio ships its own sub-SDK rather than riding ``client.transport``:

* lists are **cursor**-paginated (``--limit`` plus an opaque ``--page``
  token), not page-number paginated;
* ``client.agentstudio`` is built lazily and resolves its own base URL
  from ``workspace`` + ``region``, so it needs a workspace to be useful.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import typer

from dashscope.cli.confirm import confirm_or_die
from dashscope.cli.errors import usage_error
from dashscope.cli.input import (
    parse_json_arg,
    parse_json_object_arg,
    read_text_arg,
)

_HELP = r"""Drive an AgentStudio managed agent.

The core loop, in order:

    dashscope agentstudio agents create --name demo --model qwen3.8-max
    dashscope agentstudio sessions create --agent "$AGENT_ID"
    dashscope agentstudio sessions events send "$SESSION_ID" --message "hi"
    dashscope agentstudio sessions events stream "$SESSION_ID"

AgentStudio is workspace-scoped: set `workspace` in the config file or
pass `--workspace` before the subcommand. Lists are cursor-paginated, so
page through them with the `next_page` value the previous page reported.
"""

_AGENTS_HELP = "Create and inspect agents."
_SESSIONS_HELP = "Open and inspect sessions on an agent."
_EVENTS_HELP = "Send messages to a session and read its events back."

_AGENT_COLUMNS = ("id", "name", "version", "created_at")
_AGENT_LIST_COLUMNS = ("id", "name", "version")
_AGENT_VERSION_COLUMNS = ("agent_id", "version", "created_at")
_SESSION_COLUMNS = ("id", "title", "status", "created_at")
_SESSION_LIST_COLUMNS = ("id", "title", "status")
_EVENT_COLUMNS = ("type", "role", "sequence_number", "created_at")


def _envelope(page: Any) -> Dict[str, Any]:
    """A machine-format envelope for one cursor page.

    ``CursorPage`` is not a response model, so the fields a caller needs
    to keep paging are assembled here rather than dumped.
    """
    payload: Dict[str, Any] = {
        "data": page.data,
        "next_page": page.next_page,
        "request_id": page.request_id,
    }
    payload.update(page.extra)
    return payload


def _render_page(renderer: Any, page: Any, columns: Any) -> None:
    renderer.print_rows(
        page.data,
        columns=columns,
        envelope=_envelope(page),
    )
    if renderer.table_mode and page.next_page:
        renderer.note(f"next page: {page.next_page}")


def _parse_array(raw: Optional[str], *, option_name: str) -> Optional[List]:
    """A JSON array option, with the shared ``@file`` / stdin forms."""
    value = parse_json_arg(raw, option_name=option_name)
    if value is None:
        return None
    if not isinstance(value, list):
        raise usage_error(
            f"{option_name}: expected a JSON array, got "
            f"{type(value).__name__}",
        )
    return value


def _render_event_stream(
    renderer: Any,
    stream: Any,
    *,
    deltas: bool,
) -> None:
    """One session event stream, in whichever format is active.

    ``text_deltas`` only yields anything when the stream was opened with
    ``event_deltas``; ``text_stream`` is the one a reconnect can rebuild,
    so it is the default. No ``except``: a mid-stream failure has to reach
    the error handler and exit non-zero.
    """
    try:
        if not renderer.table_mode:
            for event in stream:
                renderer.print_stream_chunk(event)
            return
        for text in stream.text_deltas if deltas else stream.text_stream:
            renderer.write(text)
        renderer.echo()
    finally:
        stream.close()


def build_app() -> typer.Typer:  # pylint: disable=too-many-statements
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    # ---- agents subgroup ---- #
    agents_app = typer.Typer(help=_AGENTS_HELP, no_args_is_help=True)

    @agents_app.command("create")
    def agents_create(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: typer.Context,
        name: str = typer.Option(..., "--name", help="Agent name."),
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Model id, e.g. qwen3.8-max.",
        ),
        description: Optional[str] = typer.Option(None, "--description"),
        system_prompt: Optional[str] = typer.Option(
            None,
            "--system-prompt",
            help="System prompt. Accepts @file and stdin (-).",
        ),
        tools: Optional[str] = typer.Option(
            None,
            "--tools",
            help="Tool definitions as a JSON array. Accepts @file.",
        ),
        mcp_servers: Optional[str] = typer.Option(
            None,
            "--mcp-servers",
            help="MCP server definitions as a JSON array. Accepts @file.",
        ),
        skills: Optional[str] = typer.Option(
            None,
            "--skills",
            help="Skill definitions as a JSON array. Accepts @file.",
        ),
        metadata: Optional[str] = typer.Option(
            None,
            "--metadata",
            help="Metadata as a JSON object. Accepts @file.",
        ),
    ) -> None:
        """Create an agent."""
        obj = ctx.obj
        agent = obj.client.agentstudio.agents.create(
            name=name,
            model=model,
            description=description,
            system_prompt=read_text_arg(
                system_prompt,
                option_name="--system-prompt",
            ),
            tools=_parse_array(tools, option_name="--tools"),
            mcp_servers=_parse_array(mcp_servers, option_name="--mcp-servers"),
            skills=_parse_array(skills, option_name="--skills"),
            metadata=parse_json_object_arg(
                metadata,
                option_name="--metadata",
            ),
        )
        obj.renderer.print_object(agent, columns=_AGENT_COLUMNS)

    @agents_app.command("get")
    def agents_get(
        ctx: typer.Context,
        agent_id: str = typer.Argument(..., help="Agent id."),
        version: Optional[int] = typer.Option(
            None,
            "--version",
            help="Read one specific version instead of the latest.",
        ),
    ) -> None:
        """Show one agent."""
        obj = ctx.obj
        agent = obj.client.agentstudio.agents.retrieve(
            agent_id,
            version=version,
        )
        obj.renderer.print_object(agent, columns=_AGENT_COLUMNS)

    @agents_app.command("list")
    def agents_list(
        ctx: typer.Context,
        limit: Optional[int] = typer.Option(
            None,
            "--limit",
            help="Maximum items on this page.",
        ),
        page: Optional[str] = typer.Option(
            None,
            "--page",
            help="Opaque cursor from a previous page's next_page.",
        ),
        include_archived: Optional[bool] = typer.Option(
            None,
            "--include-archived/--exclude-archived",
        ),
    ) -> None:
        """List agents."""
        obj = ctx.obj
        result = obj.client.agentstudio.agents.list(
            limit=limit,
            page=page,
            include_archived=include_archived,
        )
        _render_page(obj.renderer, result, _AGENT_LIST_COLUMNS)

    @agents_app.command("versions")
    def agents_versions(
        ctx: typer.Context,
        agent_id: str = typer.Argument(..., help="Agent id."),
        limit: Optional[int] = typer.Option(
            None,
            "--limit",
            help="Maximum items on this page.",
        ),
        page: Optional[str] = typer.Option(
            None,
            "--page",
            help="Opaque cursor from a previous page's next_page.",
        ),
    ) -> None:
        """List an agent's versions.

        `agents get --version N` reads one of these back; this is how you
        find out which N to ask for.
        """
        obj = ctx.obj
        result = obj.client.agentstudio.agents.list_versions(
            agent_id,
            limit=limit,
            page=page,
        )
        _render_page(obj.renderer, result, _AGENT_VERSION_COLUMNS)

    @agents_app.command("update")
    def agents_update(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: typer.Context,
        agent_id: str = typer.Argument(..., help="Agent id."),
        version: int = typer.Option(
            ...,
            "--version",
            help="The agent's current version; a mismatch is a 409.",
        ),
        name: Optional[str] = typer.Option(
            None,
            "--name",
            help="New name. The server requires one on every update.",
        ),
        model: Optional[str] = typer.Option(None, "-m", "--model"),
        description: Optional[str] = typer.Option(None, "--description"),
        system_prompt: Optional[str] = typer.Option(
            None,
            "--system-prompt",
            help="System prompt. Accepts @file and stdin (-).",
        ),
        tools: Optional[str] = typer.Option(
            None,
            "--tools",
            help="Tool definitions as a JSON array. Accepts @file.",
        ),
        mcp_servers: Optional[str] = typer.Option(
            None,
            "--mcp-servers",
            help="MCP server definitions as a JSON array. Accepts @file.",
        ),
        skills: Optional[str] = typer.Option(
            None,
            "--skills",
            help="Skill definitions as a JSON array. Accepts @file.",
        ),
        metadata: Optional[str] = typer.Option(
            None,
            "--metadata",
            help="Metadata as a JSON object. Accepts @file.",
        ),
    ) -> None:
        """Update the latest version of an agent."""
        obj = ctx.obj
        agent = obj.client.agentstudio.agents.update(
            agent_id,
            version=version,
            name=name,
            model=model,
            description=description,
            system_prompt=read_text_arg(
                system_prompt,
                option_name="--system-prompt",
            ),
            tools=_parse_array(tools, option_name="--tools"),
            mcp_servers=_parse_array(mcp_servers, option_name="--mcp-servers"),
            skills=_parse_array(skills, option_name="--skills"),
            metadata=parse_json_object_arg(
                metadata,
                option_name="--metadata",
            ),
        )
        obj.renderer.print_object(agent, columns=_AGENT_COLUMNS)

    @agents_app.command("archive")
    def agents_archive(
        ctx: typer.Context,
        agent_id: str = typer.Argument(..., help="Agent id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Archive an agent."""
        confirm_or_die(f"archive agent {agent_id}?", yes=yes)
        obj = ctx.obj
        agent = obj.client.agentstudio.agents.archive(agent_id)
        obj.renderer.print_object(agent, columns=_AGENT_COLUMNS)

    # ---- sessions subgroup ---- #
    sessions_app = typer.Typer(help=_SESSIONS_HELP, no_args_is_help=True)

    @sessions_app.command("create")
    def sessions_create(  # pylint: disable=too-many-arguments
        ctx: typer.Context,
        agent: str = typer.Option(
            ...,
            "--agent",
            help="Agent id to run this session on.",
        ),
        environment_id: Optional[str] = typer.Option(
            None,
            "--environment-id",
        ),
        title: Optional[str] = typer.Option(None, "--title"),
        vault_ids: Optional[str] = typer.Option(
            None,
            "--vault-ids",
            help="Vault ids as a JSON array. Accepts @file.",
        ),
        environment_variables: Optional[str] = typer.Option(
            None,
            "--env",
            help="Environment variables as a JSON object. Accepts @file.",
        ),
        mcp_configs: Optional[str] = typer.Option(
            None,
            "--mcp-configs",
            help="MCP configs as a JSON array. Accepts @file.",
        ),
        metadata: Optional[str] = typer.Option(
            None,
            "--metadata",
            help="Metadata as a JSON object. Accepts @file.",
        ),
    ) -> None:
        """Open a session on an agent."""
        obj = ctx.obj
        session = obj.client.agentstudio.sessions.create(
            agent=agent,
            environment_id=environment_id,
            title=title,
            vault_ids=_parse_array(vault_ids, option_name="--vault-ids"),
            environment_variables=parse_json_object_arg(
                environment_variables,
                option_name="--env",
            ),
            mcp_configs=_parse_array(mcp_configs, option_name="--mcp-configs"),
            metadata=parse_json_object_arg(
                metadata,
                option_name="--metadata",
            ),
        )
        obj.renderer.print_object(session, columns=_SESSION_COLUMNS)

    @sessions_app.command("get")
    def sessions_get(
        ctx: typer.Context,
        session_id: str = typer.Argument(..., help="Session id."),
    ) -> None:
        """Show one session."""
        obj = ctx.obj
        session = obj.client.agentstudio.sessions.retrieve(session_id)
        obj.renderer.print_object(session, columns=_SESSION_COLUMNS)

    @sessions_app.command("list")
    def sessions_list(
        ctx: typer.Context,
        limit: Optional[int] = typer.Option(None, "--limit"),
        page: Optional[str] = typer.Option(
            None,
            "--page",
            help="Opaque cursor from a previous page's next_page.",
        ),
        agent_id: Optional[str] = typer.Option(
            None,
            "--agent-id",
            help="Only sessions of this agent.",
        ),
        status: Optional[List[str]] = typer.Option(
            None,
            "--status",
            help="Only sessions in this status. Repeatable.",
        ),
    ) -> None:
        """List sessions."""
        obj = ctx.obj
        result = obj.client.agentstudio.sessions.list(
            limit=limit,
            page=page,
            agent_id=agent_id,
            statuses=list(status) if status else None,
        )
        _render_page(obj.renderer, result, _SESSION_LIST_COLUMNS)

    @sessions_app.command("update")
    def sessions_update(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: typer.Context,
        session_id: str = typer.Argument(..., help="Session id."),
        title: Optional[str] = typer.Option(None, "--title"),
        vault_ids: Optional[str] = typer.Option(
            None,
            "--vault-ids",
            help="Vault ids as a JSON array. Accepts @file. An empty "
            "array clears them.",
        ),
        environment_variables: Optional[str] = typer.Option(
            None,
            "--env",
            help="Environment variables as a JSON object. Accepts @file. "
            "An empty object clears them.",
        ),
        mcp_configs: Optional[str] = typer.Option(
            None,
            "--mcp-configs",
            help="MCP configs as a JSON array. Accepts @file.",
        ),
        metadata: Optional[str] = typer.Option(
            None,
            "--metadata",
            help="Metadata as a JSON object. Accepts @file.",
        ),
        agent_config: Optional[str] = typer.Option(
            None,
            "--agent-config",
            help="Session-level agent overrides as a JSON object carrying "
            "system / tools / mcp_servers / skills. Accepts @file. Not the "
            "agent id -- that is set at create time.",
        ),
    ) -> None:
        """Patch a session. Omitted fields keep their current value."""
        obj = ctx.obj
        parsed = {
            "title": title,
            "vault_ids": _parse_array(vault_ids, option_name="--vault-ids"),
            "environment_variables": parse_json_object_arg(
                environment_variables,
                option_name="--env",
            ),
            "mcp_configs": _parse_array(
                mcp_configs,
                option_name="--mcp-configs",
            ),
            "metadata": parse_json_object_arg(
                metadata,
                option_name="--metadata",
            ),
            "agent": parse_json_object_arg(
                agent_config,
                option_name="--agent-config",
            ),
        }
        if all(value is None for value in parsed.values()):
            raise usage_error(
                "nothing to update: pass at least one of --title, "
                "--vault-ids, --env, --mcp-configs, --metadata, "
                "--agent-config",
            )
        session = obj.client.agentstudio.sessions.update(
            session_id,
            **{k: v for k, v in parsed.items() if v is not None},
        )
        obj.renderer.print_object(session, columns=_SESSION_COLUMNS)

    @sessions_app.command("archive")
    def sessions_archive(
        ctx: typer.Context,
        session_id: str = typer.Argument(..., help="Session id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Archive a session."""
        confirm_or_die(f"archive session {session_id}?", yes=yes)
        obj = ctx.obj
        session = obj.client.agentstudio.sessions.archive(session_id)
        obj.renderer.print_object(session, columns=_SESSION_COLUMNS)

    @sessions_app.command("delete")
    def sessions_delete(
        ctx: typer.Context,
        session_id: str = typer.Argument(..., help="Session id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Delete a session."""
        confirm_or_die(f"delete session {session_id}?", yes=yes)
        obj = ctx.obj
        result = obj.client.agentstudio.sessions.delete(session_id)
        obj.renderer.print_object(result)

    # ---- session events subgroup ---- #
    events_app = typer.Typer(help=_EVENTS_HELP, no_args_is_help=True)

    @events_app.command("send")
    def events_send(
        ctx: typer.Context,
        session_id: str = typer.Argument(..., help="Session id."),
        message: Optional[str] = typer.Option(
            None,
            "--message",
            help="Text to send as one user message. Accepts @file and "
            "stdin (-).",
        ),
        events: Optional[str] = typer.Option(
            None,
            "--events",
            help="Client events as a raw JSON array, sent as-is. Accepts "
            "@file and stdin (-). Overrides --message.",
        ),
        thread_id: Optional[str] = typer.Option(
            None,
            "--thread-id",
            help="Session thread to post the --message into.",
        ),
    ) -> None:
        """Send events to a session."""
        obj = ctx.obj
        payload = _parse_array(events, option_name="--events")
        if payload is None:
            text = read_text_arg(message, option_name="--message")
            if not text:
                raise usage_error(
                    "no input given: pass --message or --events",
                )
            # Imported here, not at module scope: a command module must
            # not pay for the SDK when the app tree is being built.
            # pylint: disable-next=import-outside-toplevel
            from dashscope.agentstudio import user_message

            payload = [user_message(text, session_thread_id=thread_id)]
        result = obj.client.agentstudio.sessions.events.send(
            session_id,
            payload,
        )
        obj.renderer.print_object(result)

    @events_app.command("list")
    def events_list(
        ctx: typer.Context,
        session_id: str = typer.Argument(..., help="Session id."),
        limit: Optional[int] = typer.Option(None, "--limit"),
        page: Optional[str] = typer.Option(
            None,
            "--page",
            help="Opaque cursor from a previous page's next_page.",
        ),
        event_type: Optional[List[str]] = typer.Option(
            None,
            "--type",
            help="Only events of this type. Repeatable.",
        ),
        order: Optional[str] = typer.Option(
            None,
            "--order",
            help="Sort order, e.g. asc or desc.",
        ),
    ) -> None:
        """Read a session's persisted events."""
        obj = ctx.obj
        result = obj.client.agentstudio.sessions.events.list(
            session_id,
            types=list(event_type) if event_type else None,
            limit=limit,
            order=order,
            page=page,
        )
        _render_page(obj.renderer, result, _EVENT_COLUMNS)

    @events_app.command("stream")
    def events_stream(
        ctx: typer.Context,
        session_id: str = typer.Argument(..., help="Session id."),
        deltas: Optional[List[str]] = typer.Option(
            None,
            "--deltas",
            help="Opt into incremental frames for these event types "
            "(message, reasoning). Repeatable.",
        ),
        timeout: Optional[float] = typer.Option(
            None,
            "--timeout",
            help="Stream read timeout in seconds.",
        ),
    ) -> None:
        """Follow a session's event stream until the run settles."""
        obj = ctx.obj
        stream = obj.client.agentstudio.sessions.events.stream(
            session_id,
            event_deltas=list(deltas) if deltas else None,
            timeout=timeout,
        )
        _render_event_stream(obj.renderer, stream, deltas=bool(deltas))

    sessions_app.add_typer(events_app, name="events")
    app.add_typer(agents_app, name="agents")
    app.add_typer(sessions_app, name="sessions")
    return app
