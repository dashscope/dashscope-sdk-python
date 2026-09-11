# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unified type definitions for the AgentStudio SDK.

This module merges all types from the ``types/`` sub-package into a single
file organized into six sections:

    1. Common types         (EnvironmentConfig, Usage, Metadata, …)
    2. Content blocks       (TextBlock, ImageBlock, … ContentBlock)
    3. Resource objects     (Agent, Session, File, Skill, …)
    4. Unified Message      (replaces 20+ individual server-event classes)
    5. Client event helpers (user_message, user_interrupt, …)
    6. Backward compatibility aliases
"""

from __future__ import annotations

from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from dashscope.agentstudio.constants import (
    SSEEventType,
    MessageRole,
    BlockType,
)


# ===========================================================================
# Section 0 – Base model
# ===========================================================================


class BaseModel:
    """Lightweight typed container.

    Subclasses declare the attributes they expect via :attr:`_fields` so
    :meth:`to_dict` can emit a deterministic ordering. Extra fields are
    preserved in :attr:`extra` to maintain forward compatibility when the
    server adds new fields.
    """

    _fields: ClassVar[Iterable[str]] = ()

    def __init__(self, **kwargs: Any) -> None:
        self._raw: Dict[str, Any] = dict(kwargs)
        for name in self._fields:
            setattr(self, name, kwargs.get(name))
        self.extra: Dict[str, Any] = {
            k: v for k, v in kwargs.items() if k not in set(self._fields)
        }

    @classmethod
    def from_dict(
        cls,
        payload: Optional[Mapping[str, Any]],
    ) -> Optional["BaseModel"]:
        if payload is None:
            return None
        return cls(**dict(payload))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name in self._fields:
            value = getattr(self, name, None)
            if value is None:
                continue
            out[name] = _serialize(value)
        for k, v in self.extra.items():
            if v is None:
                continue
            out[k] = _serialize(v)
        return out

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        body = ", ".join(
            f"{name}={getattr(self, name)!r}"
            for name in self._fields
            if getattr(self, name, None) is not None
        )
        return f"{type(self).__name__}({body})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseModel):
            return NotImplemented
        return self.to_dict() == other.to_dict()


def _serialize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.to_dict()
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    return value


# ===========================================================================
# Section 1 – Common types
# ===========================================================================


class Metadata(BaseModel):
    """Free-form metadata bag attached to most AgentStudio resources.

    The server stores ``metadata`` as a flat string-keyed dictionary
    (max 16 keys, key length <= 64, value length <= 512). The SDK does
    not validate these limits client-side – the server returns
    ``invalid_request_error`` if they are exceeded.
    """

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._raw)


class Scope(BaseModel):
    """Resource scope (organization / session)."""

    _fields = ("type", "id")


class Mount(BaseModel):
    """Session resource mount descriptor (create-time input on POST /sessions)."""

    _fields = ("type", "file_id", "skill_id", "mount_path")


class SessionResource(BaseModel):
    """A resource mounted into a session at runtime.

    The response ``file_id`` is the session-scoped copy id — **not** the
    source file id passed at mount time. ``mount_path`` is the absolute
    sandbox path (the server prepends a prefix; the user-supplied path is
    preserved losslessly and must be under ``/uploads/``).
    """

    _fields = (
        "id",
        "type",
        "file_id",
        "mount_path",
        "created_at",
        "updated_at",
        "request_id",
    )


class Networking(BaseModel):
    _fields = ("type",)  # "unrestricted" | "restricted"


class Packages(BaseModel):
    _fields = ("apt", "gem", "pip", "cargo", "go", "npm")


class EnvironmentConfig(BaseModel):
    _fields = ("type", "networking", "packages")

    def __init__(self, **kwargs: Any) -> None:
        if "networking" in kwargs and isinstance(kwargs["networking"], dict):
            kwargs["networking"] = Networking(**kwargs["networking"])
        if "packages" in kwargs and isinstance(kwargs["packages"], dict):
            kwargs["packages"] = Packages(**kwargs["packages"])
        super().__init__(**kwargs)


class StopReason(BaseModel):
    """Carried by ``session_status`` idle events (and ``Session.stop_reason``).

    A running Session's ``stop_reason`` is ``null``. ``requires_action``
    carries ``pending_batch_id`` + ``pending_call_ids`` (only the calls not
    yet adjudicated); do not derive pending from request/response diffs.
    """

    _fields = ("type", "pending_batch_id", "pending_call_ids")


class PermissionPolicy(BaseModel):
    """Tool approval policy (``permission_policy`` on tool configs).

    Must be the object form ``{"type": "always_allow" | "always_ask"}``;
    the string form is rejected (the server rejects it as well). Defaults
    to ``always_allow`` when omitted.
    """

    _fields = ("type",)

    def __init__(self, **kwargs: Any) -> None:
        t = kwargs.get("type")
        if isinstance(t, str) and t not in ("always_allow", "always_ask"):
            raise ValueError(
                "permission_policy.type must be 'always_allow' or "
                "'always_ask'",
            )
        if t is None:
            kwargs["type"] = "always_allow"
        super().__init__(**kwargs)


class Stats(BaseModel):
    """Session runtime statistics."""

    _fields = ("active_seconds", "duration_seconds")


class Usage(BaseModel):
    """Token usage carried by ``span.model_request_end`` events."""

    _fields = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "cache_creation",
        "speed",
    )


# ===========================================================================
# Section 2 – Content blocks
# ===========================================================================


class TextBlock(BaseModel):
    _fields = ("type", "text", "citations")

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("type", BlockType.TEXT)
        super().__init__(**kwargs)

    def __str__(self) -> str:
        return getattr(self, "text", "") or ""


class ImageBlock(BaseModel):
    _fields = ("type", "image_url", "file_id", "image_data", "media_type")

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("type", "image")
        super().__init__(**kwargs)

    def __str__(self) -> str:
        url = getattr(self, "image_url", "") or ""
        fid = getattr(self, "file_id", "") or ""
        return f"[image] {url or fid}"


class AudioBlock(BaseModel):
    _fields = ("type", "data", "format", "file_id")

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("type", "audio")
        super().__init__(**kwargs)

    def __str__(self) -> str:
        return f"[audio] {getattr(self, 'file_id', '') or ''}"


class DataBlock(BaseModel):
    _fields = ("type", "data", "name", "title", "context")

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("type", "data")
        super().__init__(**kwargs)

    def __str__(self) -> str:
        d = getattr(self, "data", None) or {}
        if not isinstance(d, dict):
            return str(d)
        name = d.get("name", "")
        args = d.get("arguments", "")
        output = d.get("output", "")
        status = d.get("session_status", "")
        if name and args:
            return f"{name}({args})"
        if output:
            return str(output)
        if status:
            return status
        return str(d) if d else ""


class FileBlock(BaseModel):
    _fields = (
        "type",
        "file_url",
        "file_id",
        "file_data",
        "media_type",
        "filename",
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("type", "file")
        super().__init__(**kwargs)

    def __str__(self) -> str:
        name = getattr(self, "filename", "") or ""
        fid = getattr(self, "file_id", "") or ""
        return f"[file] {name or fid}"


class RefusalBlock(BaseModel):
    _fields = ("type", "refusal")

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("type", "refusal")
        super().__init__(**kwargs)

    def __str__(self) -> str:
        return f"[refusal] {getattr(self, 'refusal', '')}"


class ErrorBlock(BaseModel):
    _fields = ("type", "error_code", "message")

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("type", BlockType.ERROR)
        super().__init__(**kwargs)

    def __str__(self) -> str:
        msg = getattr(self, "message", "") or ""
        code = getattr(self, "error_code", "") or ""
        return f"[error] {msg or code}"


_CONTENT_REGISTRY: Dict[str, type] = {
    BlockType.TEXT: TextBlock,
    BlockType.IMAGE: ImageBlock,
    BlockType.AUDIO: AudioBlock,
    BlockType.DATA: DataBlock,
    BlockType.FILE: FileBlock,
    BlockType.REFUSAL: RefusalBlock,
    BlockType.ERROR: ErrorBlock,
    SSEEventType.TOOL_CALL: DataBlock,
    SSEEventType.TOOL_CALL_OUTPUT: DataBlock,
    SSEEventType.TOOL_APPROVAL_REQUEST: DataBlock,
    SSEEventType.SESSION_STATUS: DataBlock,
    SSEEventType.REASONING: DataBlock,
    SSEEventType.MCP_CALL: DataBlock,
    SSEEventType.MCP_CALL_OUTPUT: DataBlock,
    SSEEventType.FUNCTION_CALL: DataBlock,
    SSEEventType.FUNCTION_CALL_OUTPUT: DataBlock,
}

ContentBlock = Union[
    TextBlock,
    ImageBlock,
    AudioBlock,
    DataBlock,
    FileBlock,
    RefusalBlock,
    ErrorBlock,
]


def parse_content_block(payload: Mapping[str, Any]) -> ContentBlock:
    cls = _CONTENT_REGISTRY.get(payload.get("type", ""), TextBlock)
    return cls(**dict(payload))


def parse_content_blocks(
    items: Optional[List[Mapping[str, Any]]],
) -> List[ContentBlock]:
    if not items:
        return []
    return [parse_content_block(it) for it in items if isinstance(it, Mapping)]


# ===========================================================================
# Section 3 – Resource objects
# ===========================================================================


class MultiAgentRosterEntry(BaseModel):
    """One entry in a coordinator agent's multiagent roster.

    ``type`` is ``"agent"`` (reference another agent by ``id`` + optional
    ``version``) or ``"self"`` (a copy of the coordinator; at most one).
    ``name`` / ``description`` are populated by the server on retrieval
    (enriched from the referenced agent); they are ignored on write.
    """

    _fields = ("type", "id", "version", "name", "description")

    def __init__(self, **kwargs: Any) -> None:
        if not kwargs.get("type"):
            kwargs["type"] = "agent"
        if kwargs.get("type") == "self":
            # A self-reference has no id/version of its own.
            kwargs.pop("id", None)
            kwargs.pop("version", None)
        super().__init__(**kwargs)


class MultiAgentConfig(BaseModel):
    """Multi-agent coordinator config (the ``multiagent`` field).

    ``type`` is currently always ``"coordinator"``; ``agents`` is the
    roster of entries (the server enforces the 1-20 size limit and the
    at-most-one ``"self"`` rule; the SDK normalizes but does not reject
    server-returned data, so it never fails to parse a valid agent).
    An empty list clears the roster. The agent version is snapshotted when
    a session is created; changes only affect new sessions.
    """

    _fields = ("type", "agents")

    def __init__(self, **kwargs: Any) -> None:
        if not kwargs.get("type"):
            kwargs["type"] = "coordinator"
        agents = kwargs.get("agents")
        if isinstance(agents, list):
            kwargs["agents"] = [
                (
                    MultiAgentRosterEntry(**dict(a))
                    if isinstance(a, Mapping)
                    else a
                )
                for a in agents
            ]
        elif agents is None:
            kwargs["agents"] = []
        super().__init__(**kwargs)


class Agent(BaseModel):
    _fields = (
        "id",
        "type",
        "version",
        "name",
        "description",
        "model",
        "system",
        "tools",
        "mcp_servers",
        "skills",
        "multiagent",
        "metadata",
        "workspace_id",
        "archived_at",
        "created_at",
        "updated_at",
        "request_id",
    )

    def __init__(self, **kwargs: Any) -> None:
        multiagent = kwargs.get("multiagent")
        if isinstance(multiagent, Mapping):
            kwargs["multiagent"] = MultiAgentConfig(**dict(multiagent))
        super().__init__(**kwargs)

    @property
    def system_prompt(self) -> Optional[str]:
        """Alias: server field is ``system``, kept for SDK user convenience."""
        return self.system


class AgentVersion(BaseModel):
    _fields = (
        "agent_id",
        "version",
        "config",
        "created_at",
    )


class Environment(BaseModel):
    _fields = (
        "id",
        "type",
        "name",
        "description",
        "config",
        "metadata",
        "scope",
        "archived_at",
        "created_at",
        "updated_at",
        "request_id",
    )

    def __init__(self, **kwargs: Any) -> None:
        cfg = kwargs.get("config")
        if isinstance(cfg, Mapping):
            kwargs["config"] = EnvironmentConfig(**dict(cfg))
        super().__init__(**kwargs)


class File(BaseModel):
    _fields = (
        "id",
        "type",
        "filename",
        "downloadable",
        "mime_type",
        "size_bytes",
        "status",
        "created_at",
        "request_id",
    )


class Skill(BaseModel):
    _fields = (
        "id",
        "type",
        "name",
        "description",
        "source",
        "status",
        "latest_version",
        "version",  # backward compat alias
        "file_id",
        "scope",
        "created_at",
        "updated_at",
        "request_id",
    )


class SkillVersion(BaseModel):
    _fields = (
        "id",
        "type",
        "skill_id",
        "name",
        "description",
        "version",
        "status",
        "additional_properties",
        "created_at",
        "updated_at",
    )


class Session(BaseModel):
    _fields = (
        "id",
        "type",
        "title",
        "agent",
        "environment_id",
        "status",
        "stop_reason",
        "resources",
        "metadata",
        "stats",
        "usage",
        "vault_ids",
        "archived_at",
        "created_at",
        "updated_at",
        "request_id",
    )

    def __init__(self, **kwargs: Any) -> None:
        agent = kwargs.get("agent")
        if isinstance(agent, Mapping):
            kwargs["agent"] = Agent(**dict(agent))
        sr = kwargs.get("stop_reason")
        if isinstance(sr, Mapping):
            kwargs["stop_reason"] = StopReason(**dict(sr))
        usage = kwargs.get("usage")
        if isinstance(usage, Mapping):
            kwargs["usage"] = Usage(**dict(usage))
        stats = kwargs.get("stats")
        if isinstance(stats, Mapping):
            kwargs["stats"] = Stats(**dict(stats))
        super().__init__(**kwargs)

    @property
    def agent_id(self) -> Optional[str]:
        a = self.agent
        if isinstance(a, Agent):
            return a.id
        return a if isinstance(a, str) else None

    @property
    def agent_version(self) -> Optional[int]:
        a = self.agent
        return a.version if isinstance(a, Agent) else None


class SessionThread(BaseModel):
    """A sub-agent thread within a session.

    ``agent`` is a ``{id, version}`` reference to the thread's bound agent.
    ``status`` is ``idle`` / ``running`` / ``terminated``; ``archived_at``
    is non-null once archived (archived threads are excluded from list by
    default).
    """

    _fields = (
        "id",
        "type",
        "session_id",
        "parent_thread_id",
        "agent",
        "status",
        "created_at",
        "updated_at",
        "archived_at",
        "request_id",
    )


# ===========================================================================
# Security (overview + agent logs)
# ===========================================================================


class SecurityCapability(BaseModel):
    """A single capability/protection switch in the overview."""

    _fields = ("key", "enabled")


class SecurityScanStat(BaseModel):
    """Scan hit/scanned counters (content_safety / file_scan / skill_scan)."""

    _fields = ("hit", "scanned")


class SecurityOverview(BaseModel):
    """Response of ``GET /security/overview`` (last-24h dashboard)."""

    _fields = (
        "capabilities",
        "protection",
        "content_safety",
        "file_scan",
        "skill_scan",
        "request_id",
    )

    def __init__(self, **kwargs: Any) -> None:
        for k in ("capabilities", "protection"):
            v = kwargs.get(k)
            if isinstance(v, list):
                kwargs[k] = [
                    SecurityCapability(**dict(it))
                    if isinstance(it, Mapping)
                    else it
                    for it in v
                ]
        for k in ("content_safety", "file_scan", "skill_scan"):
            v = kwargs.get(k)
            if isinstance(v, Mapping):
                kwargs[k] = SecurityScanStat(**dict(v))
        super().__init__(**kwargs)


class SecurityAlertStats(BaseModel):
    """Alert counts by severity."""

    _fields = ("total", "high", "medium", "low")


class SecurityAlert(BaseModel):
    """A single security alert row. ``check_time`` / ``handle_time`` are
    millisecond-string timestamps (not ISO 8601)."""

    _fields = (
        "alert_id",
        "risk_level",
        "risk_name",
        "risk_desc",
        "asset_type",
        "asset_name",
        "app_id",
        "app_name",
        "agent_name",
        "status",
        "source",
        "check_time",
        "handle_time",
        "vendor",
    )


class SecurityAlertList(BaseModel):
    """Response of ``GET /security/agent_logs`` (page-number + cursor)."""

    _fields = ("stats", "data", "next_page", "request_id")

    def __init__(self, **kwargs: Any) -> None:
        stats = kwargs.get("stats")
        if isinstance(stats, Mapping):
            kwargs["stats"] = SecurityAlertStats(**dict(stats))
        data = kwargs.get("data")
        if isinstance(data, list):
            kwargs["data"] = [
                SecurityAlert(**dict(it)) if isinstance(it, Mapping) else it
                for it in data
            ]
        super().__init__(**kwargs)


class DeleteResponse(BaseModel):
    _fields = ("id", "type", "request_id")


class WebhookEndpoint(BaseModel):
    """Managed Agent webhook endpoint."""

    _fields = (
        "id",
        "description",
        "url",
        "events",
        "status",
        "disabled_reason",
        "consecutive_fail",
        "last_success_at",
        "last_failure_at",
        "signing_secret",
        "created_at",
        "updated_at",
        "request_id",
    )


class WebhookEndpointList(BaseModel):
    """Non-paginated list of webhook endpoints in the current workspace."""

    _fields = ("data", "request_id")

    def __init__(self, **kwargs: Any) -> None:
        data = kwargs.get("data")
        if isinstance(data, list):
            kwargs["data"] = [
                WebhookEndpoint(**dict(item))
                if isinstance(item, Mapping)
                else item
                for item in data
            ]
        super().__init__(**kwargs)


class WebhookSecretReset(BaseModel):
    """Result returned after resetting a webhook signing secret."""

    _fields = ("id", "signing_secret", "updated_at", "request_id")


class WebhookEventData(BaseModel):
    """Resource data carried by a webhook event envelope."""

    _fields = (
        "id",
        "type",
        "workspace_id",
        "session_thread_id",
        "vault_id",
        "extensions",
    )


class WebhookDelivery(BaseModel):
    """Delivery information attached to an endpoint event."""

    _fields = (
        "webhook_id",
        "status",
        "attempt_count",
        "delivery_at",
        "finish_at",
        "failure_reason",
    )


class WebhookEvent(BaseModel):
    """Webhook event envelope returned by test and endpoint event APIs."""

    _fields = (
        "type",
        "id",
        "created_at",
        "data",
        "delivery",
        "request_id",
    )

    def __init__(self, **kwargs: Any) -> None:
        data = kwargs.get("data")
        if isinstance(data, Mapping):
            kwargs["data"] = WebhookEventData(**dict(data))
        delivery = kwargs.get("delivery")
        if isinstance(delivery, Mapping):
            kwargs["delivery"] = WebhookDelivery(**dict(delivery))
        super().__init__(**kwargs)


class DeploymentAgentReference(BaseModel):
    """Agent version reference stored on a deployment run."""

    _fields = ("id", "version")


class DeploymentSchedule(BaseModel):
    """Cron schedule returned with a deployment."""

    _fields = (
        "type",
        "expression",
        "timezone",
        "last_run_at",
        "next_run_at",
    )


class DeploymentResource(BaseModel):
    """Resource mounted into sessions created by a deployment."""

    _fields = ("type", "file_id", "mount_path")


class DeploymentError(BaseModel):
    """Error information attached to a failed run or automatic pause."""

    _fields = ("code", "message")


class DeploymentPausedReason(BaseModel):
    """Why a deployment is paused (``manual`` or ``error``)."""

    _fields = ("type", "error")

    def __init__(self, **kwargs: Any) -> None:
        error = kwargs.get("error")
        if isinstance(error, Mapping):
            kwargs["error"] = DeploymentError(**dict(error))
        super().__init__(**kwargs)


class Deployment(BaseModel):
    """Managed Agent deployment."""

    environment_variables: Optional[Dict[str, str]]
    metadata: Optional[Dict[str, str]]

    _fields = (
        "id",
        "type",
        "name",
        "description",
        "agent",
        "environment_id",
        "schedule",
        "initial_events",
        "resources",
        "vault_ids",
        "environment_variables",
        "metadata",
        "status",
        "paused_reason",
        "archived_at",
        "created_at",
        "updated_at",
        "request_id",
    )

    def __init__(self, **kwargs: Any) -> None:
        agent = kwargs.get("agent")
        if isinstance(agent, Mapping):
            kwargs["agent"] = Agent(**dict(agent))
        schedule = kwargs.get("schedule")
        if isinstance(schedule, Mapping):
            kwargs["schedule"] = DeploymentSchedule(**dict(schedule))
        resources = kwargs.get("resources")
        if isinstance(resources, list):
            kwargs["resources"] = [
                DeploymentResource(**dict(resource))
                if isinstance(resource, Mapping)
                else resource
                for resource in resources
            ]
        paused_reason = kwargs.get("paused_reason")
        if isinstance(paused_reason, Mapping):
            kwargs["paused_reason"] = DeploymentPausedReason(
                **dict(paused_reason),
            )
        super().__init__(**kwargs)


class DeploymentRun(BaseModel):
    """A single manual or scheduled deployment execution."""

    _fields = (
        "id",
        "type",
        "deployment_id",
        "agent",
        "session_id",
        "trigger_source",
        "status",
        "error",
        "started_at",
        "finished_at",
        "request_id",
    )

    def __init__(self, **kwargs: Any) -> None:
        agent = kwargs.get("agent")
        if isinstance(agent, Mapping):
            kwargs["agent"] = DeploymentAgentReference(**dict(agent))
        error = kwargs.get("error")
        if isinstance(error, Mapping):
            kwargs["error"] = DeploymentError(**dict(error))
        super().__init__(**kwargs)


# ===========================================================================
# Section 4 – Unified Message class  (replaces server_events.py)
# ===========================================================================


class Message(BaseModel):
    """Unified event/message model corresponding to the AgentStudio API
    protocol Message structure.

    All SSE stream events and history events share this structure.
    The ``type`` field discriminates between event kinds::

        message, tool_call, tool_call_output, function_call,
        function_call_output, mcp_call, mcp_call_output, reasoning,
        session_status, error, session_updated, thread_created,
        thread_status, thread_message_sent, thread_message_received,
        thread_context_compacted, model_request_start, model_request_end,
        outcome_evaluation, interrupt, tool_confirmation, define_outcome

    Legacy ``agent.*`` / ``session.*`` / ``span.*`` type strings are also
    accepted transparently for backward compatibility.
    """

    _fields = (
        "object",
        "status",
        "id",
        "type",
        "role",
        "content",
        "metadata",
        "is_error",
        "created_at",
        "sequence_number",
        "session_thread_id",
        "thread_id",
        "code",
        "message",
    )

    def __init__(self, **kwargs: Any) -> None:
        content = kwargs.get("content")
        if isinstance(content, list):
            kwargs["content"] = parse_content_blocks(content)
        super().__init__(**kwargs)

    def __str__(self) -> str:
        etype = getattr(self, "type", "unknown")
        content = self.content or []
        if content:
            return f"[{etype}] " + " ".join(str(b) for b in content)
        return f"[{etype}]"

    @property
    def stop_reason(self) -> Optional[Dict[str, Any]]:
        """``stop_reason`` carried by ``session_status`` idle events.

        The server puts ``stop_reason`` inside the ``session_status`` event's
        data block (not in ``session.retrieve()``).  Returns a dict like
        ``{"type": "end_turn"}`` or ``None`` for non-session_status events.
        """
        if getattr(self, "type", None) != SSEEventType.SESSION_STATUS:
            return None
        for block in self.content or []:
            d = getattr(block, "data", None)
            if isinstance(d, dict) and "stop_reason" in d:
                return d["stop_reason"]
        return None

    @property
    def session_status(self) -> Optional[str]:
        """``session_status`` value from ``session_status`` events.

        Returns ``"idle"``, ``"running"``, ``"rescheduled"``,
        ``"terminated"``, ``"deleted"`` or ``None`` for non-session_status
        events.
        """
        if getattr(self, "type", None) != SSEEventType.SESSION_STATUS:
            return None
        for block in self.content or []:
            d = getattr(block, "data", None)
            if isinstance(d, dict) and "session_status" in d:
                return d["session_status"]
        return None

    @property
    def _data(self) -> Optional[Dict[str, Any]]:
        """The first content block's ``data`` payload, if any."""
        for block in self.content or []:
            d = getattr(block, "data", None)
            if isinstance(d, dict):
                return d
        return None

    @property
    def tool_approval_request(self) -> Optional[Dict[str, Any]]:
        """``tool_approval_request`` payload: ``batch_id`` / ``call_id`` /
        ``name`` / ``arguments`` (JSON string) / ``tool_type`` /
        ``server_label`` (MCP only).

        Returns ``None`` for non-``tool_approval_request`` events. The
        approval identity is the ``(batch_id, call_id)`` composite key —
        ``call_id`` may be reused across turns, so never match on
        ``call_id`` alone.
        """
        if getattr(self, "type", None) != SSEEventType.TOOL_APPROVAL_REQUEST:
            return None
        return self._data

    @property
    def error(self) -> Optional[Dict[str, Any]]:
        """``{"code", "message"}`` from ``type: error`` events, else ``None``.

        Approval failures surface in the event stream as ``type: error``
        events (not as HTTP exceptions). Use :attr:`pending_tool_approvals`
        to read the suspend signal alongside this.
        """
        if getattr(self, "type", None) != SSEEventType.ERROR:
            return None
        err = self.extra.get("error")
        if err is None:
            raw = getattr(self, "_raw", None) or {}
            if isinstance(raw, Mapping):
                err = raw.get("error")
        return err if isinstance(err, dict) else None

    @property
    def pending_tool_approvals(self) -> Optional[Dict[str, Any]]:
        """Suspend signal ``{"batch_id", "call_ids"}`` carried in the
        ``metadata`` of an error / response frame while the approval
        barrier is up. ``None`` when the barrier is not up.

        This is the reliable way to tell pending state — do NOT derive it
        from request/response event diffs (the server emits it only while
        the barrier stands).
        """
        md = self.metadata if isinstance(self.metadata, dict) else None
        if md is None:
            return None
        pending = md.get("pending_tool_approvals")
        return pending if isinstance(pending, dict) else None

    # -- delta-protocol frames (opt-in via ``event_deltas``) -------------
    # ``event_start`` / ``event_delta`` carry incremental text when the
    # stream is opened with ``event_deltas``; a terminal ``object:"message"``
    # event always follows with the full content. They are distinct from the
    # business event types carried in ``type``.

    @property
    def event_start(self) -> Optional[Dict[str, Any]]:
        """``{"id", "type"}`` from an ``event_start`` delta frame (a preview
        of an upcoming ``message``/``reasoning`` event; carries no content).
        ``None`` for other frames."""
        if getattr(self, "type", None) != "event_start":
            return None
        ev = self.extra.get("event")
        if ev is None:
            raw = getattr(self, "_raw", None) or {}
            if isinstance(raw, Mapping):
                ev = raw.get("event")
        return ev if isinstance(ev, dict) else None

    @property
    def event_delta(self) -> Optional[Dict[str, Any]]:
        """``{"event_id", "delta": {"type", "index", "content"}}`` from an
        ``event_delta`` frame. ``None`` for other frames."""
        if getattr(self, "type", None) != "event_delta":
            return None
        return {
            "event_id": self.extra.get("event_id"),
            "delta": self.extra.get("delta"),
        }

    @property
    def delta_text(self) -> Optional[str]:
        """Incremental text chunk from an ``event_delta`` frame (the
        ``delta.content.text`` of a ``content_delta``), else ``None``.
        Use the :attr:`text_deltas` iterator on the stream for the full
        sequence."""
        if getattr(self, "type", None) != "event_delta":
            return None
        delta = self.extra.get("delta") or {}
        if not isinstance(delta, dict):
            return None
        content = delta.get("content") or {}
        text = content.get("text") if isinstance(content, dict) else None
        return text

    # -- business event data accessors -----------------------------------
    # Convenience accessors over the first content block's ``data`` payload
    # (and ``metadata`` where the routing lives). Each returns ``None`` for
    # events whose ``type`` does not match.

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        """The first content block's ``data`` payload, for events that carry
        one (``tool_call`` / ``tool_call_output`` / ``mcp_call`` /
        ``mcp_call_output`` / ``session_status`` / ``tool_approval_request``
        / ``model_request_end`` / ``outcome_evaluation`` / ``thread_status``
        / ``thread_created`` / ``session_updated``). ``None`` otherwise.
        """
        return self._data

    @property
    def model_request_end(self) -> Optional[Dict[str, Any]]:
        """``model_request_end`` payload: ``model_request_start_id``,
        ``is_error``, ``input_tokens`` / ``output_tokens`` /
        ``cache_creation_input_tokens`` / ``cache_read_input_tokens``,
        ``speed``. ``None`` for other events."""
        if getattr(self, "type", None) != SSEEventType.MODEL_REQUEST_END:
            return None
        return self._data

    @property
    def outcome_evaluation(self) -> Optional[Dict[str, Any]]:
        """``outcome_evaluation`` payload: ``outcome_id``, ``iteration``,
        ``phase`` (start/ongoing/end), ``result``, ``explanation``,
        token usage, ``speed``. ``None`` for other events."""
        if getattr(self, "type", None) != SSEEventType.OUTCOME_EVALUATION:
            return None
        return self._data

    @property
    def thread_status(self) -> Optional[Dict[str, Any]]:
        """``thread_status`` payload: ``session_thread_id``, ``agent_name``,
        ``thread_status`` (running/idle/terminated/rescheduled), and
        ``stop_reason`` when idle. ``None`` for other events."""
        if getattr(self, "type", None) != SSEEventType.THREAD_STATUS:
            return None
        return self._data

    @property
    def thread_created(self) -> Optional[Dict[str, Any]]:
        """``thread_created`` payload: ``session_thread_id``, ``agent_name``.
        ``None`` for other events."""
        if getattr(self, "type", None) != SSEEventType.THREAD_CREATED:
            return None
        return self._data

    @property
    def session_updated(self) -> Optional[Dict[str, Any]]:
        """``session_updated`` payload: ``title``, ``session_metadata``,
        ``agent`` (only the changed fields, present when changed).
        ``None`` for other events."""
        if getattr(self, "type", None) != SSEEventType.SESSION_UPDATED:
            return None
        return self._data

    @property
    def thread_message_routing(self) -> Optional[Dict[str, str]]:
        """Sub-agent routing from ``thread_message_sent`` / ``thread_message_received``
        ``metadata``: ``to_session_thread_id`` / ``to_agent_name`` on sent,
        ``from_session_thread_id`` / ``from_agent_name`` on received.
        ``None`` for other events."""
        if getattr(self, "type", None) not in (
            SSEEventType.THREAD_MESSAGE_SENT,
            SSEEventType.THREAD_MESSAGE_RECEIVED,
        ):
            return None
        md = self.metadata if isinstance(self.metadata, dict) else {}
        keys = (
            "to_session_thread_id",
            "to_agent_name",
            "from_session_thread_id",
            "from_agent_name",
        )
        return {k: md[k] for k in keys if k in md} or None


def parse_message(payload: Mapping[str, Any]) -> Message:
    """Turn a parsed SSE ``data`` dict into a :class:`Message` instance."""
    return Message(**dict(payload))


# Keep the original function name as an alias so existing code that calls
# ``parse_server_event`` continues to work unchanged.
def parse_server_event(payload: Mapping[str, Any]) -> Message:
    """Alias for :func:`parse_message`; kept for backward compatibility."""
    return parse_message(payload)


# ===========================================================================
# Section 5 – Client event helpers  (AGENTSTUDIO wire format)
# ===========================================================================


def user_message(
    text_or_blocks: Any,
    *,
    session_thread_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a ``message`` client event (AGENTSTUDIO wire format).

    ``text_or_blocks`` may be a plain string (wrapped as a text block)
    or a list of pre-built content block dicts.
    """

    if isinstance(text_or_blocks, str):
        content: List[Dict[str, Any]] = [
            {"type": BlockType.TEXT, "text": text_or_blocks},
        ]
    else:
        content = list(text_or_blocks)
    evt: Dict[str, Any] = {
        "role": MessageRole.USER,
        "type": SSEEventType.MESSAGE,
        "content": content,
    }
    if session_thread_id:
        evt["session_thread_id"] = session_thread_id
    if metadata:
        evt["metadata"] = dict(metadata)
    return evt


def user_interrupt(
    *,
    session_thread_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Cancel the agent's current turn (best-effort)."""

    evt: Dict[str, Any] = {
        "role": MessageRole.USER,
        "type": SSEEventType.INTERRUPT,
    }
    if session_thread_id:
        evt["session_thread_id"] = session_thread_id
    if metadata:
        evt["metadata"] = dict(metadata)
    return evt


def user_tool_approval_response(
    *,
    batch_id: str,
    call_id: str,
    result: str,
    deny_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Submit a tool approval ruling for an ``always_ask`` tool call.

    ``result`` must be ``"allow"`` or ``"deny"``; ``deny_message`` is
    optional and only meaningful when denying. The approval identity is the
    ``(batch_id, call_id)`` composite key — ``call_id`` may be reused across
    turns, so never match on ``call_id`` alone. Approval responses target
    the primary thread only (no ``session_thread_id``); the legacy
    ``tool_confirmation`` type is rejected by the server (HTTP 400
    ``bma_invalid_event``).
    """
    if result not in ("allow", "deny"):
        raise ValueError("result must be 'allow' or 'deny'")
    data: Dict[str, Any] = {
        "batch_id": batch_id,
        "call_id": call_id,
        "result": result,
    }
    if deny_message and result == "deny":
        data["deny_message"] = deny_message
    return {
        "role": MessageRole.USER,
        "type": SSEEventType.TOOL_APPROVAL_RESPONSE,
        "content": [{"type": "data", "data": data}],
    }


def user_custom_tool_result(
    *,
    custom_tool_use_id: str,
    content: Any,
    is_error: bool = False,
    session_thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Reply to an :class:`AgentCustomToolUseEvent` with a tool result.

    Emits a ``function_call_output`` event in the AGENTSTUDIO wire format.
    """

    if isinstance(content, str):
        output = content
    elif isinstance(content, Sequence) and not isinstance(content, str):
        output = list(content)
    else:
        output = content
    evt: Dict[str, Any] = {
        "role": MessageRole.TOOL,
        "type": SSEEventType.FUNCTION_CALL_OUTPUT,
        "content": [
            {
                "type": "data",
                "data": {
                    "call_id": custom_tool_use_id,
                    "output": output,
                },
            },
        ],
        "is_error": bool(is_error),
    }
    if session_thread_id:
        evt["session_thread_id"] = session_thread_id
    return evt


def user_tool_result(
    *,
    tool_use_id: str,
    content: Any,
    is_error: bool = False,
    session_thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Self-hosted tool result (advanced; mirrors built-in tool execution).

    Emits a ``tool_call_output`` event in the AGENTSTUDIO wire format.
    """

    if isinstance(content, str):
        output = content
    elif isinstance(content, Sequence) and not isinstance(content, str):
        output = list(content)
    else:
        output = content
    evt: Dict[str, Any] = {
        "role": MessageRole.TOOL,
        "type": SSEEventType.TOOL_CALL_OUTPUT,
        "content": [
            {
                "type": "data",
                "data": {
                    "call_id": tool_use_id,
                    "output": output,
                },
            },
        ],
        "is_error": bool(is_error),
    }
    if session_thread_id:
        evt["session_thread_id"] = session_thread_id
    return evt


def user_define_outcome(
    *,
    description: str,
    rubric: Optional[str] = None,
    max_iterations: Optional[int] = None,
    session_thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Steer the agent towards a measurable outcome."""

    data: Dict[str, Any] = {"description": description}
    if rubric is not None:
        data["rubric"] = rubric
    if max_iterations is not None:
        data["max_iterations"] = int(max_iterations)
    evt: Dict[str, Any] = {
        "role": MessageRole.USER,
        "type": SSEEventType.DEFINE_OUTCOME,
        "content": [{"type": "data", "data": data}],
    }
    if session_thread_id:
        evt["session_thread_id"] = session_thread_id
    return evt


# ===========================================================================
# Section 6 – Vaults & Credentials
# ===========================================================================


class CredentialAuth(BaseModel):
    _fields = (
        "type",
        "access_token",
        "mcp_server_url",
        "expires_at",
        "refresh",
        "token",
        "secret_name",
        "secret_value",
        "networking",
    )

    def __init__(self, **kwargs: Any) -> None:
        net = kwargs.get("networking")
        if isinstance(net, dict):
            kwargs["networking"] = Networking(**net)
        super().__init__(**kwargs)


class Vault(BaseModel):
    _fields = (
        "id",
        "archived_at",
        "created_at",
        "display_name",
        "metadata",
        "type",
        "updated_at",
        "request_id",
    )


class Credential(BaseModel):
    _fields = (
        "id",
        "archived_at",
        "auth",
        "created_at",
        "display_name",
        "metadata",
        "type",
        "updated_at",
        "vault_id",
        "request_id",
    )

    def __init__(self, **kwargs: Any) -> None:
        auth = kwargs.get("auth")
        if isinstance(auth, dict):
            kwargs["auth"] = CredentialAuth(**auth)
        super().__init__(**kwargs)


# ===========================================================================
# Section 7 – Backward compatibility aliases
# ===========================================================================

# server_events.py  ──  ServerEvent was the base class; now it's Message
ServerEvent = Message
UnknownServerEvent = Message

# All the concrete subclasses from server_events.py map to Message
AgentMessageEvent = Message
AgentThinkingEvent = Message
AgentToolUseEvent = Message
AgentToolResultEvent = Message
AgentCustomToolUseEvent = Message
AgentCustomToolResultEvent = Message
AgentMcpToolUseEvent = Message
AgentMcpToolResultEvent = Message
AgentThreadMessageSentEvent = Message
AgentThreadMessageReceivedEvent = Message
AgentThreadContextCompactedEvent = Message
SessionStatusRunningEvent = Message
SessionStatusIdleEvent = Message
SessionStatusReschedulingEvent = Message
SessionStatusTerminatedEvent = Message
SessionErrorEvent = Message
SessionUpdatedEvent = Message
SessionThreadCreatedEvent = Message
SessionThreadStatusEvent = Message
SpanModelRequestStartEvent = Message
SpanModelRequestEndEvent = Message
