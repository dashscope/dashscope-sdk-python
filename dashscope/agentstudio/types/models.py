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
    """Session resource mount descriptor."""

    _fields = ("type", "file_id", "skill_id", "mount_path")


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
    """Carried by ``session_status`` idle events."""

    _fields = ("type", "event_ids")


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
        "metadata",
        "workspace_id",
        "archived_at",
        "created_at",
        "updated_at",
        "request_id",
    )

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
    _fields = (
        "id",
        "session_id",
        "parent_thread_id",
        "title",
        "status",
        "created_at",
        "updated_at",
    )


class DeleteResponse(BaseModel):
    _fields = ("id", "type", "request_id")


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

        Returns ``"idle"``, ``"running"``, ``"rescheduling"``,
        ``"terminated"`` or ``None`` for non-session_status events.
        """
        if getattr(self, "type", None) != SSEEventType.SESSION_STATUS:
            return None
        for block in self.content or []:
            d = getattr(block, "data", None)
            if isinstance(d, dict) and "session_status" in d:
                return d["session_status"]
        return None


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


def user_tool_confirmation(
    *,
    tool_use_id: str,
    result: str,
    deny_message: Optional[str] = None,
    session_thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Approve or deny a built-in tool invocation.

    ``result`` must be ``"allow"`` or ``"deny"``. ``deny_message`` is
    only meaningful when denying.
    """

    if result not in ("allow", "deny"):
        raise ValueError("tool_confirmation result must be 'allow' or 'deny'")
    data: Dict[str, Any] = {
        "call_id": tool_use_id,
        "result": result,
    }
    if deny_message and result == "deny":
        data["deny_message"] = deny_message
    evt: Dict[str, Any] = {
        "role": MessageRole.USER,
        "type": SSEEventType.TOOL_CONFIRMATION,
        "content": [{"type": "data", "data": data}],
    }
    if session_thread_id:
        evt["session_thread_id"] = session_thread_id
    return evt


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
