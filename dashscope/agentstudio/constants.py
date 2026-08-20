# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AgentStudio wire-protocol and configuration constants."""

import enum
import sys

import httpx

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, enum.Enum):  # type: ignore[no-redef]
        """Minimal StrEnum shim for Python < 3.11."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_HOST = "https://{workspace}.{region}.maas.aliyuncs.com"
AGENTSTUDIO_BASE_URL_TEMPLATE = _HOST + "/api/v1/agentstudio"
AGENTSTUDIO_DEFAULT_REGION = "cn-beijing"
AGENTSTUDIO_DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
AGENTSTUDIO_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Wire-protocol enums
# ---------------------------------------------------------------------------


class SSEEventType(StrEnum):
    """Server-sent event types (the value of ``event.type`` in SSE payloads).

    Client-sendable: MESSAGE, INTERRUPT, TOOL_CONFIRMATION,
    FUNCTION_CALL_OUTPUT, TOOL_CALL_OUTPUT, DEFINE_OUTCOME.
    Server-emitted: all types.
    """

    # Client-sendable
    MESSAGE = "message"
    INTERRUPT = "interrupt"
    TOOL_CONFIRMATION = "tool_confirmation"
    FUNCTION_CALL_OUTPUT = "function_call_output"
    TOOL_CALL_OUTPUT = "tool_call_output"
    DEFINE_OUTCOME = "define_outcome"

    # Server-emitted
    FUNCTION_CALL = "function_call"
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    MCP_CALL = "mcp_call"
    MCP_CALL_OUTPUT = "mcp_call_output"
    THREAD_MESSAGE_SENT = "thread_message_sent"
    THREAD_MESSAGE_RECEIVED = "thread_message_received"
    THREAD_CONTEXT_COMPACTED = "thread_context_compacted"
    SESSION_STATUS = "session_status"
    ERROR = "error"
    SESSION_UPDATED = "session_updated"
    THREAD_CREATED = "thread_created"
    THREAD_STATUS = "thread_status"
    MODEL_REQUEST_START = "model_request_start"
    MODEL_REQUEST_END = "model_request_end"
    OUTCOME_EVALUATION = "outcome_evaluation"


class MessageRole(StrEnum):
    """Roles used in message/event payloads."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class BlockType(StrEnum):
    """Content block types (``block.type`` values)."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DATA = "data"
    FILE = "file"
    REFUSAL = "refusal"
    ERROR = "error"


class SessionStatus(StrEnum):
    """Session run-status values (``session_status``)."""

    IDLE = "idle"
    RUNNING = "running"
    RESCHEDULING = "rescheduling"
    TERMINATED = "terminated"


class WebhookStatus(StrEnum):
    """Webhook endpoint activation status."""

    # The endpoint can receive webhook deliveries.
    ACTIVE = "ACTIVE"
    # The endpoint does not receive webhook deliveries.
    DISABLED = "DISABLED"


class WebhookDisabledReason(StrEnum):
    """Reason why a webhook endpoint was disabled."""

    # The endpoint was disabled manually.
    MANUAL = "MANUAL"
    # The endpoint reached the consecutive delivery failure threshold.
    CONSECUTIVE_DELIVERY_FAILURES = "CONSECUTIVE_DELIVERY_FAILURES"
    # The endpoint returned an HTTP redirect response.
    REDIRECT_RESPONSE = "REDIRECT_RESPONSE"
    # The endpoint failed server-side request forgery validation.
    SSRF_VALIDATION_FAILED = "SSRF_VALIDATION_FAILED"
    # The endpoint failed Transport Layer Security validation.
    TLS_VALIDATION_FAILED = "TLS_VALIDATION_FAILED"


class WebhookDeliveryStatus(StrEnum):
    """Webhook delivery status."""

    # The delivery is waiting to start.
    PENDING = "PENDING"
    # The delivery is currently being sent.
    DELIVERING = "DELIVERING"
    # The delivery is waiting for its next retry.
    WAITING_RETRY = "WAITING_RETRY"
    # The delivery completed successfully.
    SUCCEEDED = "SUCCEEDED"
    # The delivery exhausted all attempts.
    FAILED = "FAILED"
    # The delivery was canceled because its endpoint was disabled or deleted.
    CANCELED = "CANCELED"


class WebhookEventType(StrEnum):
    """Event types accepted by webhook endpoint subscriptions."""

    # A session was created.
    SESSION_CREATED = "session.created"
    # A session was updated.
    SESSION_UPDATED = "session.updated"
    # A session was archived.
    SESSION_ARCHIVED = "session.archived"
    # A session was deleted.
    SESSION_DELETED = "session.deleted"
    # A session run started.
    SESSION_STATUS_RUN_STARTED = "session.status_run_started"
    # A session entered the idle state.
    SESSION_STATUS_IDLED = "session.status_idled"
    # A session entered the terminated state.
    SESSION_STATUS_TERMINATED = "session.status_terminated"
    # A session thread was created.
    SESSION_THREAD_CREATED = "session.thread_created"
    # A session thread run started.
    SESSION_THREAD_RUN_STARTED = "session.thread_run_started"
    # A session thread entered the idle state.
    SESSION_THREAD_IDLED = "session.thread_idled"
    # A session thread entered the terminated state.
    SESSION_THREAD_TERMINATED = "session.thread_terminated"
    # An agent was created.
    AGENT_CREATED = "agent.created"
    # An agent was updated.
    AGENT_UPDATED = "agent.updated"
    # An agent was archived.
    AGENT_ARCHIVED = "agent.archived"
    # A deployment was created.
    DEPLOYMENT_CREATED = "deployment.created"
    # A deployment was updated.
    DEPLOYMENT_UPDATED = "deployment.updated"
    # A deployment was archived.
    DEPLOYMENT_ARCHIVED = "deployment.archived"
    # A deployment was paused.
    DEPLOYMENT_PAUSED = "deployment.paused"
    # A deployment was resumed.
    DEPLOYMENT_UNPAUSED = "deployment.unpaused"
    # A deployment run started.
    DEPLOYMENT_RUN_STARTED = "deployment_run.started"
    # A deployment run failed.
    DEPLOYMENT_RUN_FAILED = "deployment_run.failed"
    # A deployment run succeeded.
    DEPLOYMENT_RUN_SUCCEEDED = "deployment_run.succeeded"
    # An environment was created.
    ENVIRONMENT_CREATED = "environment.created"
    # An environment was updated.
    ENVIRONMENT_UPDATED = "environment.updated"
    # An environment was archived.
    ENVIRONMENT_ARCHIVED = "environment.archived"
    # An environment was deleted.
    ENVIRONMENT_DELETED = "environment.deleted"
    # A vault was created.
    VAULT_CREATED = "vault.created"
    # A vault was archived.
    VAULT_ARCHIVED = "vault.archived"
    # A vault was deleted.
    VAULT_DELETED = "vault.deleted"
    # A vault credential was created.
    VAULT_CREDENTIAL_CREATED = "vault_credential.created"
    # A vault credential was archived.
    VAULT_CREDENTIAL_ARCHIVED = "vault_credential.archived"
    # A vault credential was deleted.
    VAULT_CREDENTIAL_DELETED = "vault_credential.deleted"
