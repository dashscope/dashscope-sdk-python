# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The historical AgentStudio facade over the shared application core.

The ``agentstudio`` package wraps the AgentStudio managed-agent HTTP
API.  All resource classes are accessed through a client instance::

    from dashscope.applications.agentstudio import Client
    from dashscope.applications.agentstudio.types import (
        user_message, user_tool_result,
    )

Example
-------
>>> from dashscope.applications.agentstudio import Client
>>> from dashscope.applications.agentstudio.types import user_message
>>> client = Client(api_key="sk-xxx")
>>> agent = client.agents.create(name="demo", model="qwen-plus")
>>> session = client.sessions.create(agent=agent.id)
>>> client.sessions.events.send(session.id, [user_message("Hello!")])
>>> with client.sessions.events.stream(session.id) as stream:
...     for event in stream:
...         print(event.type, event.to_dict())
...         if event.type == "session_status":
...             break
"""

# yapf: disable

from dashscope.legacy_1x.agentstudio.client import Client, AsyncClient
from dashscope.legacy_1x.agentstudio.transport import APIResponse, RequestOptions
from dashscope.applications.agentstudio import types
from dashscope.legacy_1x.agentstudio.exceptions import (
    AgentStudioError,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    ConflictError,
    InternalServerError,
    InvalidRequestError,
    NotFoundError,
    OverloadedError,
    PermissionDeniedError,
    RateLimitError,
    StreamClosedError,
    StreamError,
)
from dashscope.applications.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
)
from dashscope.applications.agentstudio.types import (
    Memory,
    MemoryStore,
    MemoryVersion,
    Message,
    PermissionPolicy,
    SecurityAlert,
    SecurityAlertList,
    SecurityOverview,
    ServerEvent,
    SessionResource,
    StopReason,
    user_define_outcome,
    user_interrupt,
    user_message,
    user_tool_approval_response,
    user_custom_tool_result,
    user_tool_result,
)
from dashscope.applications.agentstudio.types.params import (
    AgentCreateParams,
    AgentUpdateParams,
    SessionCreateParams,
    SessionUpdateParams,
    EnvironmentCreateParams,
    EnvironmentUpdateParams,
    DeploymentCreateParams,
    DeploymentUpdateParams,
    SessionEventSendParams,
    WebhookEndpointCreateParams,
    WebhookEndpointUpdateParams,
    WebhookEventListParams,
)
from dashscope.applications.agentstudio.constants import (
    PermissionPolicyType,
    SSEEventType,
    SessionStatus,
    StopReasonType,
    WebhookDeliveryStatus,
    WebhookDisabledReason,
    WebhookEventType,
    WebhookStatus,
)

__all__ = [
    # client
    "Client",
    "AsyncClient",
    # base types
    "RequestOptions",
    "APIResponse",
    # exceptions
    "AgentStudioError",
    "APIConnectionError",
    "APIStatusError",
    "APITimeoutError",
    "AuthenticationError",
    "ConflictError",
    "InternalServerError",
    "InvalidRequestError",
    "NotFoundError",
    "OverloadedError",
    "PermissionDeniedError",
    "RateLimitError",
    "StreamError",
    "StreamClosedError",
    # pagination
    "CursorPage",
    "AsyncCursorPage",
    # unified message type
    "Memory",
    "MemoryStore",
    "MemoryVersion",
    "Message",
    "PermissionPolicy",
    "SecurityAlert",
    "SecurityAlertList",
    "SecurityOverview",
    "ServerEvent",
    "SessionResource",
    "StopReason",
    # client event helpers (re-exported for convenience)
    "user_message",
    "user_interrupt",
    "user_tool_approval_response",
    "user_custom_tool_result",
    "user_tool_result",
    "user_define_outcome",
    # typing module
    "types",
    # params (commonly used request-body types)
    "AgentCreateParams",
    "AgentUpdateParams",
    "SessionCreateParams",
    "SessionUpdateParams",
    "EnvironmentCreateParams",
    "EnvironmentUpdateParams",
    "DeploymentCreateParams",
    "DeploymentUpdateParams",
    "SessionEventSendParams",
    "WebhookEndpointCreateParams",
    "WebhookEndpointUpdateParams",
    "WebhookEventListParams",
    "WebhookStatus",
    "WebhookDisabledReason",
    "WebhookDeliveryStatus",
    "WebhookEventType",
    "SSEEventType",
    "SessionStatus",
    "StopReasonType",
    "PermissionPolicyType",
]
