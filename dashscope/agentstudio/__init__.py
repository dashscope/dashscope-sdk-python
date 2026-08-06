# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AgentStudio sub-SDK for the dashscope Python client.

The ``agentstudio`` package wraps the AgentStudio managed-agent HTTP
API.  All resource classes are accessed through a client instance::

    from dashscope.agentstudio import Client
    from dashscope.agentstudio.types import (
        user_message, user_tool_result,
    )

Example
-------
>>> from dashscope.agentstudio import Client
>>> from dashscope.agentstudio.types import user_message
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

from .client import Client, AsyncClient
from .transport import APIResponse, RequestOptions
from . import types
from .exceptions import (
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
from .pagination import (
    AsyncCursorPage,
    CursorPage,
)
from .types import (
    Message,
    ServerEvent,
    user_define_outcome,
    user_interrupt,
    user_message,
    user_tool_confirmation,
    user_custom_tool_result,
    user_tool_result,
)
from .types.params import (
    AgentCreateParams,
    AgentUpdateParams,
    SessionCreateParams,
    SessionUpdateParams,
    EnvironmentCreateParams,
    EnvironmentUpdateParams,
    DeploymentCreateParams,
    DeploymentUpdateParams,
    SessionEventSendParams,
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
    "Message",
    "ServerEvent",
    # client event helpers (re-exported for convenience)
    "user_message",
    "user_interrupt",
    "user_tool_confirmation",
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
]
