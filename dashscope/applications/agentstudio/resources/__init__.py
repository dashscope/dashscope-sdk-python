# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Resource classes for the AgentStudio SDK.

All resource classes are instantiated by the client and accessed as
attributes (e.g. ``client.agents``, ``client.sessions``).

Usage::

    from dashscope.applications.agentstudio import Client

    client = Client(api_key="sk-...")
    agent = client.agents.create(name="demo", model="qwen-plus")
    session = client.sessions.create(agent=agent.id)
    client.sessions.events.send(session.id, [user_message("hello")])
"""

from dashscope.applications.agentstudio.resources.agents import (
    Agents,
    AsyncAgents,
)
from dashscope.applications.agentstudio.resources.deployments import (
    AsyncDeploymentRuns,
    AsyncDeployments,
    DeploymentRuns,
    Deployments,
)
from dashscope.applications.agentstudio.resources.environments import (
    Environments,
    AsyncEnvironments,
)
from dashscope.applications.agentstudio.resources.files import (
    Files,
    AsyncFiles,
)
from dashscope.applications.agentstudio.resources.memory_stores import (
    AsyncMemories,
    AsyncMemoryStores,
    AsyncMemoryVersions,
    Memories,
    MemoryStores,
    MemoryVersions,
)
from dashscope.applications.agentstudio.resources.session_events import (
    AsyncSessionEvents,
    SessionEvents,
    _AioTypedEventStream,
    _TypedEventStream,
)
from dashscope.applications.agentstudio.resources.sessions import (
    Sessions,
    AsyncSessions,
)
from dashscope.applications.agentstudio.resources.skills import (
    AsyncSkillVersions,
    AsyncSkills,
    SkillVersions,
    Skills,
)
from dashscope.applications.agentstudio.resources.vaults import (
    AsyncCredentials,
    AsyncVaults,
    Credentials,
    Vaults,
)
from dashscope.applications.agentstudio.resources.webhook_endpoints import (
    AsyncWebhookEndpoints,
    WebhookEndpoints,
)

__all__ = [
    "Agents",
    "AsyncAgents",
    "Deployments",
    "AsyncDeployments",
    "DeploymentRuns",
    "AsyncDeploymentRuns",
    "Credentials",
    "AsyncCredentials",
    "Environments",
    "AsyncEnvironments",
    "Files",
    "AsyncFiles",
    "Skills",
    "AsyncSkills",
    "SkillVersions",
    "AsyncSkillVersions",
    "Sessions",
    "AsyncSessions",
    "SessionEvents",
    "AsyncSessionEvents",
    "Vaults",
    "AsyncVaults",
    "MemoryStores",
    "AsyncMemoryStores",
    "Memories",
    "AsyncMemories",
    "MemoryVersions",
    "AsyncMemoryVersions",
    "WebhookEndpoints",
    "AsyncWebhookEndpoints",
    "_TypedEventStream",
    "_AioTypedEventStream",
]
