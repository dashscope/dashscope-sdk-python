---
description: "API reference for the AgentStudio sub-SDK: managed agents, sessions, deployments, memory and vaults."
---
# AgentStudio

`dashscope.applications.agentstudio` is the sub-SDK for the AgentStudio
managed-agent API: agents, sessions, events, environments, deployments,
memory, skills, vaults, security and webhooks. All resources are reached
through its own client:

```python
from dashscope.applications.agentstudio import Client
from dashscope.applications.agentstudio import user_message

client = Client(api_key="sk-xxx")
agent = client.agents.create(name="demo", model="qwen-plus")
session = client.sessions.create(agent=agent.id)
client.sessions.events.send(session.id, [user_message("Hello!")])
with client.sessions.events.stream(session.id) as stream:
    for event in stream:
        print(event.type)
        # session_status may also report running; stop only on a terminal state
        if event.session_status in ("idle", "terminated"):
            break
```

!!! note "Session status and text consumption"
    A `session_status` event may report `running` (still working) as well
    as terminal states such as `idle` / `terminated` — do not treat the
    first `session_status` as the end of the run. When you only need the
    text reply, use `stream.text_stream`: it yields the text of agent
    messages and stops automatically once the session reaches a terminal
    state. If the agent uses tools that require approval, the session
    pauses at `idle` with `stop_reason.type == "requires_action"`; submit
    the decision with `user_tool_approval_response` and the session
    resumes.

Every resource class has an async twin (`AsyncAgents`, `AsyncSessions`, ...)
used the same way through `AsyncClient`.

## Clients

::: dashscope.applications.agentstudio.Client

::: dashscope.applications.agentstudio.AsyncClient

## Agents & sessions

::: dashscope.applications.agentstudio.resources.agents.Agents

::: dashscope.applications.agentstudio.resources.agents.AsyncAgents

::: dashscope.applications.agentstudio.resources.sessions.Sessions

::: dashscope.applications.agentstudio.resources.sessions.AsyncSessions

::: dashscope.applications.agentstudio.resources.session_events.SessionEvents

::: dashscope.applications.agentstudio.resources.session_events.AsyncSessionEvents

::: dashscope.applications.agentstudio.resources.session_threads.SessionThreads

::: dashscope.applications.agentstudio.resources.session_threads.AsyncSessionThreads

::: dashscope.applications.agentstudio.resources.session_threads.SessionThreadEvents

::: dashscope.applications.agentstudio.resources.session_threads.AsyncSessionThreadEvents

::: dashscope.applications.agentstudio.resources.session_resources.SessionResources

::: dashscope.applications.agentstudio.resources.session_resources.AsyncSessionResources

## Environments & deployments

::: dashscope.applications.agentstudio.resources.environments.Environments

::: dashscope.applications.agentstudio.resources.environments.AsyncEnvironments

::: dashscope.applications.agentstudio.resources.deployments.Deployments

::: dashscope.applications.agentstudio.resources.deployments.AsyncDeployments

::: dashscope.applications.agentstudio.resources.deployments.DeploymentRuns

::: dashscope.applications.agentstudio.resources.deployments.AsyncDeploymentRuns

## Files

::: dashscope.applications.agentstudio.resources.files.Files

::: dashscope.applications.agentstudio.resources.files.AsyncFiles

::: dashscope.applications.agentstudio.resources.files.FileContent

## Memory

::: dashscope.applications.agentstudio.resources.memory_stores.MemoryStores

::: dashscope.applications.agentstudio.resources.memory_stores.AsyncMemoryStores

::: dashscope.applications.agentstudio.resources.memory_stores.Memories

::: dashscope.applications.agentstudio.resources.memory_stores.AsyncMemories

::: dashscope.applications.agentstudio.resources.memory_stores.MemoryVersions

::: dashscope.applications.agentstudio.resources.memory_stores.AsyncMemoryVersions

## Skills

::: dashscope.applications.agentstudio.resources.skills.Skills

::: dashscope.applications.agentstudio.resources.skills.AsyncSkills

::: dashscope.applications.agentstudio.resources.skills.SkillVersions

::: dashscope.applications.agentstudio.resources.skills.AsyncSkillVersions

## Vaults & credentials

::: dashscope.applications.agentstudio.resources.vaults.Vaults

::: dashscope.applications.agentstudio.resources.vaults.AsyncVaults

::: dashscope.applications.agentstudio.resources.vaults.Credentials

::: dashscope.applications.agentstudio.resources.vaults.AsyncCredentials

## Security & webhooks

::: dashscope.applications.agentstudio.resources.security.Security

::: dashscope.applications.agentstudio.resources.security.AsyncSecurity

::: dashscope.applications.agentstudio.resources.webhook_endpoints.WebhookEndpoints

::: dashscope.applications.agentstudio.resources.webhook_endpoints.AsyncWebhookEndpoints

## Usage

`client.usage` — workspace, agent and session usage summaries and trends.

::: dashscope.applications.agentstudio.resources.usage.Usage

::: dashscope.applications.agentstudio.resources.usage.AsyncUsage

## Client event helpers

::: dashscope.applications.agentstudio.user_message

::: dashscope.applications.agentstudio.user_interrupt

::: dashscope.applications.agentstudio.user_tool_result

::: dashscope.applications.agentstudio.user_custom_tool_result

::: dashscope.applications.agentstudio.user_tool_approval_response

::: dashscope.applications.agentstudio.user_define_outcome

## Core types

::: dashscope.applications.agentstudio.Message

::: dashscope.applications.agentstudio.ServerEvent

::: dashscope.applications.agentstudio.SessionResource

::: dashscope.applications.agentstudio.StopReason

::: dashscope.applications.agentstudio.Memory

::: dashscope.applications.agentstudio.MemoryStore

::: dashscope.applications.agentstudio.MemoryVersion

::: dashscope.applications.agentstudio.PermissionPolicy

::: dashscope.applications.agentstudio.SecurityAlert

::: dashscope.applications.agentstudio.SecurityAlertList

::: dashscope.applications.agentstudio.SecurityOverview

## Request parameter types

::: dashscope.applications.agentstudio.AgentCreateParams

::: dashscope.applications.agentstudio.AgentUpdateParams

::: dashscope.applications.agentstudio.SessionCreateParams

::: dashscope.applications.agentstudio.SessionUpdateParams

::: dashscope.applications.agentstudio.SessionEventSendParams

::: dashscope.applications.agentstudio.EnvironmentCreateParams

::: dashscope.applications.agentstudio.EnvironmentUpdateParams

::: dashscope.applications.agentstudio.DeploymentCreateParams

::: dashscope.applications.agentstudio.DeploymentUpdateParams

::: dashscope.applications.agentstudio.WebhookEndpointCreateParams

::: dashscope.applications.agentstudio.WebhookEndpointUpdateParams

::: dashscope.applications.agentstudio.WebhookEventListParams

## Constants & enums

::: dashscope.applications.agentstudio.SessionStatus

::: dashscope.applications.agentstudio.SSEEventType

::: dashscope.applications.agentstudio.StopReasonType

::: dashscope.applications.agentstudio.PermissionPolicyType

::: dashscope.applications.agentstudio.WebhookStatus

::: dashscope.applications.agentstudio.WebhookEventType

::: dashscope.applications.agentstudio.WebhookDeliveryStatus

::: dashscope.applications.agentstudio.WebhookDisabledReason

## Pagination, transport & exceptions

::: dashscope.applications.agentstudio.CursorPage

::: dashscope.applications.agentstudio.AsyncCursorPage

::: dashscope.applications.agentstudio.RequestOptions

::: dashscope.applications.agentstudio.APIResponse

::: dashscope.applications.agentstudio.AgentStudioError

::: dashscope.applications.agentstudio.APIConnectionError

::: dashscope.applications.agentstudio.APITimeoutError

::: dashscope.applications.agentstudio.APIStatusError

::: dashscope.applications.agentstudio.AuthenticationError

::: dashscope.applications.agentstudio.PermissionDeniedError

::: dashscope.applications.agentstudio.NotFoundError

::: dashscope.applications.agentstudio.ConflictError

::: dashscope.applications.agentstudio.InvalidRequestError

::: dashscope.applications.agentstudio.RateLimitError

::: dashscope.applications.agentstudio.OverloadedError

::: dashscope.applications.agentstudio.InternalServerError

::: dashscope.applications.agentstudio.StreamError

::: dashscope.applications.agentstudio.StreamClosedError
