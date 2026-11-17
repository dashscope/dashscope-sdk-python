---
description: "AgentStudio 子 SDK API 参考：托管智能体、会话、部署、记忆与密钥库。"
---
# AgentStudio（托管智能体）

`dashscope.applications.agentstudio` 是 AgentStudio 托管智能体 API 的
子 SDK：智能体、会话、事件、环境、部署、记忆、技能、密钥库、
安全与 Webhook。所有资源都通过它自己的客户端访问：

```python
from dashscope.applications.agentstudio import Client
from dashscope.applications.agentstudio import user_message

client = Client(api_key="sk-xxx")
agent = client.agents.create(name="demo", model="qwen-plus")
session = client.sessions.create(agent=agent.id)
client.sessions.events.send(session.id, [user_message("你好！")])
with client.sessions.events.stream(session.id) as stream:
    for event in stream:
        print(event.type)
        if event.type == "session_status":
            break
```

每个资源类都有异步孪生类（`AsyncAgents`、`AsyncSessions`……），
通过 `AsyncClient` 以相同方式使用。

## 客户端

::: dashscope.applications.agentstudio.Client

::: dashscope.applications.agentstudio.AsyncClient

## 智能体与会话

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

## 环境与部署

::: dashscope.applications.agentstudio.resources.environments.Environments

::: dashscope.applications.agentstudio.resources.environments.AsyncEnvironments

::: dashscope.applications.agentstudio.resources.deployments.Deployments

::: dashscope.applications.agentstudio.resources.deployments.AsyncDeployments

::: dashscope.applications.agentstudio.resources.deployments.DeploymentRuns

::: dashscope.applications.agentstudio.resources.deployments.AsyncDeploymentRuns

## 文件

::: dashscope.applications.agentstudio.resources.files.Files

::: dashscope.applications.agentstudio.resources.files.AsyncFiles

## 记忆

::: dashscope.applications.agentstudio.resources.memory_stores.MemoryStores

::: dashscope.applications.agentstudio.resources.memory_stores.AsyncMemoryStores

::: dashscope.applications.agentstudio.resources.memory_stores.Memories

::: dashscope.applications.agentstudio.resources.memory_stores.AsyncMemories

::: dashscope.applications.agentstudio.resources.memory_stores.MemoryVersions

::: dashscope.applications.agentstudio.resources.memory_stores.AsyncMemoryVersions

## 技能

::: dashscope.applications.agentstudio.resources.skills.Skills

::: dashscope.applications.agentstudio.resources.skills.AsyncSkills

::: dashscope.applications.agentstudio.resources.skills.SkillVersions

::: dashscope.applications.agentstudio.resources.skills.AsyncSkillVersions

## 密钥库与凭证

::: dashscope.applications.agentstudio.resources.vaults.Vaults

::: dashscope.applications.agentstudio.resources.vaults.AsyncVaults

::: dashscope.applications.agentstudio.resources.vaults.Credentials

::: dashscope.applications.agentstudio.resources.vaults.AsyncCredentials

## 安全与 Webhook

::: dashscope.applications.agentstudio.resources.security.Security

::: dashscope.applications.agentstudio.resources.security.AsyncSecurity

::: dashscope.applications.agentstudio.resources.webhook_endpoints.WebhookEndpoints

::: dashscope.applications.agentstudio.resources.webhook_endpoints.AsyncWebhookEndpoints

## 客户端事件辅助函数

::: dashscope.applications.agentstudio.user_message

::: dashscope.applications.agentstudio.user_interrupt

::: dashscope.applications.agentstudio.user_tool_result

::: dashscope.applications.agentstudio.user_custom_tool_result

::: dashscope.applications.agentstudio.user_tool_approval_response

::: dashscope.applications.agentstudio.user_define_outcome

## 核心类型

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

## 请求参数类型

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

## 常量与枚举

::: dashscope.applications.agentstudio.SessionStatus

::: dashscope.applications.agentstudio.SSEEventType

::: dashscope.applications.agentstudio.StopReasonType

::: dashscope.applications.agentstudio.PermissionPolicyType

::: dashscope.applications.agentstudio.WebhookStatus

::: dashscope.applications.agentstudio.WebhookEventType

::: dashscope.applications.agentstudio.WebhookDeliveryStatus

::: dashscope.applications.agentstudio.WebhookDisabledReason

## 分页、传输与异常

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