---
name: agent
description: Quick reference for Agent APIs (agentstudio-hosted Agent/Application/Assistants, plugins and MCP): parameters, inputs/outputs, and error codes
---
# Agent

For Python SDK 2.0.3, use `dashscope.Client` / `dashscope.AsyncClient` at the package root; `dashscope.v2` does not exist. Close the client with a context manager or `close()`/`aclose()`. New application calls use `client.apps.completions`; hosted-agent management uses the lazily mounted `client.agentstudio`. Assistants APIs below are exclusively for 1.x maintenance.

## Use Cases and Selection

- **Bailian applications:** `client.apps.completions.create(...)` invokes an AppBuilder/RAG/workflow/plugin application by `app_id`; application tools and flows remain configured in the console.
- **AgentStudio:** `client.agentstudio` manages hosted agents, sessions, environments, files, skills, deployments, and related resources, with typed models and SSE execution streams.
- **Assistants / Threads / Messages / Runs / Steps:** deprecated 1.x compatibility APIs for repairing existing applications only. There is no 2.0 `Responses` resource, so do not recommend migrating these APIs to Responses. Choose Apps or AgentStudio for new hosted-agent scenarios based on the required service, not a nonexistent resource.

## Bailian Applications (`client.apps`)

```text
client.apps.completions.create(
    *, app_id, prompt=None, messages=None, stream=False, session_id=None,
    biz_params=None, has_thoughts=None, ..., parameters=None, timeout=None,
)
```

Supply `prompt` or `messages`; when both are present, `prompt` becomes the latest user message. Other supported options include document/memory/image/file inputs, `rag_options`, `incremental_output`, model controls, and service-specific `parameters`. For a continued conversation, pass the previous result's `session_id`.

```python
import dashscope

with dashscope.Client() as client:  # reads DASHSCOPE_API_KEY by default
    result = client.apps.completions.create(
        app_id="YOUR_APP_ID",
        prompt="Introduce Bailian",
        has_thoughts=True,
    )
    print(result.text)
    print(result.session_id, result.finish_reason)
```

`AppCompletion` exposes flat `text`, `finish_reason`, `session_id`, `thoughts`, `doc_references`, `workflow_message`, and `usage`, together with common response metadata. It does not require legacy `.output.text` access.

With `stream=True`, the return value is `AppStream` (`AsyncAppStream` on `AsyncClient`). Iterate `text_stream` for text and close the stream when finished, especially on early exit:

```python
import dashscope

with dashscope.Client() as client:
    stream = client.apps.completions.create(
        app_id="YOUR_APP_ID",
        prompt="Write a short greeting",
        stream=True,
        incremental_output=True,
    )
    try:
        for text in stream.text_stream:
            print(text, end="")
    finally:
        stream.close()
```

## AgentStudio (`client.agentstudio`)

Mount AgentStudio from the main client so configuration and cleanup stay connected:

```python
import dashscope
from dashscope.agentstudio.types import user_message

with dashscope.Client(workspace="my-ws") as client:
    studio = client.agentstudio
    agent = studio.agents.create(name="demo", model="qwen3.8-flash")
    session = studio.sessions.create(agent=agent.id)
    studio.sessions.events.send(session.id, [user_message("Hello!")])
    with studio.sessions.events.stream(session.id) as stream:
        for text in stream.text_stream:
            print(text, end="")
# Closing the main client also closes its mounted AgentStudio client.
```

`Client(api_key=..., workspace=..., region=..., ...)` forwards API key, workspace, and region to AgentStudio. A workspace is required unless an AgentStudio base URL is configured. Standalone `dashscope.agentstudio.Client/AsyncClient` remain available, but are unnecessary for normal 2.0 main-client usage.

Mounted resources are `agents`, `deployments`, `deployment_runs`, `sessions`, `environments`, `files`, `skills`, `vaults`, `memory_stores`, `webhook_endpoints`, and `security`.

### Agents and Sessions

- `agents.create(*, name, model, description=None, system_prompt=None, tools=None, mcp_servers=None, skills=None, multiagent=None, metadata=None)` returns `Agent`. `model` can be a model ID string or `ModelConfig(id=..., temperature=..., max_tokens=..., enable_thinking=..., thinking_budget=...)`.
- `agents.retrieve(agent_id, version=None)` (alias `get`), `list(...)`, and `list_versions(agent_id)` query agent state. `archive(agent_id)` archives it.
- `agents.update(agent_id, *, version, name, model=None, ...)` is optimistic-concurrency checked: `version` must match the current version. The service also requires `name`; sending `model` replaces its model configuration, so retain unspecified settings (for example with the existing model's `copy_with(...)`) rather than accidentally dropping them.
- `sessions.create(*, agent, environment_id=None, title=None, resources=None, vault_ids=None, environment_variables=None, mcp_configs=None, metadata=None)` returns `Session`. Sessions also support `retrieve`/`get`, `update`, `list`, `archive`, and `delete`.
- `environments.create(*, name, config, description=None, scope="organization", metadata=None)` creates an execution environment. `files.upload(file, *, mime_type=None, progress=None)` accepts a path, binary stream, or `(name, fileobj)`. Skills can be created from `file_id` or `file`, with nested version APIs.

### AgentStudio SSE Execution

1. `studio.sessions.events.send(session_id, events)` sends a non-empty event list; an empty list raises local `ValueError`. Constructors in `dashscope.agentstudio.types` include `user_message(...)`, `user_interrupt()`, `user_tool_confirmation()`, `user_tool_result()`, `user_custom_tool_result()`, and `user_define_outcome()`.
2. `with studio.sessions.events.stream(session_id, event_deltas=None, timeout=None) as stream:` opens a single-pass SSE stream. Iterating it yields typed server `Message` events with fields such as `type`, `content`, `stop_reason`, and `session_status`.
3. `stream.text_stream` yields text blocks from message events and stops on terminal stream/session states such as `idle`, `terminated`, or `rescheduling`. Opt into `event_deltas` when incremental `text_deltas` are required; do not assume all delta types arrive by default.

Content blocks can represent text, images, audio, data, files, refusal, or errors. Client event types include message, interrupt, tool confirmation/results, and defined outcomes; server events additionally include reasoning, tool/MCP calls, session status, thread, model-request, and outcome-evaluation events. A typical session state flow is `idle → running → (rescheduling) → idle | terminated`.

### Plugins, Tools, and MCP

- AgentStudio: pass tool schemas through `agents.create(tools=[...])` and MCP configurations through `mcp_servers`. The runtime emits tool/MCP call events; execute client-side tools as required and send results or confirmations back through session events.
- Apps: bind plugins and workflows in the Bailian console. Callers pass business parameters with `biz_params`; `has_thoughts=True` requests process information in flat `thoughts`.

## Errors (Brief)

New code catches API exception classes from `dashscope.errors`; AgentStudio's stream-specific `StreamError` / `StreamClosedError` remain in `dashscope.agentstudio.exceptions`. Match exception classes rather than inferring the class solely from a raw wire `.code`.

| HTTP / condition | Current class | Handling |
| --- | --- | --- |
| 400 | `BadRequestError` (`InvalidRequestError` is an alias) | Check parameter names/types; empty event lists are local `ValueError` |
| 401 | `AuthenticationError` | Check `DASHSCOPE_API_KEY` or explicit `api_key` |
| 403 | `PermissionDeniedError` | Check workspace, UID, region, and resource ownership |
| 413 | `RequestTooLargeError` | Reduce payload size and check service limits |
| 404 | `NotFoundError` | Verify agent/session/resource IDs |
| 409 | `APIStatusError` (there is no dedicated conflict catalog class) | Retrieve the latest agent version before retrying update |
| 429 | `RateLimitError` | Retry with backoff and check quota |
| 503 / mapped overloaded error | `ServiceUnavailableError` (`OverloadedError` is an alias) | Retry with backoff |
| 504 | `GatewayTimeoutError` | Retry or troubleshoot with `request_id`; do not label it `InternalServerError` |
| 500 or a code mapped to internal error | `InternalServerError` | Retry where appropriate and retain `request_id` |
| Unmapped HTTP status, including an unknown 5xx | `APIStatusError` | Inspect status/code/body; do not blindly guess `InternalServerError`. An unrecognized wire code may still use the known HTTP status mapping. |
| Connection/read/write timeout | `APIConnectionError` / `APITimeoutError` | Check network and timeout before retrying |
| AgentStudio stream protocol/closed stream | `StreamError` / `StreamClosedError` | Reopen `events.stream` when the operation is retryable |

See the dedicated errors guide for the complete hierarchy and retry rules.

## 1.x Compatibility — Maintenance Only

These APIs retain their old response envelopes and local exception behavior. Do not mix them with 2.0 flat `AppCompletion` models or `client.agentstudio`.

### Legacy Application

`dashscope.Application.call(app_id, prompt=None, history=None, workspace=None, api_key=None, messages=None, **kwargs)` invokes a published app. At least one of `prompt` or `messages` is required; missing `app_id` raises `InputRequired`. Legacy kwargs include `stream`, `incremental_output`, `session_id`, `biz_params`, `has_thoughts`, document/memory/image/file inputs, `rag_options`, and model controls.

`ApplicationResponse.output` contains `text`, `finish_reason`, `session_id`, `thoughts`, `doc_references`, and `workflow_message`. Legacy failures generally do not raise: check `status_code`, then inspect `code/message/request_id`. This is the opposite of the default 2.0 exception flow.

### Legacy Assistants / Threads / Messages / Runs / Steps

- `Assistants.call/create(*, model, name=None, description=None, instructions=None, tools=None, file_ids=[], metadata=None, ...)`; `retrieve/list/update/delete`.
- `Threads.create(*, messages=None, metadata=None)`; `Messages.create(thread_id, *, content, role="user", file_ids=[], metadata=None)`.
- `Runs.create(thread_id, *, assistant_id, instructions=None, tools=None, stream=False, ...)` and `Runs.create_thread_and_run(...)`; also `retrieve`, `submit_tool_outputs`, and `cancel`. Steps belong to this same legacy maintenance surface.
- Run states: `queued → in_progress → (requires_action) → completed | failed | cancelled | expired`, with `cancelling` also possible. On `requires_action`, read `run.required_action.submit_tool_outputs`, execute the requested tools, submit outputs, and continue polling.
- Legacy tool schemas and the Run tool-output loop must remain in this compatibility implementation; do not rewrite them as AgentStudio events unless intentionally migrating the application.

## Java SDK

Java SDK v2.22.23 entry points:

- `Application` (Bailian app invocation): `ApplicationResult call(ApplicationParam param)` / `Flowable<ApplicationResult> streamCall(param)` — pass `appId` + `prompt`
- `Assistants` (OpenAI-compatible assistants): `create(AssistantParam)` / `update(String assistantId, AssistantParam)` / `retrieve(String assistantId)` / `list(GeneralListParam)`

```java
import com.alibaba.dashscope.app.*;

Application app = new Application();
ApplicationParam param = ApplicationParam.builder()
        .appId("YOUR_APP_ID").prompt("hello").build();
ApplicationResult result = app.call(param);
String text = result.getOutput().getText();
```

Samples: `ApplicationCalls.java`, `AssistantSimple.java`, `AssistantFunctionCall.java`, `AssistantCallSearchStream.java`.
