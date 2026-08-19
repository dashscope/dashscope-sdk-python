---
name: agent
description: Quick reference for Agent APIs (agentstudio-hosted Agent/Application/Assistants, plugins and MCP): parameters, inputs/outputs, and error codes
---
# Agent

## Use Cases and Selection
- `dashscope.Application`: calls applications published in the Bailian console (AppBuilder/RAG/workflow/plugin apps) directly by `app_id`; no need to manage models or tools — the simplest option.
- `dashscope.agentstudio`: full lifecycle management of AgentStudio-hosted Agents (agents/sessions/environments/files/skills/deployments), with SSE event streaming, client-side tools and MCP; suitable when you need a hosted runtime and multi-turn sessions.
- `dashscope.Assistants/Threads/Runs`: OpenAI-like Assistants API. **Deprecated** (DeprecationWarning in source; migrate to the Responses API); use only for maintaining legacy code.

## SDK APIs
### agentstudio (`from dashscope.agentstudio import Client, AsyncClient`)
- `Client(api_key=, workspace=, region=None, base_url=None, uid=None, timeout=None, max_retries=2)`; if base_url is not given, `workspace` (or env var `DASHSCOPE_WORKSPACE`) is required; base_url can also be set via `DASHSCOPE_AGENTSTUDIO_URL`/`AGENTSTUDIO_URL`; default region `cn-beijing`, timeout 600s.
- Resource entry points: `client.agents / .sessions / .environments / .files / .skills / .vaults / .deployments / .deployment_runs`.
- `agents.create(*, name, model, description=None, system_prompt=None, tools=None, mcp_servers=None, skills=None, metadata=None)` → `Agent`; `retrieve(agent_id, *, version=None)` (alias `get`); `update(agent_id, *, version, ...)` (version must equal the server's current version, otherwise HTTP 409); `archive(agent_id)`; `list(*, limit, page, include_archived)` → `CursorPage`; `list_versions(agent_id)`.
- `sessions.create(*, agent, environment_id=None, title=None, resources=None, metadata=None)` → `Session`; also `retrieve/update(title, metadata)/list(*, limit, page, agent_id, statuses, created_at_gt/gte/lt/lte)/archive/delete`.
- `environments.create(*, name, config, description=None, scope="organization", metadata=None)`; `files.upload(file, *, mime_type=None, progress=None)` (file accepts a path/binary stream/(name, fileobj)); `skills.create(*, file_id=None, file=None, mime_type=None)`, `skills.versions.create(skill_id, ...)`.

### agentstudio SSE Streaming Execution (Core)
1. `client.sessions.events.send(session_id, events)`: sends an event list (at least 1 event, otherwise `ValueError`); request body is `{"input": [...]}`. Use the constructors in `types`: `user_message(text_or_blocks, *, session_thread_id=None, metadata=None)`, `user_interrupt()`, `user_tool_confirmation()`, `user_tool_result()`, `user_custom_tool_result()`, `user_define_outcome()`.
2. `with client.sessions.events.stream(session_id, *, timeout=None) as stream:` opens the SSE stream (GET `/sessions/{id}/events/stream`); iterating yields `Message` (`ServerEvent`) objects: `event.type`, `event.content` (block list), `event.stop_reason`, `event.session_status`.
3. `stream.text_stream`: yields only the text of blocks with `type=="text"` from `message` events, and stops automatically when `session_status` becomes `idle/terminated/rescheduling` — the most convenient option.
- Event types (`SSEEventType`): clients may send `message/interrupt/tool_confirmation/function_call_output/tool_call_output/define_outcome`; the server additionally sends `reasoning/tool_call/tool_call_output/mcp_call/mcp_call_output/session_status/error/thread_*/model_request_*/outcome_evaluation`, etc. Session state machine: `idle → running → (rescheduling) → idle | terminated`.

### Application (Bailian App)
`Application.call(app_id, prompt=None, history=None, workspace=None, api_key=None, messages=None, **kwargs)`; `app_id` is required (raises `InputRequired` if missing); at least one of `prompt` or `messages` is required. kwargs: `stream`, `incremental_output`, `session_id` (for multi-turn continuation, pass the session_id from the previous response), `biz_params` (flow/plugin business parameters), `has_thoughts`, `doc_tag_codes`, `doc_reference_type` (`simple`/`indexed`), `memory_id`, `image_list`, `file_list`, `rag_options`, `temperature/top_p/top_k/seed`.

### Assistants / Threads / Runs (Deprecated)
- `Assistants.call/create(*, model, name=None, description=None, instructions=None, tools=None, file_ids=[], metadata=None, workspace=None, api_key=None, top_p/top_k/temperature/max_tokens=None)`; `retrieve/list/update/delete`.
- `Threads.create(*, messages=None, metadata=None)`; `Messages.create(thread_id, *, content, role="user", file_ids=[], metadata=None)`.
- `Runs.create(thread_id, *, assistant_id, instructions=None, tools=None, stream=False, ...)`; `Runs.create_thread_and_run(*, assistant_id, thread=None, stream=False, ...)`; `Runs.retrieve(run_id, *, thread_id)`; `Runs.submit_tool_outputs(run_id, *, thread_id, tool_outputs, stream=False)`; `Runs.cancel(...)`.

## Inputs/Outputs
- **Application**: returns `ApplicationResponse` (`status_code/request_id/code/message`); `output` is an `ApplicationOutput`: `text` (answer text), `finish_reason`, `session_id` (returned for multi-turn), `thoughts` (plugin/RAG process when has_thoughts=True), `doc_references`, `workflow_message`; `usage.models[]` contains `model_id/input_tokens/output_tokens`. With `stream=True` a generator is returned; with `stream=True` and `incremental_output=False` the SDK automatically merges deltas into full output. **Failures do not raise exceptions** — you must check `response.status_code == HTTPStatus.OK`, otherwise read `code/message`.
- **agentstudio event stream**: each event is a `Message` (fields `id/type/role/content/metadata/status/session_thread_id/...`); `content` is a block list with block.type ∈ `text/image/audio/data/file/refusal/error`. The block.data of a `session_status` event contains `session_status` and `stop_reason` (e.g. `{"type": "end_turn"}`).
- **Assistants Run state machine**: `queued → in_progress → (requires_action) → completed | failed | cancelled | expired` (plus `cancelling`); on `requires_action`, take tool calls from `run.required_action.submit_tool_outputs`, execute them, send results back via `Runs.submit_tool_outputs`, and poll `retrieve` until `completed`.

## Plugin and MCP Integration
- agentstudio: pass a list of function/plugin schemas to `tools` of `agents.create` and a list of MCP server configs to `mcp_servers`; the runtime automatically emits `tool_call`/`mcp_call` events; after executing local tools, send results back with `user_tool_result(...)`/`user_tool_confirmation(...)` events.
- Application: plugins are bound to the app in the Bailian console; callers only pass plugin/flow parameters via `biz_params`; set `has_thoughts=True` to see the invocation process.
- Assistants: `tools=[{"type": ...}]`, combined with the Run `requires_action` loop.

## Minimal Examples
```python
from http import HTTPStatus
from dashscope import Application
resp = Application.call(app_id="YOUR_APP_ID", prompt="Introduce Bailian", has_thoughts=True)
if resp.status_code == HTTPStatus.OK:
    print(resp.output.text, resp.output.session_id)
else:
    print(resp.request_id, resp.code, resp.message)
```
```python
from dashscope.agentstudio import Client
from dashscope.agentstudio.types import user_message
client = Client(api_key="sk-xxx", workspace="my-ws")
agent = client.agents.create(name="demo", model="qwen-max")
session = client.sessions.create(agent=agent.id)
client.sessions.events.send(session.id, [user_message("Hello!")])
with client.sessions.events.stream(session.id) as stream:
    for text in stream.text_stream:   # auto-stops on idle/terminated
        print(text, end="")
```

## Common Error Codes
| Error code/exception | HTTP status | Meaning | Handling |
| --- | --- | --- | --- |
| `InputRequired` / `ModelRequired` / `InvalidInput` | raised locally | missing required params such as app_id/prompt/model | add params per the exception message; Application failures do not raise — check status_code |
| `invalid_request_error` (InvalidRequestError) | 400 | invalid agentstudio request parameters | check parameter names/types, e.g. empty events raises a local ValueError |
| `authentication_error` (AuthenticationError) | 401 | api_key missing or invalid | check api_key or DASHSCOPE_API_KEY |
| `permission_denied_error` (PermissionDeniedError) | 403 | no access to the resource | check workspace/uid and resource ownership |
| `not_found_error` (NotFoundError) | 404 | agent/session and other resources do not exist | verify the id and region/workspace are correct |
| `conflict_error` (ConflictError) | 409 | version in agents.update does not match the server | retrieve the latest version first, then update |
| `rate_limit_error` (RateLimitError) | 429 | rate limited | retry with backoff (client default max_retries=2) |
| `api_error` (InternalServerError) / `overloaded_error` (OverloadedError) | 500/502/504 / 503 | server error/overloaded | troubleshoot with `request_id` or retry |
| `APITimeoutError` / `APIConnectionError` | no response | connection/read/write timeout | increase timeout, check network, then retry |
| `StreamError` / `StreamClosedError` | mid-stream | SSE protocol error/stream closed | call `events.stream` again to reopen the stream |

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
