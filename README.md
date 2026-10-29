# DashScope Python SDK

> **English** | [中文](README_zh.md)

The DashScope Python SDK is the official client for [Alibaba Cloud Model Studio (Bailian)](https://www.alibabacloud.com/help/en/model-studio/), covering text generation, multi-modal understanding, embeddings, reranking, image/video generation, speech synthesis & recognition, fine-tuning, agentic RL, applications and more.

**Version 2.0** introduces a resource-style client — `dashscope.Client()` — modeled on the ergonomics of the OpenAI SDK: one client holds a shared connection pool, every capability hangs off it as a typed resource (`client.chat`, `client.images`, `client.audio`, …), responses are typed objects instead of raw dicts, and failures raise exceptions by default. The complete 1.x public surface remains available at its historical import paths and is implemented by facades under `dashscope.legacy_1x`; AgentStudio and Agentic RL share their large business cores with 2.0 without sharing public return or error contracts.

## What's new in 2.0

| | 1.x | 2.0 |
|---|---|---|
| Entry point | Module-level static classes: `Generation.call(...)` | A resource-style client: `client.chat.completions.create(...)` |
| One-liner | — | `dashscope.chat("qwen-plus", "Tell me a joke")` |
| Response | Raw dict-like; check `status_code == HTTPStatus.OK` | Typed object; read `resp.output_text`, `resp.usage.total_tokens` |
| Errors | Inspect `resp.code` / `resp.message` by hand | Typed `DashScopeError` exceptions raised by default *(evolving)* |
| Streaming | Generator of raw frames | `ChatStream` with `.text_stream`, `.reasoning_stream`, `.get_final_message()` |
| Protocols | DashScope native + a separate compatible-mode entry | One client speaks `native`, `openai`, or `anthropic` via `protocol=` |
| Async | Separate `Aio*` classes | `dashscope.AsyncClient()` mirrors the sync surface (except the duplex audio WebSocket sessions) |
| Connections | A fresh session per call | One keep-alive pool per client, shared across resources |
| Config | Module globals (`dashscope.api_key`, `set_region`) | Layered: args → `dashscope.settings` → env → `~/.dashscope/config.toml` |

## Installation

```shell
pip install -U dashscope
```

The base install covers SDK API calls, the `dashscope` CLI command, Agentic RL fine-tuning and the local tokenizer. Python 3.9+ is required. One optional feature group remains:

| Extra | Provides | Install |
|-------|----------|---------|
| `acli` | Interactive AI assistant (DashScope SDK Expert) | `pip install "dashscope[acli]"` |

To install from a source checkout:

```shell
pip install -e .
```

## Quick start

Set your API key ([how to get one](https://help.aliyun.com/en/model-studio/get-api-key)):

```shell
export DASHSCOPE_API_KEY="sk-xxx"
```

**One line** — the fastest way to a completion:

```python
import dashscope

completion = dashscope.chat("qwen-plus", "Tell me a joke")
print(completion.output_text)
```

**Client** — the recommended form for anything real. One client, reused across calls:

```python
import dashscope

client = dashscope.Client()  # reads DASHSCOPE_API_KEY

resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who are you?"},
    ],
)
print(resp.output_text)
print(resp.usage.total_tokens, "tokens |", resp.request_id)
```

**Streaming** — `stream=True` returns a `ChatStream`:

```python
stream = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Write a haiku about the sea"}],
    stream=True,
)
for text in stream.text_stream:
    print(text, end="", flush=True)

# Or collapse the whole stream into one merged ChatCompletion (with usage):
# final = stream.get_final_message()
```

**Async** — `AsyncClient` mirrors the sync surface:

```python
import asyncio
import dashscope

async def main():
    client = dashscope.AsyncClient()
    resp = await client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(resp.output_text)
    await client.aclose()

asyncio.run(main())
```

## Core concepts

### The client

`dashscope.Client()` (and `dashscope.AsyncClient()`) is the single entry point. It owns one keep-alive connection pool shared by every resource call, and exposes capabilities as attributes:

```python
client = dashscope.Client(
    api_key="sk-xxx",       # default: DASHSCOPE_API_KEY
    workspace="ws-xxx123",  # default: DASHSCOPE_WORKSPACE
    region="cn-beijing",    # default: DASHSCOPE_REGION
    protocol="native",      # native | openai | anthropic
    timeout=300.0,
    max_retries=2,
)
```

Use it as a context manager to close the pool automatically, or derive a variant with `with_options()` (unmentioned options are inherited; the connection pool is shared when connection-level options are unchanged):

```python
with dashscope.Client() as client:
    client.chat.completions.create(model="qwen-plus", messages=["hi"])

low_latency = client.with_options(timeout=10.0, max_retries=0)
```

Any call also accepts per-request `api_key=`, `workspace=`, and `timeout=` overrides.

Need proxies, custom connection limits or event hooks? Pass your own `http_client=httpx.Client(...)` (`httpx.AsyncClient` for `AsyncClient`). The SDK still applies authentication, retries and absolute endpoint URLs on top of it, and never closes a client it does not own.

### Protocols

One client can speak three wire protocols, selected with `protocol=` (or `DASHSCOPE_PROTOCOL` / `settings.protocol`). The request/response objects stay the same canonical, typed shape regardless of protocol — only the wire format changes:

| `protocol` | Talks to | Use when |
|------------|----------|----------|
| `native` (default) | DashScope native API (`/api/v1`) | You want the full DashScope feature set |
| `openai` | OpenAI-compatible mode (`/compatible-mode/v1`) | You're porting OpenAI code — no `openai` package needed |
| `anthropic` | Anthropic-compatible mode (`/apps/anthropic`) | You're porting Anthropic Messages code |

```python
client = dashscope.Client(protocol="openai")
resp = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Typed responses

Every response is a typed object. For chat, read `output_text` for the first choice, or dig into `choices`, `usage`, `search_info`, `request_id`, and `model`:

```python
resp = client.chat.completions.create(model="qwen-plus", messages=["hi"])

resp.output_text                       # choices[0].message.content
resp.choices[0].message.content        # same, explicit
resp.choices[0].message.reasoning_content  # thinking-model reasoning, if any
resp.choices[0].message.tool_calls     # tool calls, if any
resp.usage.input_tokens                # also .output_tokens / .total_tokens
resp.request_id
```

All responses share a common envelope: `request_id`, `status_code`, `code`, `message`, `is_error`, and raw `headers`.

### Streaming

`stream=True` returns a `ChatStream` (`AsyncChatStream` for `AsyncClient`) with typed helpers:

```python
stream = client.chat.completions.create(model="qwen-plus", messages=["hi"], stream=True)

for text in stream.text_stream:            # content deltas
    ...
# stream.reasoning_stream                  # reasoning deltas (thinking models)
# stream.tool_call_stream                  # tool-call deltas
# stream.get_final_message()               # merged ChatCompletion, incl. usage
```

By default each frame is incremental. Pass `incremental_output=False` to receive cumulative snapshots instead (the SDK merges deltas for you).

Thinking models surface reasoning on `.reasoning_stream` and, in the merged message, as `message.reasoning_content`. Under `protocol="anthropic"` the SDK also keeps `message.reasoning_signature` plus the raw `thinking` / `redacted_thinking` blocks in `message.reasoning_blocks`, so echoing the assistant message back on the next turn preserves extended thinking.

> **Note:** the SDK does **not** auto-retry streaming requests. Wrap stream consumption in your own retry/backoff for production — see [`examples/04_real_world_scenarios/production_ready_rag_harness.py`](examples/04_real_world_scenarios/production_ready_rag_harness.py).

### Errors

> **Still evolving.** The 2.0 typed error surface is under active development — new exception classes may be added and the API-error-code → exception mapping may be refined. Catch the base `errors.DashScopeError` if you need to stay forward-compatible.

Failures raise a typed `DashScopeError` by default (both non-2xx HTTP statuses and HTTP-200-with-a-business-error land here):

```python
from dashscope import errors

try:
    client.chat.completions.create(model="qwen-plus", messages=["hi"])
except errors.RateLimitError as exc:
    print("throttled:", exc.code, exc.status_code, exc.request_id)
except errors.ContentFilterError as exc:
    print("blocked by moderation:", exc.message)
except errors.DashScopeError as exc:       # base class for all SDK errors
    print("request failed:", exc.code, exc.message)
```

Everything derives from `DashScopeError`, in two families:

- **Server errors** — under `APIStatusError`: `BadRequestError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `RequestTooLargeError`, `RateLimitError`, `InternalServerError`, `ServiceUnavailableError`, `GatewayTimeoutError`, and `ContentFilterError` (a `BadRequestError` subclass).
- **Client-side errors** — under `SdkError`: `APIConnectionError` / `APITimeoutError` for network failures, plus the WebSocket family `WsEventError`, `WsStateError`, `WsQueueCapacityExceededError` and `WsBinaryOwnerConflictError`. Each carries a stable `sdk.*` code.

A 200 response whose body does not parse into the expected envelope is reported as `InternalServerError` rather than leaking a decoder exception.

Prefer to inspect errors inline instead of catching? Build the client with `raise_on_error=False` — failures come back as a normal response object with `is_error=True`, `code`, `message`, and `status_code`:

```python
client = dashscope.Client(raise_on_error=False)
resp = client.chat.completions.create(model="qwen-plus", messages=["hi"])
if resp.is_error:
    print(resp.code, resp.message)
```

## Authentication

The SDK authenticates with an API key. Get one from the [Model Studio console](https://help.aliyun.com/en/model-studio/get-api-key). Provide it in any of these ways (highest priority first):

1. **Per client** — `dashscope.Client(api_key="sk-xxx")`
2. **Global settings** — `dashscope.settings.api_key = "sk-xxx"` (applies to clients created afterwards)
3. **Environment variable** — `export DASHSCOPE_API_KEY="sk-xxx"`
4. **Config file** — `api_key` in `~/.dashscope/config.toml`
5. **1.x key file** — the file written by `dashscope.save_api_key(...)` (`~/.dashscope/api_key`, or `DASHSCOPE_API_KEY_FILE_PATH`), honored as the lowest tier so existing 1.x setups keep working

Any call also accepts a per-request `api_key=` override.

> **API keys are region-specific.** Each region issues its own keys; a key from another region fails with `401`. Switch `api_key` together with `region`.

## Configuration

Configuration resolves in layers — **explicit arguments → `dashscope.settings` → environment variables → `~/.dashscope/config.toml`**. Set it once at startup, per client, or not at all.

**Global settings facade:**

```python
import dashscope

dashscope.settings.api_key = "sk-xxx"
dashscope.settings.workspace = "ws-xxx123"
dashscope.settings.region = "ap-southeast-1"
dashscope.settings.protocol = "openai"
dashscope.settings.timeout = 60.0
dashscope.settings.max_retries = 3
dashscope.settings.log_level = "INFO"
```

**Environment variables:**

| Variable | Sets | Default |
|----------|------|---------|
| `DASHSCOPE_API_KEY` | API key | — |
| `DASHSCOPE_WORKSPACE` | Workspace ID (endpoint subdomain) | — |
| `DASHSCOPE_REGION` | Region | `cn-beijing` |
| `DASHSCOPE_PROTOCOL` | `native` / `openai` / `anthropic` | `native` |
| `DASHSCOPE_BASE_URL` | Override the endpoint entirely (escape hatch) | — |
| `DASHSCOPE_TIMEOUT` | Request timeout (seconds) | `300` |
| `DASHSCOPE_MAX_RETRIES` | Automatic retries | `2` |
| `DASHSCOPE_LOG` | Log level | `WARNING` |
| `DASHSCOPE_CONFIG` | Path to `config.toml` | `~/.dashscope/config.toml` |

**Config file** (`~/.dashscope/config.toml`):

```toml
api_key = "sk-xxx"
workspace = "ws-xxx123"
region = "ap-southeast-1"
protocol = "native"
timeout = 300.0
max_retries = 2
```

### Regions and endpoints

By default requests go to the China (Beijing) gateway. When a `workspace` is set, endpoints are built from the workspace-scoped MaaS gateway `https://{workspace}.{region}.maas.aliyuncs.com`, with each protocol served under its own path. Point at another region with `region=` (or `DASHSCOPE_REGION` / `settings.region`):

```python
client = dashscope.Client(region="ap-southeast-1", workspace="ws-xxx123")
# -> https://ws-xxx123.ap-southeast-1.maas.aliyuncs.com/api/v1
```

| Region | Location |
|--------|----------|
| `cn-beijing` | China (Beijing) — default |
| `cn-hongkong` | China (Hong Kong) |
| `ap-southeast-1` | Singapore |
| `ap-northeast-1` | Japan (Tokyo) |
| `eu-central-1` | Germany (Frankfurt) |
| `us-east-1` | US (Virginia) |

Notes:

- **No workspace configured?** The SDK falls back to the legacy shared gateway (`https://dashscope.aliyuncs.com` for Beijing) and emits a `UserWarning`. It works, but the workspace-scoped gateway is recommended for performance and stability.
- WebSocket-based realtime APIs (realtime speech, multimodal dialog) are served in `cn-beijing` and `ap-southeast-1`.
- Batch inference, fine-tuning and application development are currently available in `cn-beijing` and `ap-southeast-1`.
- `base_url=` (or `DASHSCOPE_BASE_URL`) overrides everything — the escape hatch for custom/private gateways; it does not require a workspace.
- The 1.x `dashscope.set_region(region, workspace_id)` global still works for the 1.x module-level API.

## Capabilities

Every capability is a resource on the client. Sync and async clients expose the same surface (async methods are `await`ed) — with one exception: the duplex WebSocket audio sessions are sync-only by design, as noted below.

| Capability | 2.0 resource | Example |
|------------|--------------|---------|
| Text chat / generation | `client.chat.completions.create(...)` | [`text_generation/chat.py`](examples/01_atomic_capabilities/text_generation/chat.py) |
| Multi-modal chat (vision, video) | `client.multimodal.chat.create(...)` | [`multimodal_vision/multimodal_chat.py`](examples/01_atomic_capabilities/multimodal_vision/multimodal_chat.py) |
| Text embedding | `client.embeddings.create(...)` (+ `.jobs` for batch) | [`embeddings/embeddings.py`](examples/01_atomic_capabilities/embeddings/embeddings.py) |
| Multi-modal embedding | `client.multimodal.embeddings.create(...)` | [`embeddings/`](examples/01_atomic_capabilities/embeddings/) |
| Rerank | `client.rerank.create(...)` | [`rerank/rerank.py`](examples/01_atomic_capabilities/rerank/rerank.py) |
| Decision (System One) | `client.decision.create(...)` | [`decision/decision.py`](examples/01_atomic_capabilities/decision/decision.py) |
| Token counting | `client.tokenization.count(...)` | [`text_generation/tokenization.py`](examples/01_atomic_capabilities/text_generation/tokenization.py) |
| Image generation | `client.images.generate(...)` (+ `.jobs`) | [`image_video_synthesis/images.py`](examples/01_atomic_capabilities/image_video_synthesis/images.py) |
| Video generation | `client.videos.generate(...)` (+ `.jobs`) | [`image_video_synthesis/videos.py`](examples/01_atomic_capabilities/image_video_synthesis/videos.py) |
| Speech recognition (ASR) | Batch `client.audio.asr.transcribe(...)` / `.jobs`; realtime `client.audio.asr.recognize(...)` | [`audio/asr_file.py`](examples/01_atomic_capabilities/audio/asr_file.py) · [`audio/asr_websocket.py`](examples/01_atomic_capabilities/audio/asr_websocket.py) |
| Speech synthesis (TTS) | HTTP `client.audio.tts.http.synthesize(...)` / `.stream(...)`; WebSocket `client.audio.tts.connect(...)` | [`audio/tts_http.py`](examples/01_atomic_capabilities/audio/tts_http.py) · [`audio/tts_websocket.py`](examples/01_atomic_capabilities/audio/tts_websocket.py) |
| Realtime omni dialog | `client.audio.realtime.connect(...)` | [`audio/realtime_omni.py`](examples/01_atomic_capabilities/audio/realtime_omni.py) |
| Voice & vocabulary management | `client.audio.voices` · `.voice_design` · `.vocabularies` · `.phrases` | [`audio/voice_clone.py`](examples/01_atomic_capabilities/audio/voice_clone.py) |
| Meeting & audio understanding (TingWu) | `client.tingwu.call(...)` / `.realtime(...)` | [`audio/tingwu.py`](examples/01_atomic_capabilities/audio/tingwu.py) |
| Files | `client.files.upload/list/get/delete(...)` | [`files/files.py`](examples/01_atomic_capabilities/files/files.py) |
| Models | `client.models.list/get(...)` | [`models/models.py`](examples/01_atomic_capabilities/models/models.py) |
| Fine-tuning | `client.fine_tunes.create/get/list/wait/cancel/delete(...)` | [`fine_tuning/fine_tunes.py`](examples/01_atomic_capabilities/fine_tuning/fine_tunes.py) |
| Model deployments | `client.deployments.create/scale/list/delete(...)` | [`fine_tuning/`](examples/01_atomic_capabilities/fine_tuning/) |
| Agentic RL | `client.rl` (`.jobs`, `register_functions`, `upload_data`) | [`fine_tuning/reinforcement_learning.py`](examples/01_atomic_capabilities/fine_tuning/reinforcement_learning.py) |
| Model Studio applications | `client.apps.completions.create(...)` | [`applications/apps.py`](examples/01_atomic_capabilities/applications/apps.py) |
| AgentStudio (agents, sessions, memory) | `client.agentstudio` | [`03_agent_recipes/agentstudio.py`](examples/03_agent_recipes/agentstudio.py) |

A few common shapes:

```python
# Embeddings
resp = client.embeddings.create(model="text-embedding-v4", input=["hello", "world"], dimension=1024)
resp.embeddings[0].embedding

# Multi-modal chat (image understanding)
resp = client.multimodal.chat.create(
    model="qwen3-vl-plus",
    messages=[{"role": "user", "content": [{"image": "https://.../cat.png"}, {"text": "What is this?"}]}],
)
resp.output_text

# Image generation (submit + wait); use client.images.jobs for manual control
job = client.images.generate(model="wanx2.1-t2i-turbo", prompt="a cozy flower shop", size="1024*1024")
if job.succeeded:
    print(job.image_urls)

# HTTP speech synthesis: one POST, audio comes back as a temporary URL
synth = client.audio.tts.http.synthesize(
    "Hello there", model="cosyvoice-v3-flash", voice="longanhuan", format="pcm",
)
print(synth.audio_url)

# Streaming synthesis yields audio chunks as they are rendered
for chunk in client.audio.tts.http.stream(
    "Hello there", model="cosyvoice-v3-flash", voice="longanhuan", format="pcm",
):
    if chunk.audio:
        ...  # chunk.is_final marks the closing part

# A Model Studio application
resp = client.apps.completions.create(app_id="YOUR_APP_ID", prompt="How do I make stew?")
resp.text
```

### Async audio

`AsyncClient` covers the HTTP-shaped audio resources — `await client.audio.asr.transcribe(...)` for batch transcription, `client.audio.tts.http.synthesize/.stream` for synthesis, and the management resources (`voices`, `voice_design`, `vocabularies`, `phrases`). The duplex WebSocket sessions (`asr.recognize`, `tts.connect`, `realtime.connect`) are sync-only by design; run them in a thread if you need them inside an async app.

### Realtime over one WebSocket

For latency-sensitive realtime audio, open a shared duplex socket and run multiple sessions on it instead of handshaking per task:

```python
connection = client.ws_connection()
session = client.audio.asr.recognize(model="fun-asr-realtime", connection=connection)
# ... drive the session; the caller owns the connection ...
connection.close()
```

Each task on the connection reads through a bounded queue (`task_queue_size=`, default 1024), so a slow consumer fails its own task with `errors.WsQueueCapacityExceededError` instead of holding audio in memory indefinitely or tearing down the shared socket.

## Supported models

| Category | Example models | 2.0 resource |
|----------|----------------|--------------|
| Text generation | qwen-max, qwen-plus, qwen-flash | `client.chat` |
| Multi-modal understanding | qwen3-vl-plus, qwen-omni | `client.multimodal.chat` |
| Text embedding | text-embedding-v4, text-embedding-v3 | `client.embeddings` |
| Multi-modal embedding | tongyi-embedding-vision-plus | `client.multimodal.embeddings` |
| Rerank | gte-rerank-v2, qwen3-rerank | `client.rerank` |
| Image generation | wanx2.1-t2i-turbo, wan-image | `client.images` |
| Video generation | wanx2.1-t2v, wan-i2v | `client.videos` |
| Speech synthesis (TTS) | cosyvoice-v1, qwen-tts | `client.audio.tts` |
| Speech recognition (ASR) | fun-asr-realtime, paraformer-v1 | `client.audio.asr` |
| Omni (realtime) | qwen-omni-realtime | `client.audio.realtime` |

For the latest model list, visit the [Model Studio model plaza](https://bailian.console.aliyun.com/).

## AI Assistant: DashScope SDK Expert

The SDK ships with an interactive AI assistant, **DashScope SDK Expert**, built on the bundled Agentic CLI (`dashscope/acli`). It answers SDK/API questions, generates runnable examples, shows CLI usage, and diagnoses errors right in your terminal.

- Run `dashscope` with no arguments to start the assistant. On first run it offers to install the knowledge pack (per-domain quick-reference skills: text, multimodal, speech, retrieval, fine-tuning, agent, cli), so guidance comes from the SDK's public interfaces — parameters, outputs, error codes — without reading the source.
- Ask it instead of reading docs — e.g. `dashscope expert "how do I stream chat output"` (asks once and exits). Type `/help` inside the assistant for available commands; `dashscope expert --help` shows all forms.
- Install with the `acli` extra: `pip install "dashscope[acli]"`.
- Full walkthrough: [DashScope SDK Expert guide](https://help.aliyun.com/en/model-studio/dashscope-sdk-expert).

## Shell completion

Run the command for your shell once, then restart the shell (or re-source your config):

| Shell | Install command |
|-------|-----------------|
| **bash** | `dashscope --install-completion bash` |
| **zsh** | `dashscope --install-completion zsh` |
| **fish** | `dashscope --install-completion fish` |

Preview the script without installing:

```shell
dashscope --show-completion bash
```

## Logging

Set the level via settings or the environment:

```python
dashscope.settings.log_level = "INFO"
```

```shell
export DASHSCOPE_LOG="INFO"
```

## Examples

The [`examples/`](examples/) directory is a production best-practices collection built on the 2.0 client, organized as an advancement path:

```
examples/
├── 01_atomic_capabilities/   Minimal runnable quickref per capability
├── 02_advanced_engineering/  Mock debugging, semantic cache, circuit breaker + multi-region DR
├── 03_agent_recipes/         Tool-call loop, RAG, MCP server, AgentStudio
├── 04_real_world_scenarios/  End-to-end production harnesses (RAG, realtime meetings, customer service)
└── legacy_1x/                1.x module-level API examples, for reference
```

See [`examples/README.md`](examples/README.md). Several flagship cases run fully offline (mock transports) with no API key.

## Migrating from 1.x

The 1.x module-level API is unchanged and still supported — you can adopt 2.0 incrementally. When you're ready, the mapping is direct:

| 1.x | 2.0 |
|-----|-----|
| `Generation.call(model=..., messages=..., result_format="message")` | `client.chat.completions.create(model=..., messages=...)` |
| `resp.status_code == HTTPStatus.OK` then `resp.output.choices[0].message.content` | `resp.output_text` (errors raise by default) |
| `MultiModalConversation.call(...)` | `client.multimodal.chat.create(...)` |
| `TextEmbedding.call(...)` | `client.embeddings.create(...)` |
| `TextReRank.call(...)` | `client.rerank.create(...)` |
| `ImageSynthesis.call(...)` / `.async_call(...)` | `client.images.generate(...)` / `client.images.jobs` |
| `dashscope.api_key = "sk-xxx"` | `dashscope.Client(api_key=...)` or `dashscope.settings.api_key = ...` |
| `dashscope.set_region(region, workspace_id)` | `dashscope.Client(region=..., workspace=...)` |
| OpenAI-compatible via `dashscope.aigc.chat_completion.Completions` | `dashscope.Client(protocol="openai")` |

### Package layout and compatibility

The 2.0 implementation sits directly under `dashscope`; the former `dashscope.v2` namespace is gone. AgentStudio and FineTune/Agentic RL each have a single business core under `dashscope.applications`, with two thin public facades on top:

- the historical `dashscope.agentstudio` and `dashscope.finetune.agentic_rl` imports keep the 1.x clients, methods, return models and exception classes;
- `client.agentstudio` and `client.rl` expose the 2.0 resource APIs, the httpx transport and the `dashscope.errors` family.

Historical deep modules such as `dashscope.aigc`, `dashscope.client.base_api`, `dashscope.files` and `dashscope.tokenizers` remain importable through lazy aliases to `dashscope.legacy_1x`. The local `get_tokenizer` utility therefore keeps its 1.x exception contract, while 2.0 remote token counting is the independent `client.tokenization` resource. The application cores are internal implementation paths — 2.0 callers should use the client resource tree.

Concretely: the historical Agentic RL class still returns `FineTune` from `submit_job` and retains `cancel/list/get/delete/logs`, whereas the 2.0 job lifecycle is `client.rl.jobs` with errors from the SDK catalog. Likewise, standalone `dashscope.agentstudio.Client` keeps the old `AgentStudioError` hierarchy and its `raw` response attribute, while `client.agentstudio` raises `dashscope.errors` and exposes failed response data through `body`.

> **Breaking (2.0 AgentStudio errors):** on the 2.0 surface — `client.agentstudio`, whose exceptions come from `dashscope.applications.agentstudio` — `ConflictError` is an alias of `APIStatusError` rather than a 409-specific subclass, so `except ConflictError` catches *every* status error (400/401/404/429/5xx), not just conflicts. Match on `err.status_code == 409` if you need conflict-only handling. The historical `dashscope.agentstudio.ConflictError` is untouched and still 409-specific.

> **Breaking (`submit_job` / realtime WebSocket):** `submit_job` forwards extra keyword arguments straight to the transport, so the 1.x `request_timeout=` is renamed `timeout=`, and `workspace=` / `headers=` / `stream=` are no longer accepted (passing any of them raises `TypeError`). The realtime WebSocket errors (`WsStateError`, `WsQueueCapacityExceededError`, `WsBinaryOwnerConflictError`) now derive from `DashScopeError` instead of `RuntimeError`, so a bare `except RuntimeError` no longer catches them — catch `errors.DashScopeError` (or the specific class) instead.

`dashscope.save_api_key` now comes from `dashscope._config`. It accepts an optional key-file path, expands `~`, creates missing parent directories, and writes UTF-8. Logs from the current client use the independent `dashscope._logging` logger hierarchy and do not reconfigure the shared `dashscope` root logger.

### Development tests

```bash
python -m pytest tests --ignore=tests/legacy_1x -q  # current
python -m pytest tests/legacy_1x -q                 # archived compatibility
python -m pytest tests -q                           # everything
```

For current coverage, run `coverage run -m pytest tests
--ignore=tests/legacy_1x`, then `coverage report -m`.

## License

This project is licensed under the Apache License (Version 2.0).
