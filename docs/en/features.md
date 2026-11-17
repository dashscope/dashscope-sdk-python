---
description: "Feature overview of the dashscope 2.0 SDK: typed resources, sync/async, protocols, streaming, errors, dev mock, CLI and 1.x compatibility."
---
# Features

The dashscope 2.0 SDK is a typed, resource-style client for Alibaba Cloud
Model Studio (DashScope). Highlights:

## Typed resource client

- One `Client()` holds a keep-alive connection pool shared by every resource.
- Capabilities hang off the client as typed namespaces: `client.chat`,
  `client.images`, `client.audio`, `client.files`, `client.fine_tunes`, …
- Responses are typed objects: read `resp.output_text` and
  `resp.usage.total_tokens` instead of digging through raw dicts.
- Failures raise typed exceptions by default — no manual `status_code` checks.

See [Client & Configuration](reference/client.md).

## Sync and async, one surface

- `dashscope.AsyncClient()` mirrors the sync client method-for-method.
- Shared configuration layering: arguments → `dashscope.settings` →
  environment variables → `~/.dashscope/config.toml`.
- Exception: the duplex audio WebSocket sessions are sync-only.

## One client, three protocols

- The same `client.chat.completions` call speaks DashScope `native`,
  OpenAI-compatible or Anthropic-compatible wire protocols.
- Select with `Client(protocol="openai")`, the `protocol` setting or
  `DASHSCOPE_PROTOCOL`.

See [Protocols](reference/protocols.md).

## Streaming that reads like text

- `ChatStream` exposes `.text_stream`, `.reasoning_stream` and
  `.get_final_message()` — iterate plain strings, get the assembled
  typed message at the end.
- SSE transport with automatic retry and reconnect handling.

See [Chat](reference/chat.md).

## Realtime voice over WebSocket

- Speech synthesis (TTS), recognition (ASR), omni realtime and Qwen Live
  sessions under `client.audio`.
- Voice and vocabulary customization resources.

See [Audio](reference/audio.md) and the
[Qwen Live guide](guides/qwen-live-python-sdk.md).

## Typed errors and self-diagnosis

- A `DashScopeError` hierarchy covering transport, validation, HTTP status
  and WebSocket failures, with `code`, `request_id` and response body attached.
- Opt-in local diagnosis turns a failure into structured, bilingual fix
  guidance on `error.diagnosis`.

See [Errors](reference/errors.md) and [Devkit](reference/devkit.md).

## Offline development with dev mock

- `static` / `replay` modes exercise the HTTP request path offline without
  spending cloud tokens.
- `record` sends real requests, which may incur charges, and saves the
  responses for later offline replay.
- Zero overhead when disabled; nothing is imported on the success path.

See [Devkit](reference/devkit.md).

## Managed agents and agentic RL

- `dashscope.applications.agentstudio`: a self-contained sub-SDK for managed
  agents, sessions, environments, deployments, memory, skills and vaults.
- `dashscope.applications.finetune.AgenticRL` plus decorators to author
  rollout / reward functions locally.

See [AgentStudio](reference/agentstudio.md) and
[Agentic RL](reference/agentic-rl.md).

## A real command line

- The `dashscope` CLI covers chat, audio, images, files, fine-tunes,
  deployments, RL, apps and more with consistent global options and
  machine-readable output (`-o json`).
- Optional interactive AI assistant via `pip install "dashscope[acli]"`.

See [CLI](cli.md).

## 1.x compatibility, frozen

- The complete 1.x public surface still imports from its historical paths
  (`from dashscope import Generation`), implemented by frozen facades under
  `dashscope.legacy_1x`.

See [Legacy 1.x](reference/legacy.md).
