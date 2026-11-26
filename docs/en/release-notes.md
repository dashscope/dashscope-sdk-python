---
description: "Release notes and changelog pointers for the dashscope Python SDK, with 2.0 highlights."
---
# Release Notes

The authoritative, per-release changelog is published on
[GitHub Releases](https://github.com/dashscope/dashscope-sdk-python/releases).
The recent-releases table below is rendered from GitHub Releases at docs
build time, so it refreshes automatically when a release is published;
each tag links to its full release notes.
Current SDK version: **{{ sdk_version }}**.

{{ release_notes }}

## 2.0 highlights

Version 2.0 introduced the resource-style client modeled on the ergonomics
of the OpenAI SDK:

- **Entry point** — `client.chat.completions.create(...)` instead of
  module-level static classes; one-liner `dashscope.chat("qwen-plus", "...")`.
- **Responses** — typed objects (`resp.output_text`, `resp.usage.total_tokens`)
  instead of raw dict-like results.
- **Errors** — typed `DashScopeError` exceptions raised by default.
- **Streaming** — `ChatStream` with `.text_stream`, `.reasoning_stream` and
  `.get_final_message()`.
- **Protocols** — one client speaks `native`, `openai` or `anthropic`
  via `protocol=`.
- **Async** — `dashscope.AsyncClient()` mirrors the sync surface.
- **Connections** — one keep-alive pool per client, shared across resources.
- **Configuration** — layered: arguments → `dashscope.settings` → environment
  → `~/.dashscope/config.toml`.

## 1.x line

The 1.x API is preserved for backward compatibility under
`dashscope.legacy_1x` and remains importable from the `dashscope` root.
It is frozen: new capabilities land only in the 2.x surface.
See [Legacy 1.x](reference/legacy.md).