---
description: "API reference for dev mock modes and local self-diagnosis."
---
# Devkit (Mock & Diagnosis)

`dashscope.devkit` provides two opt-in developer aids with near-zero overhead
when disabled. Nothing here is imported on the production success path.

- **Dev mock** — `static` / `replay` / `record` modes that intercept HTTP
  requests at the transport. HTTP only; the WebSocket transport is not
  covered. Activate in code with `dashscope.enable_dev_mock(...)` or via
  `DASHSCOPE_DEV_MOCK=<mode>`. See the table below for what each mode covers.
- **Self-diagnosis** — a local, bilingual rule table that turns a failed call
  into structured fix guidance, surfaced on `DashScopeError.diagnosis`.

| Mode | Network | Coverage |
| --- | --- | --- |
| `static` | Offline | A parser smoke mock for chat / multimodal only: synthesizes a response locally to exercise protocol adapters, SSE parsing and typed models. Does not simulate tool calls, structured output or job state machines; other resources (embeddings, rerank, ...) raise `DevMockMiss` |
| `replay` | Offline | Answers from recorded real responses; a miss raises `DevMockMiss` and never falls through to the network |
| `record` | Real requests | Passes through to the real service (may incur charges) and records responses for later offline replay |
| WebSocket | — | Not covered by any mode |

The root module forwards the mock switches: `dashscope.enable_dev_mock`,
`dashscope.disable_dev_mock` and `dashscope.dev_mock_active`.

!!! warning "Sharing recordings"
    `record` mode writes full real responses to disk, which can contain
    generated content, tool-call arguments or account data. Register a
    redactor with `set_redactor` before sharing fixtures, but note: the
    redactor only runs on bodies that parse as a single JSON document —
    **SSE streams and binary bodies are stored verbatim and are never
    redacted**. Registering a redactor does not mean every fixture is
    scrubbed.

## Dev mock switches

::: dashscope.devkit.enable_dev_mock

::: dashscope.devkit.disable_dev_mock

::: dashscope.devkit.is_active

::: dashscope.devkit.mock.DevMockMiss

::: dashscope.devkit.mock.set_redactor

## Self-diagnosis

::: dashscope.devkit.diagnosis.Diagnosis

::: dashscope.devkit.diagnosis.diagnose

::: dashscope.devkit.diagnosis.render

::: dashscope.devkit.diagnosis.install_excepthook
