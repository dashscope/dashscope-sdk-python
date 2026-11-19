---
description: "API reference for dev mock modes and local self-diagnosis."
---
# Devkit (Mock & Diagnosis)

`dashscope.devkit` provides two opt-in developer aids with near-zero overhead
when disabled. Nothing here is imported on the production success path.

- **Dev mock** — `static` / `replay` / `record` modes that exercise the
  SDK's HTTP request path. `static` and `replay` run offline without spending
  cloud tokens; `record` sends real requests to the service, which may incur
  charges, and saves the responses for later replay. HTTP only; the
  WebSocket transport is not covered. Activate in code with
  `dashscope.enable_dev_mock(...)` or via `DASHSCOPE_DEV_MOCK=<mode>`.
- **Self-diagnosis** — a local, bilingual rule table that turns a failed call
  into structured fix guidance, surfaced on `DashScopeError.diagnosis`.

The root module forwards the mock switches: `dashscope.enable_dev_mock`,
`dashscope.disable_dev_mock` and `dashscope.dev_mock_active`.

!!! tip "Sharing recordings"
    `record` mode writes real responses to disk; register a redactor with
    `set_redactor` before sharing fixtures.

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
