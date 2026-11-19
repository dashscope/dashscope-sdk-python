---
description: "API reference for dashscope Client, AsyncClient, Config, Settings and module-level helpers."
---
# Client & Configuration

The client entry points: `Client` (sync) and `AsyncClient` (async), plus
configuration and module-level helpers.

!!! note "Async scope"
    `AsyncClient` mirrors the sync surface except the duplex audio
    WebSocket sessions, which remain sync-only.

## Clients

::: dashscope.client.Client

::: dashscope.client.AsyncClient

## Configuration

`strict_validation` and `raise_on_error` interact with the per-call
`raw=True` escape hatch — see
[Types — raw responses & parsing](types.md#raw-responses-parsing).

::: dashscope._config.Config

::: dashscope._config.Settings

## Top-level helpers

!!! warning "`set_region` is a legacy 1.x helper"
    `dashscope.set_region()` only updates the historical module-level
    endpoint variables (`base_http_api_url` etc.) used by
    `dashscope.legacy_1x`; it does **not** affect `Client` / `AsyncClient`
    configuration resolution. For the current client use
    `Client(region=..., workspace=...)` or `dashscope.settings.region`.

::: dashscope.chat

::: dashscope.set_region

::: dashscope.save_api_key

## WebSocket transport

Low-level WebSocket primitives exported from the package root, used by the
realtime audio sessions and available for custom inference tasks:

::: dashscope.transport.ws_connection.WsConnection

::: dashscope.transport.ws_connection.WsTask
