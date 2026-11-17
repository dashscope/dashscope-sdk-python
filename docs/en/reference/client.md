---
description: "API reference for dashscope Client, AsyncClient, Config, Settings and module-level helpers."
---
# Client & Configuration

The 2.0 entry points: `Client` (sync) and `AsyncClient` (async), plus
configuration and module-level helpers.

!!! note "Async scope"
    `AsyncClient` mirrors the sync surface except the duplex audio
    WebSocket sessions, which remain sync-only.

## Clients

::: dashscope.client.Client

::: dashscope.client.AsyncClient

## Configuration

::: dashscope._config.Config

::: dashscope._config.Settings

## Top-level helpers

::: dashscope.chat

::: dashscope.set_region

::: dashscope.save_api_key

## WebSocket transport

Low-level WebSocket primitives exported from the package root, used by the
realtime audio sessions and available for custom inference tasks:

::: dashscope.transport.ws_connection.WsConnection

::: dashscope.transport.ws_connection.WsTask
