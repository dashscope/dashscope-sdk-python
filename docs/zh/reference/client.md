---
description: "dashscope Client、AsyncClient、Config、Settings 与模块级辅助函数 API 参考。"
---
# 客户端与配置

客户端入口：`Client`（同步）与 `AsyncClient`（异步），以及配置和模块级辅助函数。

!!! note "异步范围"
    `AsyncClient` 镜像同步接口，但双工语音 WebSocket 会话
    除外——它们仅支持同步。

## 客户端

::: dashscope.client.Client

::: dashscope.client.AsyncClient

## 配置

`strict_validation`、`raise_on_error` 与单次调用的 `raw=True`
相互配合——见[类型——原始响应与解析](types.md#raw-responses-parsing)。

::: dashscope._config.Config

::: dashscope._config.Settings

## 模块级辅助函数

!!! warning "`set_region` 是 1.x 历史接口"
    `dashscope.set_region()` 只修改历史模块级端点变量
    （`base_http_api_url` 等），供 `dashscope.legacy_1x` 使用，
    **不影响** `Client` / `AsyncClient` 的配置解析。
    当前客户端请使用 `Client(region=..., workspace=...)` 或
    `dashscope.settings.region`。

::: dashscope.chat

::: dashscope.set_region

::: dashscope.save_api_key

## WebSocket 传输

从包根导出的底层 WebSocket 原语，供实时语音会话使用，
也可用于自定义推理任务：

::: dashscope.transport.ws_connection.WsConnection

::: dashscope.transport.ws_connection.WsTask
