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

::: dashscope._config.Config

::: dashscope._config.Settings

## 模块级辅助函数

::: dashscope.chat

::: dashscope.set_region

::: dashscope.save_api_key

## WebSocket 传输

从包根导出的底层 WebSocket 原语，供实时语音会话使用，
也可用于自定义推理任务：

::: dashscope.transport.ws_connection.WsConnection

::: dashscope.transport.ws_connection.WsTask
