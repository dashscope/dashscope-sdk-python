# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from dashscope.client.base_api import BaseAioApi, BaseApi
from dashscope.common.constants import ApiProtocol
from dashscope.protocol.websocket import WebsocketStreamingMode


class WebSocketRequest(BaseApi, BaseAioApi):
    """API for AI-Generated Content(AIGC) models."""

    @classmethod
    async def aio_call(
        cls,
        model: str,
        prompt: str,
        task: str,
        task_group: str = "aigc",
        api_key: str = None,
        api_protocol=ApiProtocol.WEBSOCKET,
        ws_stream_mode=WebsocketStreamingMode.NONE,
        is_binary_input=False,
        **kwargs,
    ) -> DashScopeAPIResponse:
        return await BaseAioApi.call(
            model=model,
            task_group=task_group,
            task=task,
            api_key=api_key,
            input={"prompt": prompt},
            api_protocol=api_protocol,
            ws_stream_mode=ws_stream_mode,
            is_binary_input=is_binary_input,
            **kwargs,
        )

    @classmethod
    def call(  # type: ignore[override]  # pylint: disable=arguments-renamed
        cls,
        model: str,
        prompt: str,
        task: str,
        task_group: str = "aigc",
        api_key: str = None,
        api_protocol=ApiProtocol.WEBSOCKET,
        ws_stream_mode=WebsocketStreamingMode.NONE,
        is_binary_input=False,
        **kwargs,
    ) -> DashScopeAPIResponse:
        response = BaseApi.call(
            model=model,
            task_group=task_group,
            task=task,
            api_key=api_key,
            input={"prompt": prompt},
            api_protocol=api_protocol,
            ws_stream_mode=ws_stream_mode,
            is_binary_input=is_binary_input,
            **kwargs,
        )

        return response
