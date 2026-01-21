# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from dashscope.client.base_api import BaseAioApi, BaseApi
from dashscope.common.constants import ApiProtocol, HTTPMethod


class HttpRequest(BaseApi, BaseAioApi):
    """API for AI-Generated Content(AIGC) models."""

    @classmethod
    async def async_call(  # type: ignore[override]
        cls,
        model: str,
        prompt: str,
        task: str,
        task_group: str = "aigc",
        api_key: str = None,
        api_protocol=ApiProtocol.HTTP,
        http_method=HTTPMethod.POST,  # pylint: disable=unused-argument
        is_binary_input=False,
        **kwargs,
    ) -> DashScopeAPIResponse:
        return await super().async_call(  # type: ignore[misc]
            model=model,
            task_group=task_group,
            task=task,
            api_key=api_key,
            input={"prompt": prompt},
            api_protocol=api_protocol,
            is_binary_input=is_binary_input,
            **kwargs,
        )

    # pylint: disable=arguments-renamed
    @classmethod
    def call(  # type: ignore[override]
        cls,
        model: str,
        prompt: str,
        task: str,
        function: str,
        task_group: str = "aigc",
        api_key: str = None,
        api_protocol=ApiProtocol.HTTP,
        http_method=HTTPMethod.POST,  # pylint: disable=unused-argument
        is_binary_input=False,  # pylint: disable=unused-argument
        **kwargs,
    ) -> DashScopeAPIResponse:
        return super().call(
            model=model,
            task_group=task_group,
            task=task,
            function=function,
            api_key=api_key,
            input={"prompt": prompt},
            api_protocol=api_protocol,
            **kwargs,
        )
