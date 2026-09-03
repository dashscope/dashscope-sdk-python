# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from dashscope.client.base_api import GetMixin, ListMixin, _get
from dashscope.common.utils import get_api_module, join_url
import dashscope


class Models(ListMixin, GetMixin):
    SUB_PATH = "models"

    @classmethod
    def get(  # type: ignore[override]
        cls,
        name: str,
        api_key: str = None,
        **kwargs,
    ) -> DashScopeAPIResponse:
        """Get the model information.

        Args:
            name (str): The model name.
            api_key (str, optional): The api key. Defaults to None.
            workspace (str): The dashscope workspace id.

        Returns:
            DashScopeAPIResponse: The model information.
        """
        from http import HTTPStatus

        # Use query parameter to filter by model name on server side
        # API endpoint: /api/v1/models?model={name}&page_no=1&page_size=1
        url = join_url(dashscope.base_http_api_url, cls.SUB_PATH.lower())
        params = {"model": name, "page_no": 1, "page_size": 1}

        response = _get(
            url,
            params=params,
            api_key=api_key,
            module=get_api_module(cls.__module__),
            **kwargs,
        )

        if response.status_code != HTTPStatus.OK:
            return response  # type: ignore[return-value]

        output = response.output
        if not output or "models" not in output or not output["models"]:
            response.status_code = 404
            response.message = f"Model '{name}' not found"
            response.output = None
            return response  # type: ignore[return-value]

        # Return the first (and only) model from the filtered list
        response.output = output["models"][0]
        return response  # type: ignore[return-value]

    @classmethod
    def list(  # type: ignore[override]
        cls,
        page=1,
        page_size=10,
        api_key: str = None,
        **kwargs,
    ) -> DashScopeAPIResponse:
        """List models.

        Args:
            api_key (str, optional): The api key
            page (int, optional): Page number. Defaults to 1.
            page_size (int, optional): Items per page. Defaults to 10.

        Returns:
            DashScopeAPIResponse: The models.
        """
        # type: ignore
        return super().list(page, page_size, api_key=api_key, **kwargs)  # type: ignore[return-value] # pylint: disable=line-too-long # noqa: E501
