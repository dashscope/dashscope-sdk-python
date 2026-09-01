# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union
from urllib.parse import urlencode

import aiohttp
import requests

import dashscope
from dashscope.api_entities.api_request_data import ApiRequestData
from dashscope.api_entities.encryption import Encryption
from dashscope.api_entities.http_request import HttpRequest
from dashscope.api_entities.websocket_request import WebSocketRequest
from dashscope.common.constants import (
    SERVICE_API_PATH,
    ApiProtocol,
    HTTPMethod,
)
from dashscope.common.error import InputDataRequired, UnsupportedApiProtocol
from dashscope.common.logging import logger
from dashscope.common.utils import get_sdk_headers
from dashscope.protocol.websocket import WebsocketStreamingMode


def _build_api_request(  # pylint: disable=too-many-branches
    # pylint: disable=too-many-arguments,too-many-locals
    model: str,
    input: object,  # pylint: disable=redefined-builtin
    task_group: str,
    task: str,
    function: str,
    api_key: str,
    is_service: bool = True,
    # Protocol and connection configuration
    api_protocol: ApiProtocol = ApiProtocol.HTTPS,
    http_method: HTTPMethod = HTTPMethod.POST,
    stream: bool = False,
    async_request: bool = False,
    request_timeout: Optional[int] = None,
    # WebSocket specific
    ws_stream_mode: WebsocketStreamingMode = WebsocketStreamingMode.OUT,
    is_binary_input: bool = False,
    # HTTP specific
    query: bool = False,
    headers: Optional[Dict[str, str]] = None,
    form: Optional[Dict] = None,
    resources: Optional[Dict] = None,
    base_address: Optional[str] = None,
    flattened_output: bool = False,
    extra_url_parameters: Optional[Dict[str, Any]] = None,
    user_agent: str = "",
    session: Optional[Union[requests.Session, aiohttp.ClientSession]] = None,
    task_id: Optional[str] = None,
    enable_encryption: bool = False,
    pre_task_id: Optional[str] = None,
    # Additional parameters for API request data
    **kwargs,
):
    # pylint: disable=too-many-statements
    """Build API request object.

    Args:
        model (str): The model name.
        input (object): The input data for the request.
        task_group (str): The task group for the API path.
        task (str): The task name for the API path.
        function (str): The function name for the API path.
        api_key (str): The API key for authentication.
        is_service (bool, optional): Whether this is a service call.
            Defaults to True.
        api_protocol (ApiProtocol, optional): The protocol to use
            (HTTP, HTTPS, WEBSOCKET). Defaults to ApiProtocol.HTTPS.
        http_method (HTTPMethod, optional): The HTTP method (GET, POST).
            Defaults to HTTPMethod.POST.
        stream (bool, optional): Enable streaming output.
            Defaults to False.
        async_request (bool, optional): Enable async request.
            Defaults to False.
        request_timeout (int, optional): Request timeout in seconds.
            Defaults to None.
        ws_stream_mode (WebsocketStreamingMode, optional): WebSocket
            streaming mode. Defaults to WebsocketStreamingMode.OUT.
        is_binary_input (bool, optional): Whether input is binary data.
            Defaults to False.
        query (bool, optional): Whether this is a query request.
            Defaults to False.
        headers (Dict[str, str], optional): Additional HTTP headers.
            Defaults to None.
        form (Dict, optional): Form data for multipart requests.
            Defaults to None.
        resources (Dict, optional): Resource data. Defaults to None.
        base_address (str, optional): Custom base URL for the API.
            Defaults to None.
        flattened_output (bool, optional): Whether to flatten output.
            Defaults to False.
        extra_url_parameters (Dict[str, Any], optional): Extra URL query
            parameters. Defaults to None.
        user_agent (str, optional): Custom user agent string.
            Defaults to "".
        session (Union[requests.Session, aiohttp.ClientSession], optional):
            Custom session for connection reuse. Defaults to None.
        task_id (str, optional): Task ID for the request.
            Defaults to None.
        enable_encryption (bool, optional): Enable request encryption.
            Defaults to False.
        pre_task_id (str, optional): Previous task ID for WebSocket.
            Defaults to None.
        **kwargs: Additional parameters passed to the API request data.

    Returns:
        HttpRequest or WebSocketRequest: The constructed request object.

    Raises:
        InputDataRequired: If input data is missing or invalid.
        UnsupportedApiProtocol: If the API protocol is not supported.
    """
    # Handle stream mode for WebSocket
    if not stream and ws_stream_mode == WebsocketStreamingMode.OUT:
        ws_stream_mode = WebsocketStreamingMode.NONE

    # Handle user_agent from headers
    if headers and "user-agent" in headers:
        header_ua = headers.pop("user-agent")
        if user_agent:
            user_agent = (
                f"{header_ua}; {user_agent}" if header_ua else user_agent
            )
        else:
            user_agent = header_ua

    encryption = None

    if api_protocol in [ApiProtocol.HTTP, ApiProtocol.HTTPS]:
        if base_address is None:
            base_address = dashscope.base_http_api_url
        if not base_address.endswith("/"):
            http_url = base_address + "/"
        else:
            http_url = base_address

        if is_service:
            http_url = http_url + SERVICE_API_PATH + "/"

        if task_group:
            http_url += f"{task_group}/"
        if task:
            http_url += f"{task}/"
        if function:
            http_url += function
        if extra_url_parameters is not None and extra_url_parameters:
            http_url += "?" + urlencode(extra_url_parameters)

        if enable_encryption is True:
            encryption = Encryption()
            encryption.initialize()
            if encryption.is_valid():
                logger.debug("encryption enabled")

        request = HttpRequest(
            url=http_url,
            api_key=api_key,
            http_method=http_method,
            stream=stream,
            async_request=async_request,
            query=query,
            timeout=request_timeout,
            task_id=task_id,
            flattened_output=flattened_output,
            encryption=encryption,
            user_agent=user_agent,
            session=session,
        )
    elif api_protocol == ApiProtocol.WEBSOCKET:
        if base_address is not None:
            websocket_url = base_address
        else:
            websocket_url = dashscope.base_websocket_api_url
        request = WebSocketRequest(
            url=websocket_url,
            api_key=api_key,
            stream=stream,
            ws_stream_mode=ws_stream_mode,
            is_binary_input=is_binary_input,
            timeout=request_timeout,
            flattened_output=flattened_output,
            pre_task_id=pre_task_id,
            user_agent=user_agent,
        )
    else:
        raise UnsupportedApiProtocol(
            f"Unsupported protocol: {api_protocol}, support [http, https, "
            "websocket]",
        )

    merged_headers = dict(get_sdk_headers())
    if headers is not None:
        merged_headers.update(headers)
    if merged_headers:
        request.add_headers(headers=merged_headers)

    if input is None and form is None:
        raise InputDataRequired("There is no input data and form data")

    if encryption and encryption.is_valid():
        input = encryption.encrypt(input)

    request_data = ApiRequestData(
        model,
        task_group=task_group,
        task=task,
        function=function,
        input_data=input,
        form=form,
        is_binary_input=is_binary_input,
        api_protocol=api_protocol,
    )
    request_data.add_resources(resources)
    request_data.add_parameters(**kwargs)
    request.data = request_data
    return request
