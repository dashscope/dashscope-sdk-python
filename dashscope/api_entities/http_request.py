# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime
import json
import ssl
from http import HTTPStatus
from typing import Optional

import aiohttp
import certifi
import requests

from dashscope.api_entities.base_request import AioBaseRequest
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from dashscope.common.constants import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    SSE_CONTENT_TYPE,
    HTTPMethod,
)
from dashscope.common.error import UnsupportedHTTPMethod
from dashscope.common.logging import logger
from dashscope.common.utils import (
    _handle_aio_stream,
    _handle_aiohttp_failed_response,
    _handle_http_failed_response,
    _handle_stream,
)
from dashscope.api_entities.encryption import Encryption


class HttpRequest(AioBaseRequest):
    def __init__(
        self,
        url: str,
        api_key: str,
        http_method: str,
        stream: bool = True,
        async_request: bool = False,
        query: bool = False,
        timeout: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        task_id: str = None,
        flattened_output: bool = False,
        encryption: Optional[Encryption] = None,
        user_agent: str = "",
        session: Optional[requests.Session] = None,
        aio_session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        """HttpSSERequest, processing http server sent event stream.

        Args:
            url (str): The request url.
            api_key (str): The api key.
            method (str): The http method(GET|POST).
            stream (bool, optional): Is stream request. Defaults to True.
            timeout (int, optional): Total request timeout.
                Defaults to DEFAULT_REQUEST_TIMEOUT_SECONDS.
            user_agent (str, optional): Additional user agent string to
                append. Defaults to ''.
            session (Optional[requests.Session]): External session for
                connection reuse (sync). Defaults to None.
            aio_session (Optional[aiohttp.ClientSession]): External session
                for connection reuse (async). Defaults to None.
        """

        super().__init__(user_agent=user_agent)
        self.url = url
        self.flattened_output = flattened_output
        self.async_request = async_request
        self.encryption = encryption
        self._external_session = session
        self._external_aio_session = aio_session
        base_headers = getattr(self, "headers", {})
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            **base_headers,
        }

        if encryption and encryption.is_valid():
            self.headers = {
                "X-DashScope-EncryptionKey": json.dumps(
                    {
                        "public_key_id": encryption.get_pub_key_id(),
                        "encrypt_key": encryption.get_encrypted_aes_key_str(),
                        "iv": encryption.get_base64_iv_str(),
                    },
                ),
                **self.headers,
            }

        self.query = query
        if self.async_request and self.query is False:
            self.headers = {
                "X-DashScope-Async": "enable",
                **self.headers,
            }
        self.method = http_method
        if self.method == HTTPMethod.POST:
            self.headers["Content-Type"] = "application/json"

        self.stream = stream
        if self.stream:
            self.headers["Accept"] = SSE_CONTENT_TYPE
            self.headers["X-Accel-Buffering"] = "no"
            self.headers["X-DashScope-SSE"] = "enable"
        if self.query:
            self.url = self.url.replace("api", "api-task")
            self.url += f"{task_id}"
        if timeout is None:
            self.timeout = DEFAULT_REQUEST_TIMEOUT_SECONDS
        else:
            self.timeout = timeout  # type: ignore[has-type]

    def get_external_session(self) -> Optional[requests.Session]:
        """
        获取外部传入的同步 Session

        Returns:
            Optional[requests.Session]: 外部 Session，如果未设置则返回 None
        """
        return self._external_session

    def get_external_aio_session(self) -> Optional[aiohttp.ClientSession]:
        """
        获取外部传入的异步 Session

        Returns:
            Optional[aiohttp.ClientSession]: 外部异步 Session，如果未设置则返回 None
        """
        return self._external_aio_session

    def add_header(self, key, value):
        self.headers[key] = value

    def add_headers(self, headers):
        self.headers = {**self.headers, **headers}

    def call(self):
        response = self._handle_request()
        if self.stream:
            return (item for item in response)
        else:
            output = next(response)
            try:
                next(response)
            except StopIteration:
                pass
            return output

    async def aio_call(self):
        response = self._handle_aio_request()
        if self.stream:
            return (item async for item in response)
        else:
            result = await response.__anext__()
            try:
                await response.__anext__()
            except StopAsyncIteration:
                pass
            return result

    async def _get_aio_session(self):
        """获取异步 Session（优先级：外部 > 全局 > 临时）"""
        # 1. 检查是否有外部传入的 Session（最高优先级）
        if self._external_aio_session is not None:
            logger.debug(
                "Using external async session for request: %s",
                self.url,
            )
            return self._external_aio_session, False

        # 2. 尝试获取全局异步 Session
        from dashscope.common.aio_session_manager import AioSessionManager

        manager = await AioSessionManager.get_instance()
        global_session = await manager.get_session()

        if global_session is not None:
            logger.debug(
                "Using global async session for request: %s",
                self.url,
            )
            return global_session, False

        # 3. 创建临时 Session（保持向后兼容）
        connector = aiohttp.TCPConnector(
            ssl=ssl.create_default_context(
                cafile=certifi.where(),
            ),
        )
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self.headers,
        )
        logger.debug(
            "Using temporary async session for request: %s",
            self.url,
        )
        return session, True

    async def _execute_aio_request(self, session, timeout):
        """执行异步 HTTP 请求"""
        logger.debug("Starting request: %s", self.url)

        if self.method == HTTPMethod.POST:
            return await self._execute_post_request(session, timeout)
        if self.method == HTTPMethod.GET:
            return await self._execute_get_request(session, timeout)

        raise UnsupportedHTTPMethod(
            f"Unsupported http method: {self.method}",
        )

    async def _execute_post_request(self, session, timeout):
        """执行 POST 请求"""
        is_form, obj = False, {}
        if hasattr(self, "data") and self.data is not None:
            is_form, obj = self.data.get_aiohttp_payload()

        if is_form:
            headers = {**self.headers, **obj.headers}
            return await session.post(
                url=self.url,
                data=obj,
                headers=headers,
                timeout=timeout,
            )

        return await session.request(
            "POST",
            url=self.url,
            json=obj,
            headers=self.headers,
            timeout=timeout,
        )

    async def _execute_get_request(self, session, timeout):
        """执行 GET 请求"""
        params = {}
        if hasattr(self, "data") and self.data is not None:
            params = getattr(self.data, "parameters", {})
        if params:
            params = self.__handle_parameters(params)

        return await session.get(
            url=self.url,
            params=params,
            headers=self.headers,
            timeout=timeout,
        )

    async def _handle_aio_request(self):
        try:
            # 获取 Session（优先级：外部 > 全局 > 临时）
            session, should_close = await self._get_aio_session()

            try:
                # 设置超时
                timeout = aiohttp.ClientTimeout(total=self.timeout)

                # 执行请求
                response = await self._execute_aio_request(session, timeout)

                logger.debug("Response returned: %s", self.url)
                async with response:
                    async for rsp in self._handle_aio_response(response):
                        yield rsp
            finally:
                # 只关闭临时 Session
                if should_close:
                    await session.close()
                    logger.debug("Temporary async session closed")

        except aiohttp.ClientConnectorError as e:
            logger.error(e)
            raise e
        except BaseException as e:
            logger.error(e)
            raise e

    @staticmethod
    def __handle_parameters(params: dict) -> dict:
        # pylint: disable=too-many-return-statements
        def __format(value):
            if isinstance(value, bool):
                return str(value).lower()
            elif isinstance(value, (str, int, float)):
                return value
            elif value is None:
                return ""
            elif isinstance(value, (datetime.datetime, datetime.date)):
                return value.isoformat()
            elif isinstance(value, (list, tuple)):
                return ",".join(str(__format(x)) for x in value)
            elif isinstance(value, dict):
                return json.dumps(value)
            else:
                try:
                    return str(value)
                except Exception as e:
                    # pylint: disable=raise-missing-from
                    raise ValueError(
                        f"Unsupported type {type(value)} for param formatting: {e}",  # noqa: E501
                    )

        formatted = {}
        for k, v in params.items():
            formatted[k] = __format(v)
        # pylint: disable=too-many-statements
        return formatted

    async def _handle_aio_response(  # pylint: disable=too-many-branches, too-many-statements # noqa: E501
        self,
        response: aiohttp.ClientResponse,
    ):
        request_id = ""
        if (
            response.status == HTTPStatus.OK
            and self.stream
            and SSE_CONTENT_TYPE in response.content_type
        ):
            async for is_error, status_code, data in _handle_aio_stream(
                response,
            ):
                try:
                    output = None
                    usage = None
                    msg = json.loads(data)
                    if not is_error:
                        if "output" in msg:
                            output = msg["output"]
                        if "usage" in msg:
                            usage = msg["usage"]
                    if "request_id" in msg:
                        request_id = msg["request_id"]
                except json.JSONDecodeError:
                    yield DashScopeAPIResponse(
                        request_id=request_id,
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        code="Unknown",
                        message=data,
                    )
                    continue
                if is_error:
                    yield DashScopeAPIResponse(
                        request_id=request_id,
                        status_code=status_code,
                        code=msg["code"],
                        message=msg["message"],
                    )
                else:
                    if self.encryption and self.encryption.is_valid():
                        output = self.encryption.decrypt(output)
                    yield DashScopeAPIResponse(
                        request_id=request_id,
                        status_code=HTTPStatus.OK,
                        output=output,
                        usage=usage,
                    )
        elif (
            response.status == HTTPStatus.OK
            and "multipart" in response.content_type
        ):
            reader = aiohttp.MultipartReader.from_response(response)
            output = {}
            while True:
                part = await reader.next()
                if part is None:
                    # pylint: disable=consider-using-get
                    break
                output[part.name] = await part.read()
            if "request_id" in output:  # pylint: disable=consider-using-get
                request_id = output["request_id"]
            if self.encryption and self.encryption.is_valid():
                output = self.encryption.decrypt(output)
            yield DashScopeAPIResponse(
                request_id=request_id,
                status_code=HTTPStatus.OK,
                output=output,
            )
        elif response.status == HTTPStatus.OK:
            json_content = await response.json()
            output = None
            usage = None
            if "output" in json_content and json_content["output"] is not None:
                output = json_content["output"]
            # Compatible with wan
            elif (
                "data" in json_content
                and json_content["data"] is not None
                and isinstance(json_content["data"], list)
                and len(json_content["data"]) > 0
                and "task_id" in json_content["data"][0]
            ):
                output = json_content
            if "usage" in json_content:
                usage = json_content["usage"]
            if "request_id" in json_content:
                request_id = json_content["request_id"]
            if self.encryption and self.encryption.is_valid():
                output = self.encryption.decrypt(output)
            yield DashScopeAPIResponse(
                request_id=request_id,
                status_code=HTTPStatus.OK,
                output=output,
                usage=usage,
            )
        else:
            yield await _handle_aiohttp_failed_response(response)

    def _handle_response(  # pylint: disable=too-many-branches
        self,
        response: requests.Response,
    ):
        request_id = ""
        if (
            response.status_code == HTTPStatus.OK
            and self.stream
            and SSE_CONTENT_TYPE
            in response.headers.get(
                "content-type",
                "",
            )
        ):
            for is_error, status_code, event in _handle_stream(response):
                try:
                    data = event.data
                    output = None
                    usage = None
                    msg = json.loads(data)
                    logger.debug("Stream message: %s", msg)
                    if not is_error:
                        if "output" in msg:
                            output = msg["output"]
                        if "usage" in msg:
                            usage = msg["usage"]
                    if "request_id" in msg:
                        request_id = msg["request_id"]
                except json.JSONDecodeError:
                    yield DashScopeAPIResponse(
                        request_id=request_id,
                        status_code=HTTPStatus.BAD_REQUEST,
                        output=None,
                        code="Unknown",
                        message=data,
                    )
                    continue
                if is_error:
                    yield DashScopeAPIResponse(
                        request_id=request_id,
                        status_code=status_code,
                        output=None,
                        code=msg["code"]
                        if "code" in msg
                        else None,  # noqa E501
                        message=msg["message"] if "message" in msg else None,
                    )  # noqa E501
                else:
                    if self.flattened_output:
                        yield msg
                    else:
                        if self.encryption and self.encryption.is_valid():
                            output = self.encryption.decrypt(output)
                        yield DashScopeAPIResponse(
                            request_id=request_id,
                            status_code=HTTPStatus.OK,
                            output=output,
                            usage=usage,
                        )
        elif response.status_code == HTTPStatus.OK:
            json_content = response.json()
            logger.debug("Response: %s", json_content)
            output = None
            usage = None
            if "task_id" in json_content:
                output = {"task_id": json_content["task_id"]}
            if "output" in json_content:
                output = json_content["output"]
            if "usage" in json_content:
                usage = json_content["usage"]
            if "request_id" in json_content:
                request_id = json_content["request_id"]
            if self.flattened_output:
                yield json_content
            else:
                if self.encryption and self.encryption.is_valid():
                    output = self.encryption.decrypt(output)
                yield DashScopeAPIResponse(
                    request_id=request_id,
                    status_code=HTTPStatus.OK,
                    output=output,
                    usage=usage,
                )
        else:
            yield _handle_http_failed_response(response)

    def _handle_request(self):
        """
        处理 HTTP 请求

        优先级：
        1. 外部传入的 session（用户自定义）
        2. 全局 SessionManager（如果启用）
        3. 临时 session（保持原有行为）
        """
        try:
            from dashscope.common.session_manager import SessionManager

            # 优先使用外部传入的 session
            if self._external_session is not None:
                session = self._external_session
                should_close = False
            else:
                # 尝试使用全局 SessionManager
                session_manager = SessionManager.get_instance()
                session = session_manager.get_session()
                should_close = False

                # 如果未启用连接复用，创建临时 session
                if session is None:
                    session = requests.Session()
                    should_close = True

            try:
                # 执行请求
                if self.method == HTTPMethod.POST:
                    is_form, form, obj = self.data.get_http_payload()
                    if is_form:
                        headers = {**self.headers}
                        headers.pop("Content-Type")
                        response = session.post(
                            url=self.url,
                            data=obj,
                            files=form,
                            headers=headers,
                            timeout=self.timeout,
                        )
                    else:
                        logger.debug("Request body: %s", obj)
                        response = session.post(
                            url=self.url,
                            stream=self.stream,
                            json=obj,
                            headers={**self.headers},
                            timeout=self.timeout,
                        )
                elif self.method == HTTPMethod.GET:
                    response = session.get(
                        url=self.url,
                        params=self.data.parameters,
                        headers=self.headers,
                        timeout=self.timeout,
                    )
                else:
                    raise UnsupportedHTTPMethod(
                        f"Unsupported http method: {self.method}",
                    )

                for rsp in self._handle_response(response):
                    yield rsp
            finally:
                # 只关闭临时创建的 session
                if should_close:
                    session.close()

        except BaseException as e:
            logger.error(e)
            raise e
