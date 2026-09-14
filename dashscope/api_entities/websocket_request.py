# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
import json
import uuid
from http import HTTPStatus
from typing import Tuple, Union

import aiohttp

from dashscope.api_entities.base_request import AioBaseRequest
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from dashscope.common.constants import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    WEBSOCKET_ERROR_CODE,
)
from dashscope.common.error import (
    RequestFailure,
    UnexpectedMessageReceived,
    UnknownMessageReceived,
)
from dashscope.common.error_registry import (
    SERVICE_UNAVAILABLE,
    AUTH_FAILED,
    INTERNAL_ERROR,
    PERMISSION_DENIED,
)
from dashscope.common.logging import logger
from dashscope.common.utils import async_to_sync
from dashscope.protocol.websocket import (
    ACTION_KEY,
    ERROR_MESSAGE,
    ERROR_NAME,
    EVENT_KEY,
    HEADER,
    TASK_ID,
    ActionType,
    EventType,
    WebsocketStreamingMode,
)


def _is_service_unavailable(exc: BaseException) -> bool:
    """Decide whether a failed connect means the service is unavailable.

    A connect failure never receives an HTTP status, so only an explicit
    phrase qualifies. Matching a bare "503" also fires on addresses such as
    ``host:5030`` or ``10.50.30.1``.
    """
    return "service unavailable" in str(exc).lower()


def _handshake_error(status):
    """Map a WebSocket handshake status to the public error to report.

    ``None`` means the status is not one this SDK classifies, and the caller
    re-raises rather than inventing a response.
    """
    if status == HTTPStatus.UNAUTHORIZED:
        return AUTH_FAILED
    if status == HTTPStatus.FORBIDDEN:
        # The credentials worked but lack permission. Reporting a 403 as a 401
        # sends the caller to rotate an API key that is actually valid.
        return PERMISSION_DENIED
    if status == HTTPStatus.SERVICE_UNAVAILABLE:
        return SERVICE_UNAVAILABLE
    if status == HTTPStatus.INTERNAL_SERVER_ERROR:
        return INTERNAL_ERROR
    return None


def _internal_error_message(exc: BaseException, detail: str = "") -> str:
    """Compose the user-visible message for a 500 surfaced over WebSocket.

    The registry text alone says nothing about which failure happened, so the
    originating detail is appended. ``detail`` wins when the caller has a more
    specific string than ``str(exc)``, such as a handshake error message.
    """
    suffix = detail or f"{type(exc).__name__}: {exc}"
    return f"{INTERNAL_ERROR.error_msg} (SDK Internal Error: {suffix})"


class WebSocketRequest(AioBaseRequest):
    def __init__(
        self,
        url: str,
        api_key: str,
        stream: bool = True,
        ws_stream_mode: str = WebsocketStreamingMode.OUT,
        is_binary_input: bool = False,
        timeout: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        flattened_output: bool = False,
        pre_task_id=None,
        user_agent: str = "",
    ) -> None:
        super().__init__(user_agent=user_agent)
        # pylint: disable=pointless-string-statement
        """HttpRequest.

        Args:
            url (str): The request url.
            api_key (str): The api key.
            method (str): The http method(GET|POST).
            stream (bool, optional): Is stream request. Defaults to False.
            timeout (int, optional): Total request timeout.
                Defaults to DEFAULT_REQUEST_TIMEOUT_SECONDS.
        """
        self.url = url
        self.stream = stream
        self.flattened_output = flattened_output
        if timeout is None:
            self.timeout = DEFAULT_REQUEST_TIMEOUT_SECONDS
        else:
            self.timeout = timeout  # type: ignore[has-type]
        self.ws_stream_mode = ws_stream_mode
        self.is_binary_input = is_binary_input

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            **self.headers,  # type: ignore[has-type]
        }

        self.task_headers = {
            "streaming": self.ws_stream_mode,
        }
        self.pre_task_id = pre_task_id
        self.ws = None

    def add_headers(self, headers):
        self.headers.update(headers)

    def call(self):
        response = async_to_sync(self.connection_handler())
        if self.stream:
            return (item for item in response)
        else:
            output = next(response)
            try:
                next(response)
            except StopIteration:
                pass
            return output

    async def close(self):
        ws = getattr(self, "ws", None)
        if ws is not None and not ws.closed:
            await ws.close()

    async def aio_call(self):
        response = self.connection_handler()
        if self.stream:
            return (item async for item in response)
        else:
            result = await response.__anext__()
            try:
                await response.__anext__()
            except StopAsyncIteration:
                pass
            return result

    async def connection_handler(
        self,
    ):  # pylint: disable=too-many-branches,too-many-statements
        try:
            task_id = None
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout,
                ),
                trust_env=True,
            ) as session:
                async with session.ws_connect(
                    self.url,
                    headers=self.headers,
                    heartbeat=30,
                ) as ws:
                    self.ws = ws  # Store ws reference for close() method
                    await self._start_task(ws)  # send start task action.
                    task_id = self.task_headers["task_id"]
                    await self._wait_for_task_started(
                        ws,
                    )  # wait for task started event. # noqa E501
                    if self.ws_stream_mode == WebsocketStreamingMode.NONE:
                        if self.is_binary_input:  # send the binary package
                            data = self.data.get_batch_binary_data()
                            await ws.send_bytes(list(data.values())[0])
                        (
                            is_binary,
                            result,
                        ) = await self._receive_batch_data_task(  # noqa E501
                            ws,
                        )
                        # do not need send finished task message.
                        yield self._to_DashScopeAPIResponse(
                            task_id,
                            is_binary,
                            result,
                        )
                    elif self.ws_stream_mode == WebsocketStreamingMode.IN:
                        # server is in, we send streaming out.
                        await self._send_continue_task_data(ws)
                        (
                            is_binary,
                            result,
                        ) = await self._receive_batch_data_task(  # noqa E501
                            ws,
                        )
                        # do not need send finished task message.
                        yield self._to_DashScopeAPIResponse(
                            task_id,
                            is_binary,
                            result,
                        )
                    elif self.ws_stream_mode == WebsocketStreamingMode.OUT:
                        # we send batch data, server streaming output data.
                        if self.is_binary_input:  # send only binary package.
                            data = self.data.get_batch_binary_data()
                            await ws.send_bytes(list(data.values())[0])
                        async for is_binary, message in self._receive_streaming_data_task(  # noqa E501  # pylint: disable=line-too-long
                            ws,
                        ):
                            yield self._to_DashScopeAPIResponse(
                                task_id,
                                is_binary,
                                message,
                            )
                    else:  # duplex mode
                        bg_task = asyncio.create_task(
                            self._send_continue_task_data(ws),
                        )
                        try:
                            async for is_binary, message in self._receive_streaming_data_task(  # noqa E501  # pylint: disable=line-too-long
                                ws,
                            ):
                                yield self._to_DashScopeAPIResponse(
                                    task_id,
                                    is_binary,
                                    message,
                                )
                            # Normal completion: wait for the send task.
                            await bg_task
                        except BaseException:
                            # Abnormal exit (error or consumer closed the
                            # stream early): cancel to avoid leaking it.
                            if not bg_task.done():
                                bg_task.cancel()
                            await asyncio.gather(
                                bg_task,
                                return_exceptions=True,
                            )
                            raise
        except RequestFailure as e:
            yield DashScopeAPIResponse(
                request_id=e.request_id,
                status_code=e.http_code,
                output=None,
                code=e.name,
                message=e.message,
            )
        except aiohttp.ClientConnectorError as e:
            logger.exception(e)
            if _is_service_unavailable(e):
                yield DashScopeAPIResponse(
                    request_id=task_id if task_id else "",
                    status_code=SERVICE_UNAVAILABLE.status_code,
                    code=SERVICE_UNAVAILABLE.error_code,
                    message=SERVICE_UNAVAILABLE.error_msg,
                )
                return

            yield DashScopeAPIResponse(
                request_id=task_id if task_id else "",
                status_code=INTERNAL_ERROR.status_code,
                code=INTERNAL_ERROR.error_code,
                message=_internal_error_message(e),
            )
        except aiohttp.WSServerHandshakeError as e:
            original_msg = e.message or ""
            handshake_error = _handshake_error(e.status)

            if handshake_error is None:
                # Log unexpected status codes for debugging
                logger.warning(
                    "WebSocket handshake failed with unexpected "
                    "status %s: %s",
                    e.status,
                    original_msg,
                )
                raise e

            message = (
                _internal_error_message(e, original_msg)
                if handshake_error is INTERNAL_ERROR
                else handshake_error.error_msg
            )
            yield DashScopeAPIResponse(
                request_id=task_id if task_id else "",
                status_code=handshake_error.status_code,
                code=handshake_error.error_code,
                message=message,
            )
        except Exception as e:
            logger.exception(e)
            yield DashScopeAPIResponse(
                request_id=task_id if task_id else "",
                status_code=INTERNAL_ERROR.status_code,
                code=INTERNAL_ERROR.error_code,
                message=_internal_error_message(e),
            )

    def _to_DashScopeAPIResponse(self, task_id, is_binary, result):
        if is_binary:
            return DashScopeAPIResponse(
                request_id=task_id,
                status_code=HTTPStatus.OK,
                output=result,
            )
        else:
            # get output and usage.
            output = {}
            usage = {}
            if "output" in result:
                output = result["output"]
            if "usage" in result:
                usage = result["usage"]
            return DashScopeAPIResponse(
                request_id=task_id,
                status_code=HTTPStatus.OK,
                output=output,
                usage=usage,
            )

    async def _receive_streaming_data_task(self, ws):
        # check if request stream data, re return an iterator,
        # otherwise we collect data and return user.
        # no matter what, the response is streaming
        is_binary_output = False
        while True:  # pylint: disable=R1702
            msg = await ws.receive()
            await self._check_websocket_unexpected_message(msg)
            if msg.type == aiohttp.WSMsgType.TEXT:
                msg_json = msg.json()
                logger.debug("Receive %s event", msg_json[HEADER][EVENT_KEY])
                if msg_json[HEADER][EVENT_KEY] == EventType.GENERATED:
                    payload = msg_json["payload"]
                    yield False, payload
                elif msg_json[HEADER][EVENT_KEY] == EventType.FINISHED:
                    payload = None
                    if "payload" in msg_json:
                        payload = msg_json["payload"]
                    logger.debug(payload)
                    if payload:
                        yield False, payload
                    else:
                        if not self.stream:
                            if is_binary_output:
                                yield True, payload
                            else:
                                yield False, payload
                    break
                elif msg_json[HEADER][EVENT_KEY] == EventType.FAILED:
                    self._on_failed(msg_json)
                else:
                    error = f"Receive unknown message: {msg_json}"
                    logger.error(error)
                    raise UnknownMessageReceived(error)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                is_binary_output = True
                yield True, msg.data

    def _on_failed(self, details):
        error = RequestFailure(
            request_id=details[HEADER][TASK_ID],
            http_code=WEBSOCKET_ERROR_CODE,
            name=details[HEADER][ERROR_NAME],
            message=details[HEADER][ERROR_MESSAGE],
        )
        logger.error(error)
        raise error

    async def _start_task(self, ws):
        if self.pre_task_id is not None:
            self.task_headers["task_id"] = self.pre_task_id
        else:
            self.task_headers["task_id"] = uuid.uuid4().hex  # create task id.
        task_header = {**self.task_headers, ACTION_KEY: ActionType.START}
        # for binary data, the start action has no input, only parameters.
        start_data = self.data.get_websocket_start_data()
        message = self._build_up_message(task_header, start_data)
        logger.debug("Send start task: %s", message)
        await ws.send_str(message)

    async def _send_finished_task(self, ws):
        task_header = {**self.task_headers, ACTION_KEY: ActionType.FINISHED}
        payload = {"input": {}}
        message = self._build_up_message(task_header, payload)
        logger.debug("Send finish task: %s", message)
        await ws.send_str(message)

    async def _send_continue_task_data(self, ws):
        headers = {**self.task_headers, ACTION_KEY: ActionType.CONTINUE}
        for input_item in self.data.get_websocket_continue_data():
            if len(input_item) > 0:
                if self.is_binary_input and isinstance(
                    input_item,
                    (bytes, bytearray, memoryview),
                ):
                    await ws.send_bytes(input_item)
                    logger.debug(
                        "Send continue task with bytes: %s",
                        len(input_item),
                    )
                elif self.is_binary_input and isinstance(input_item, dict):
                    binary_data = next(iter(input_item.values()))
                    if isinstance(binary_data, (bytes, bytearray, memoryview)):
                        await ws.send_bytes(binary_data)
                        logger.debug(
                            "Send continue task with list[byte]: %s",
                            len(input_item),
                        )
                    else:
                        message = self._build_up_message(
                            headers=headers,
                            payload=input_item,
                        )
                        logger.debug("Send continue task: %s", message)
                        await ws.send_str(message)
                else:
                    message = self._build_up_message(
                        headers=headers,
                        payload=input_item,
                    )
                    logger.debug("Send continue task: %s", message)
                    await ws.send_str(message)
            await asyncio.sleep(0.000001)

        # data send completed, and send task completed.
        await self._send_finished_task(ws)

    async def _receive_batch_data_task(
        self,
        ws,
    ) -> Tuple[bool, Union[str, bytes]]:
        """_summary_

        Args:
            ws (connection): The ws connection.

        Raises:
            UnknownMessageReceived: The message is unexpected.

        Returns:
            Tuple[bool, str]: is output is binary, output
        """
        while True:
            msg = await ws.receive()
            await self._check_websocket_unexpected_message(msg)
            if msg.type == aiohttp.WSMsgType.TEXT:
                msg_json = msg.json()
                logger.debug("Receive %s event", msg_json[HEADER][EVENT_KEY])
                if msg_json[HEADER][EVENT_KEY] == EventType.GENERATED:
                    payload = msg_json["payload"]
                    return False, payload
                elif msg_json[HEADER][EVENT_KEY] == EventType.FINISHED:
                    payload = msg_json["payload"]
                    return False, payload
                elif msg_json[HEADER][EVENT_KEY] == EventType.FAILED:
                    self._on_failed(msg_json)
                else:
                    error = f"Receive unknown message: {msg_json}"
                    logger.error(error)
                    raise UnknownMessageReceived(error)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                return True, msg.data  # get binary result data.

    async def _wait_for_task_started(self, ws):
        while True:
            msg = await ws.receive()
            await self._check_websocket_unexpected_message(msg)
            if msg.type == aiohttp.WSMsgType.TEXT:
                msg_json = msg.json()
                logger.debug("Receive %s event", msg_json[HEADER][EVENT_KEY])
                if msg_json[HEADER][EVENT_KEY] == EventType.STARTED:
                    return
                elif msg_json[HEADER][EVENT_KEY] == EventType.FAILED:
                    self._on_failed(msg_json)
                else:
                    raise UnexpectedMessageReceived(
                        "Receive unexpected message, expect task-started, "
                        f"real: {msg_json[HEADER][EVENT_KEY]}.",
                    )
            elif msg.type == aiohttp.WSMsgType.BINARY:
                raise UnexpectedMessageReceived(
                    "Receive unexpected binary message when wait for task-started",  # noqa E501
                )

    async def _check_websocket_unexpected_message(self, msg):
        if msg.type == aiohttp.WSMsgType.CLOSED:
            details = f"WSMsgType.CLOSE, data: {msg.data}, extra: {msg.extra}"
            logger.error("Connection unexpected closed!")
            raise UnexpectedMessageReceived(
                f"Receive unexpected websocket close message, "
                f"details: {details}",
            )
        if msg.type == aiohttp.WSMsgType.ERROR:
            details = f"WSMsgType.ERROR, data: {msg.data}, extra: {msg.extra}"
            logger.error("Connection error: %s", details)
            raise UnexpectedMessageReceived(
                f"Receive unexpected websocket error message "
                f"details: {details}.",
            )

    def _build_up_message(self, headers, payload):
        message = {"header": headers, "payload": payload}
        return json.dumps(message, ensure_ascii=False)
