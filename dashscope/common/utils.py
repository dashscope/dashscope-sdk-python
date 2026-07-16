# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
import json
import os
import platform
import queue
import threading
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, Tuple
from urllib.parse import urlparse

import aiohttp
import requests

from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
from dashscope.common.api_key import get_default_api_key
from dashscope.common.constants import SSE_CONTENT_TYPE
from dashscope.common.error_registry import INTERNAL_ERROR
from dashscope.common.logging import logger
from dashscope.version import __version__


def truncate_error_message(message: str, max_length: int = 200) -> str:
    """Truncate error message for logging to avoid excessive log output.

    Args:
        message: The error message to truncate.
        max_length: Maximum length of the message. Defaults to 200.

    Returns:
        Truncated message with '...' suffix if longer than max_length,
        otherwise original message.
    """
    if len(message) > max_length:
        return message[:max_length] + "..."
    return message


def is_validate_fine_tune_file(file_path: str) -> bool:
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            try:
                json.loads(line)
            except json.decoder.JSONDecodeError:
                return False
    return True


def _get_task_group_and_task(module_name: str) -> Tuple[str, str]:
    """Get task_group and task name.
    get task_group and task name based on api file __name__

    Args:
        module_name (str): The api file __name__

    Returns:
        (str, str): task_group and task
    """
    pkg, task = module_name.rsplit(".", 1)
    task = task.replace("_", "-")
    _, task_group = pkg.rsplit(".", 1)
    return task_group, task


def is_path(path: str) -> bool:
    """Check if the input is a valid local path.

    Args:
        path: The path to check.

    Returns:
        True if it's a valid local path, False otherwise.
    """
    url_parsed = urlparse(path)
    return url_parsed.scheme in ("file", "") and os.path.exists(
        url_parsed.path,
    )


def is_url(url: str) -> bool:
    """Check if the input is a valid URL.

    Args:
        url: The URL to check.

    Returns:
        True if it's a valid URL, False otherwise.
    """
    url_parsed = urlparse(url)
    return url_parsed.scheme in ("http", "https", "oss")


def iter_over_async(ait):
    loop = asyncio.new_event_loop()
    ait = ait.__aiter__()

    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    def iter_thread(loop, message_queue):
        try:
            while True:
                try:
                    done, obj = loop.run_until_complete(get_next())
                    if done:
                        message_queue.put((True, None, None))
                        break
                    message_queue.put((False, None, obj))
                except BaseException as e:  # noqa E722
                    logger.exception(e)
                    message_queue.put((True, e, None))
                    break
        finally:
            loop.close()

    message_queue = queue.Queue()
    x = threading.Thread(
        target=iter_thread,
        args=(loop, message_queue),
        name="iter_async_thread",
        daemon=True,
    )
    x.start()
    while True:
        finished, error, obj = message_queue.get()
        if finished:
            if error is not None:
                yield DashScopeAPIResponse(
                    status_code=INTERNAL_ERROR.status_code,
                    request_id="",
                    code=INTERNAL_ERROR.error_code,
                    message=INTERNAL_ERROR.format_msg(),
                )
            break
        yield obj  # pylint: disable=no-else-break


def async_to_sync(async_generator):
    for message in iter_over_async(async_generator):
        yield message


def get_user_agent():
    try:
        platform_ = (
            platform.platform().replace("\n", "").replace("\r", "").strip()
        )
    except Exception:
        platform_ = "unknown"

    try:
        processor_ = (
            platform.processor().replace("\n", "").replace("\r", "").strip()
        )
    except Exception:
        processor_ = "unknown"

    ua = (
        f"dashscope/{__version__}; python/{platform.python_version()}; "
        f"platform/{platform_}; "
        f"processor/{processor_}"
    )
    return ua


def default_headers(api_key: str = None) -> Dict[str, str]:
    ua = get_user_agent()
    headers = {"user-agent": ua}
    if api_key is None:
        api_key = get_default_api_key()
    headers["Authorization"] = f"Bearer {api_key}"
    headers["Accept"] = "application/json; charset=utf-8"
    return headers


def join_url(base_url: str, *args: str) -> str:
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    url = base_url
    for arg in args:
        if arg is not None:
            url += arg + "/"
    return url[:-1]


@dataclass
class SSEEvent:
    """Server-Sent Events event representation.

    Attributes:
        id: Event ID from the 'id:' field.
        eventType: Event type from the 'event:' field.
        data: Event data from the 'data:' field.
    """

    id: str = ""
    eventType: str = ""
    data: str = ""


def _handle_stream(response: requests.Response):
    # TODO define done message.
    is_error = False
    status_code = HTTPStatus.BAD_REQUEST
    event = SSEEvent(None, None, None)  # type: ignore[arg-type]
    eventType = None
    for line in response.iter_lines():
        if line:
            line = line.decode("utf8")
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith("id:"):
                id = line[len("id:") :]  # pylint: disable=redefined-builtin
                event.id = id.strip()
            elif line.startswith("event:"):
                eventType = line[len("event:") :]
                event.eventType = eventType.strip()
                if eventType == "error":
                    is_error = True
            elif line.startswith("status:"):
                status_code = line[len("status:") :]
                status_code = int(status_code.strip())
            elif line.startswith("data:"):
                line = line[len("data:") :]
                event.data = line.strip()
                if eventType is not None and eventType == "done":
                    continue
                yield (is_error, status_code, event)
                if is_error:
                    break
            else:
                continue  # ignore heartbeat...


def _handle_error_message(error, status_code, flattened_output, headers):
    code = ""
    msg = ""
    request_id = ""

    # Log original error information
    original_code = error.get("code", error.get("error_code", ""))
    original_message = error.get(
        "message",
        error.get("error_message", error.get("msg", "")),
    )
    logger.error(
        "Request failed: status=%s, code=%s, message=%s",
        status_code,
        original_code or "unknown",
        original_message or "unknown",
    )

    if flattened_output:
        error["status_code"] = status_code
        return error

    # Extract message, fallback to INTERNAL_ERROR.format_msg() if empty
    if "message" in error and error["message"]:
        msg = error["message"]
    elif "msg" in error and error["msg"]:
        msg = error["msg"]
    elif "error_message" in error and error["error_message"]:
        msg = error["error_message"]
    else:
        msg = INTERNAL_ERROR.format_msg()

    # Extract code, fallback to INTERNAL_ERROR.error_code if empty
    if "code" in error and error["code"]:
        code = error["code"]
    elif "error_code" in error and error["error_code"]:
        code = error["error_code"]
    else:
        code = INTERNAL_ERROR.error_code

    # Extract request_id
    if "request_id" in error:
        request_id = error["request_id"]

    return DashScopeAPIResponse(
        request_id=request_id,
        status_code=status_code,
        code=code,
        message=msg,
        headers=headers,
    )


def _handle_http_failed_response(
    response: requests.Response,
    flattened_output: bool = False,
) -> DashScopeAPIResponse:
    request_id = ""
    headers = dict(response.headers)
    if "application/json" in response.headers.get("content-type", ""):
        error = response.json()
        return _handle_error_message(
            error,
            response.status_code,
            flattened_output,
            headers,
        )
    elif SSE_CONTENT_TYPE in response.headers.get("content-type", ""):
        msgs = response.content.decode("utf-8").split("\n")
        for msg in msgs:
            if msg.startswith("data:"):
                error = json.loads(msg.replace("data:", "").strip())
                return _handle_error_message(
                    error,
                    response.status_code,
                    flattened_output,
                    headers,
                )
        # SSE 响应中没有有效的错误数据
        error_message = "\n".join(msgs).strip() or INTERNAL_ERROR.format_msg()
        logger.error(
            "Request failed: status=%s, code=%s, message=%s",
            response.status_code,
            INTERNAL_ERROR.error_code,
            truncate_error_message(error_message),
        )
        return DashScopeAPIResponse(
            request_id=request_id,
            status_code=response.status_code,
            code=INTERNAL_ERROR.error_code,
            message=error_message,
            headers=headers,
        )
    else:
        msg = response.content.decode("utf-8")
        error_message = msg or INTERNAL_ERROR.format_msg()
        logger.error(
            "Request failed: status=%s, code=%s, message=%s",
            response.status_code,
            INTERNAL_ERROR.error_code,
            truncate_error_message(error_message),
        )
        if flattened_output:
            return {  # type: ignore[return-value]
                "status_code": response.status_code,
                "code": INTERNAL_ERROR.error_code,
                "message": error_message,
            }
        return DashScopeAPIResponse(
            request_id=request_id,
            status_code=response.status_code,
            code=INTERNAL_ERROR.error_code,
            message=error_message,
            headers=headers,
        )


async def _handle_aio_stream(response):
    # TODO define done message.
    is_error = False
    status_code = HTTPStatus.BAD_REQUEST
    async for line in response.content:
        if line:
            line = line.decode("utf8")
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith("event:error"):
                is_error = True
            elif line.startswith("status:"):
                status_code = line[len("status:") :]
                status_code = int(status_code.strip())
            elif line.startswith("data:"):
                line = line[len("data:") :]
                yield (is_error, status_code, line)
                if is_error:
                    break
            else:
                continue  # ignore heartbeat...


async def _handle_aiohttp_failed_response(
    response: aiohttp.ClientResponse,
    flattened_output: bool = False,
) -> DashScopeAPIResponse:
    request_id = ""
    headers = dict(response.headers)
    if "application/json" in response.content_type:
        error = await response.json()
        # Pass through code, fallback to
        # INTERNAL_ERROR.error_code if not available
        error_code = (
            error.get("code")
            or error.get("error_code")
            or INTERNAL_ERROR.error_code
        )
        # Pass through message, fallback to
        # INTERNAL_ERROR.error_msg if not available
        error_message = (
            error.get("message")
            or error.get("error_message")
            or INTERNAL_ERROR.format_msg()
        )
        logger.error(
            "Request failed: status=%s, code=%s, message=%s",
            response.status,
            error_code,
            truncate_error_message(error_message),
        )
        return _handle_error_message(
            error,
            response.status,
            flattened_output,
            headers,
        )
    elif SSE_CONTENT_TYPE in response.content_type:
        error = None
        raw_data = []
        async for _, _, data in _handle_aio_stream(response):
            raw_data.append(data)
            try:
                error = json.loads(data)
            except json.JSONDecodeError:
                continue
        if error is None:
            raw_content = "\n".join(raw_data).strip() if raw_data else ""
            error_code = INTERNAL_ERROR.error_code
            error_message = raw_content or INTERNAL_ERROR.format_msg()
            logger.error(
                "Request failed: status=%s, code=%s, message=%s",
                response.status,
                error_code,
                truncate_error_message(error_message),
            )
            if flattened_output:
                return {  # type: ignore[return-value]
                    "status_code": response.status,
                    "code": error_code,
                    "message": error_message,
                }
            return DashScopeAPIResponse(
                request_id=request_id,
                status_code=response.status,
                code=error_code,
                message=error_message,
                headers=headers,
            )
        return _handle_error_message(
            error,
            response.status,
            flattened_output,
            headers,
        )
    else:
        msg = await response.text()
        error_code = INTERNAL_ERROR.error_code
        error_message = msg or INTERNAL_ERROR.format_msg()
        logger.error(
            "Request failed: status=%s, code=%s, message=%s",
            response.status,
            error_code,
            truncate_error_message(error_message),
        )
        if flattened_output:
            return {  # type: ignore[return-value]
                "status_code": response.status,
                "code": error_code,
                "message": error_message,
            }
        return DashScopeAPIResponse(
            request_id=request_id,
            status_code=response.status,
            code=error_code,
            message=error_message,
            headers=headers,
        )


def _handle_http_response(
    response: requests.Response,
    flattened_output: bool = False,
):
    response_gen = _handle_http_stream_response(response, flattened_output)
    _, output = next(response_gen)
    # Consume remaining items to ensure generator completes
    for _ in response_gen:
        pass
    return output


# pylint: disable=R1702,too-many-branches,too-many-statements
def _handle_http_stream_response(
    response: requests.Response,
    flattened_output: bool = False,
):
    request_id = ""
    headers = dict(response.headers)
    if (
        response.status_code == HTTPStatus.OK
        and SSE_CONTENT_TYPE in response.headers.get("content-type", "")
    ):
        for is_error, status_code, event in _handle_stream(response):
            if not is_error:
                try:
                    output = None
                    usage = None
                    msg = json.loads(event.data)
                    if flattened_output:
                        msg["status_code"] = response.status_code
                        yield event.eventType, msg
                    else:
                        logger.debug("Stream message: %s", msg)
                        if not is_error:
                            if "output" in msg:
                                output = msg["output"]
                            if "usage" in msg:
                                usage = msg["usage"]
                        if "request_id" in msg:
                            request_id = msg["request_id"]
                        yield event.eventType, DashScopeAPIResponse(
                            request_id=request_id,
                            status_code=HTTPStatus.OK,
                            output=output,
                            usage=usage,
                            headers=headers,
                        )
                except json.JSONDecodeError:
                    error_code = INTERNAL_ERROR.error_code
                    error_message = event.data or INTERNAL_ERROR.format_msg()
                    logger.error(
                        "Request failed: status=%s, code=%s, message=%s",
                        response.status_code,
                        error_code,
                        truncate_error_message(error_message),
                    )
                    if flattened_output:
                        yield event.eventType, {
                            "status_code": response.status_code,
                            "code": error_code,
                            "message": error_message,
                        }
                    else:
                        yield event.eventType, DashScopeAPIResponse(
                            request_id=request_id,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                            output=None,
                            code=error_code,
                            message=error_message,
                            headers=headers,
                        )
                    continue
            else:
                if flattened_output:
                    yield event.eventType, {
                        "status_code": status_code,
                        "message": event.data,
                    }
                else:
                    try:
                        msg = json.loads(event.data)
                        yield event.eventType, DashScopeAPIResponse(
                            request_id=request_id,
                            status_code=status_code,
                            output=None,
                            code=msg.get("code")
                            or msg.get("error_code")
                            or f"http_{status_code}",
                            message=msg.get("message")
                            or msg.get("error_message")
                            or f"HTTP {status_code} error",
                            headers=headers,
                        )  # noqa E501
                    except json.JSONDecodeError:
                        error_code = INTERNAL_ERROR.error_code
                        error_message = (
                            event.data or INTERNAL_ERROR.format_msg()
                        )
                        logger.error(
                            "Request failed: status=%s, code=%s, message=%s",
                            status_code,
                            error_code,
                            truncate_error_message(error_message),
                        )
                        yield event.eventType, DashScopeAPIResponse(
                            request_id=request_id,
                            status_code=status_code,
                            output=None,
                            code=error_code,
                            message=error_message,
                            headers=headers,
                        )
    # pylint: disable=consider-using-in
    elif (
        response.status_code == HTTPStatus.OK
        or response.status_code == HTTPStatus.CREATED
    ):
        json_content = response.json()
        if flattened_output:
            json_content["status_code"] = response.status_code
            yield None, json_content
        else:
            output = None
            usage = None
            code = None
            msg = ""
            if "data" in json_content:
                output = json_content["data"]
            if "code" in json_content:
                code = json_content["code"]
            if "message" in json_content:
                msg = json_content["message"]
            if "output" in json_content:
                output = json_content["output"]
            if "usage" in json_content:
                usage = json_content["usage"]
            if "request_id" in json_content:
                request_id = json_content["request_id"]
                json_content.pop("request_id", None)

            if "data" not in json_content and "output" not in json_content:
                output = json_content

            yield None, DashScopeAPIResponse(
                request_id=request_id,
                status_code=response.status_code,
                code=code,  # type: ignore[arg-type]
                output=output,
                usage=usage,
                message=msg,
                headers=headers,
            )
    else:
        yield None, _handle_http_failed_response(response, flattened_output)
