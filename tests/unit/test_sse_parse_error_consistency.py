# -*- coding: utf-8 -*-
"""All four SSE parse-failure paths must report the same error.

A body the SDK cannot parse is a server-side defect, not a bad request, so
every path reports 500 InternalServerError and carries the offending payload
in ``.message``. These paths had drifted apart: two reported 400
BadRequestError (one of them claiming a parameter named "response data" was
missing) and one logged the payload but discarded it from the response.
"""
from http import HTTPStatus
from types import SimpleNamespace

import pytest

from dashscope.api_entities.aiohttp_request import AioHttpRequest
from dashscope.api_entities.http_request import HttpRequest
from dashscope.common.error_registry import INTERNAL_ERROR
from dashscope.common.utils import _handle_http_stream_response

MALFORMED = '{"output": truncated'

# The stream helpers default their status_code to 400 before any status: line
# arrives, so a parse branch that echoed it would report 400 by accident.
SSE_LINES = [b"event:result", b"data:" + MALFORMED.encode()]


class _FakeSyncResponse:
    status_code = HTTPStatus.OK
    headers = {"content-type": "text/event-stream"}

    def iter_lines(self):
        return iter(SSE_LINES)


class _FakeAsyncContent:
    def __aiter__(self):
        async def _gen():
            for line in SSE_LINES:
                yield line

        return _gen()


class _FakeAioResponse:
    status = HTTPStatus.OK
    content_type = "text/event-stream"
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self):
        self.content = _FakeAsyncContent()


def _assert_unified(rsp):
    assert rsp.status_code == HTTPStatus.INTERNAL_SERVER_ERROR, (
        "an unparsable server body must not be reported as a client error"
    )
    assert rsp.code == INTERNAL_ERROR.error_code
    assert MALFORMED in rsp.message, "the raw payload must reach the caller"
    assert "response data" not in rsp.message


def test_sync_stream_parse_failure():
    fake_self = SimpleNamespace(
        stream=True,
        encryption=None,
        flattened_output=False,
    )
    rsps = list(HttpRequest._handle_response(fake_self, _FakeSyncResponse()))
    assert len(rsps) == 1
    _assert_unified(rsps[0])


def test_utils_stream_parse_failure():
    """The baseline the other three paths are aligned to."""
    _, rsp = next(
        _handle_http_stream_response(_FakeSyncResponse()),
    )
    _assert_unified(rsp)


@pytest.mark.asyncio
async def test_aio_stream_parse_failure_in_http_request():
    fake_self = SimpleNamespace(stream=True, encryption=None)
    rsps = [
        rsp
        async for rsp in HttpRequest._handle_aio_response(
            fake_self,
            _FakeAioResponse(),
        )
    ]
    assert len(rsps) == 1
    _assert_unified(rsps[0])


@pytest.mark.asyncio
async def test_aio_stream_parse_failure_in_aiohttp_request():
    fake_self = SimpleNamespace(
        stream=True,
        # _handle_stream is unbound here and never touches self.
        _handle_stream=lambda rsp: AioHttpRequest._handle_stream(None, rsp),
    )
    rsps = [
        rsp
        async for rsp in AioHttpRequest._handle_response(
            fake_self,
            _FakeAioResponse(),
        )
    ]
    assert len(rsps) == 1
    _assert_unified(rsps[0])
