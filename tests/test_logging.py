# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for dashscope 2.0 logging (M1 task 1.4)."""

# pytest fixture injection plus intentional access to private state.
# pylint: disable=redefined-outer-name,unused-argument,protected-access

from __future__ import annotations

import io
import json
import logging

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope import _logging as logging_mod
from dashscope._logging import (
    REDACTED,
    SensitiveHeadersFilter,
    get_logger,
    redact_headers,
    redact_text,
    request_id_from_headers,
)
from dashscope.errors import (
    APIConnectionError,
    DashScopeError,
    InternalServerError,
    check_response,
    response_json,
)
from dashscope.transport import AsyncTransport, SyncTransport
from dashscope.transport.sse import AsyncStream, Stream
from dashscope.types.common import ResponseBase, response_body, with_headers


@pytest.fixture
def reset_logging(monkeypatch):
    root = logging.getLogger(logging_mod.ROOT_NAME)
    monkeypatch.setattr(logging_mod, "_configured", False)
    yield
    root.filters.clear()
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    logging_mod._configured = False


def test_redact_headers():
    out = redact_headers(
        {
            "Authorization": "Bearer sk-abcdef1234567890",
            "X-DashScope-WorkSpace": "ws-1",
            "Cookie": "session=abc",
        },
    )
    assert out["Authorization"] == REDACTED
    assert out["Cookie"] == REDACTED
    assert out["X-DashScope-WorkSpace"] == "ws-1"
    assert redact_headers(None) is None


def test_redact_text():
    text = (
        "auth: Bearer sk-abcdef1234567890 key=sk-abcdef1234567890"
        " keep=sk-short"
    )
    out = redact_text(text)
    assert "sk-abcdef1234567890" not in out
    assert "Bearer " + REDACTED in out
    assert "keep=sk-short" in out


def test_filter_redacts_record(reset_logging):
    f = SensitiveHeadersFilter()
    record = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="header authorization: Bearer sk-abcdef1234567890",
        args=("Bearer sk-abcdef1234567890",),
        exc_info=None,
    )
    record.headers = {"Authorization": "Bearer sk-abcdef1234567890"}
    assert f.filter(record) is True
    assert "sk-abcdef1234567890" not in record.msg
    assert "sk-abcdef1234567890" not in record.args[0]
    assert record.headers["Authorization"] == REDACTED


def test_env_enables_debug_logging(reset_logging, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_LOG", "debug")
    logger = get_logger("transport")
    root = logging.getLogger(logging_mod.ROOT_NAME)
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(
        root.handlers[0].filters[0],
        SensitiveHeadersFilter,
    )
    assert isinstance(logger.filters[0], SensitiveHeadersFilter)
    assert logger.name == f"{logging_mod.ROOT_NAME}.transport"


def test_env_invalid_level_no_handler(reset_logging, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_LOG", "verbose")
    logger = get_logger("transport")
    root = logging.getLogger(logging_mod.ROOT_NAME)
    assert root.level == logging.NOTSET
    assert root.handlers == []
    assert root.filters == []
    assert isinstance(logger.filters[0], SensitiveHeadersFilter)


def test_configured_once(reset_logging, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_LOG", "info")
    get_logger("a")
    get_logger("a")
    get_logger("b")
    root = logging.getLogger(logging_mod.ROOT_NAME)
    assert len(root.handlers) == 1
    for name in ("a", "b"):
        child = logging.getLogger(f"{logging_mod.ROOT_NAME}.{name}")
        assert len(child.filters) == 1


def test_configuration_does_not_mutate_legacy_root_logger(
    reset_logging,
    monkeypatch,
):
    legacy_root = logging.getLogger("dashscope")
    level = legacy_root.level
    handlers = list(legacy_root.handlers)
    filters = list(legacy_root.filters)

    monkeypatch.setenv("DASHSCOPE_LOG", "debug")
    get_logger("transport")

    assert logging_mod.ROOT_NAME != "dashscope"
    assert legacy_root.level == level
    assert legacy_root.handlers == handlers
    assert legacy_root.filters == filters


def test_transport_logs_retry_and_request_id(
    reset_logging,
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(
        "dashscope.transport.http._sleep",
        lambda s: None,
    )
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0", "x-request-id": "req-42"},
            )
        return httpx.Response(200, headers={"x-request-id": "req-43"})

    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(
        transport=mock,
        base_url="https://api.test.local",
    )
    t = SyncTransport(
        base_url="https://api.test.local",
        http_client=http_client,
    )
    with caplog.at_level(logging.DEBUG, logger=logging_mod.ROOT_NAME):
        resp = t.request("POST", "/chat")
    assert resp.status_code == 200
    messages = [r.getMessage() for r in caplog.records]
    assert any("retrying" in m and "req-42" in m for m in messages)
    assert any("request_id=req-43" in m for m in messages)
    assert not any("sk-" in m and len(m) > 20 for m in messages)


def test_child_logger_output_redacted_via_sdk_handler(
    reset_logging,
    monkeypatch,
):
    monkeypatch.setenv("DASHSCOPE_LOG", "debug")
    logger = get_logger("transport")
    root = logging.getLogger(logging_mod.ROOT_NAME)
    stream = io.StringIO()
    root.handlers[0].stream = stream
    logger.info("token: Bearer sk-abcdef1234567890")
    out = stream.getvalue()
    assert "sk-abcdef1234567890" not in out
    assert "Bearer " + REDACTED in out


def test_child_logger_output_redacted_via_user_handler(
    reset_logging,
    caplog,
):
    # No DASHSCOPE_LOG: records propagate to the user's own handlers
    # (e.g. logging.basicConfig) and must still be redacted.
    logger = get_logger("transport")
    with caplog.at_level(logging.DEBUG, logger=logging_mod.ROOT_NAME):
        logger.info("token: Bearer sk-abcdef1234567890")
    messages = [r.getMessage() for r in caplog.records]
    assert any("Bearer " + REDACTED in m for m in messages)
    assert not any("sk-abcdef1234567890" in m for m in messages)


@pytest.mark.parametrize(
    "name",
    ["X-Request-ID", "X-DashScope-RequestId", "X-DashScope-Request-ID"],
)
def test_request_id_header_aliases(name):
    assert request_id_from_headers({name: "req-alias"}) == "req-alias"
    assert request_id_from_headers({name: [None, "", "req-list"]}) == (
        "req-list"
    )


def test_request_id_header_skips_empty_values():
    assert (
        request_id_from_headers(
            {"x-request-id": " ", "X-DashScope-RequestId": "req-fallback"},
        )
        == "req-fallback"
    )
    assert request_id_from_headers(None) is None
    assert request_id_from_headers({"x-request-id": ""}) is None


@pytest.mark.parametrize("payload", ["not-json", "[]"])
@pytest.mark.parametrize("request_id", [None, "req-http-parse"])
def test_http_parse_failure_logs_request_id(
    reset_logging,
    caplog,
    payload,
    request_id,
):
    response = httpx.Response(
        200,
        text=payload,
        headers={"X-DashScope-RequestId": request_id} if request_id else {},
    )
    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        with pytest.raises(InternalServerError) as caught:
            response_json(response)
    assert caught.value.request_id == request_id
    assert any(
        "Failed to parse HTTP response" in record.message
        and f"request_id={request_id or 'unknown'}" in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize("status", [200, 400])
@pytest.mark.parametrize("raise_on_error", [True, False])
@pytest.mark.parametrize("parse", [check_response, response_body])
def test_http_service_error_logs_body_request_id(
    reset_logging,
    caplog,
    status,
    raise_on_error,
    parse,
):
    response = httpx.Response(
        status,
        json={"code": "InvalidParameter", "request_id": "req-body"},
        headers={"X-Request-ID": "req-header"},
    )
    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        if raise_on_error:
            with pytest.raises(DashScopeError) as caught:
                parse(response, raise_on_error=True)
            assert caught.value.request_id == "req-body"
        else:
            parse(response, raise_on_error=False)
    assert len(caplog.records) == 1
    assert "request_id=req-body" in caplog.records[0].message
    assert "req-header" not in caplog.records[0].message


@pytest.mark.parametrize("body_id", [None, "req-body"])
def test_response_header_request_id_backfill(body_id):
    response = httpx.Response(
        200,
        json={},
        headers={"X-DashScope-Request-ID": "req-header"},
    )
    model = with_headers(ResponseBase(request_id=body_id), response)
    assert model.request_id == (body_id or "req-header")
    assert response_body(response)["request_id"] == "req-header"
    assert with_headers(ResponseBase(), httpx.Response(200)).request_id is None


def _logging_sse_response(data):
    return httpx.Response(
        200,
        text=f"data: {data}\n\n",
        headers={
            "Content-Type": "text/event-stream",
            "X-DashScope-RequestId": "req-sse-header",
        },
    )


@pytest.mark.parametrize("data", ["not-json", "[]"])
def test_sse_parse_failure_logs_request_id(reset_logging, caplog, data):
    response = _logging_sse_response(data)
    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        with pytest.raises(DashScopeError) as caught:
            list(Stream(response, lambda body: body))
    assert caught.value.request_id == "req-sse-header"
    assert response.is_closed
    assert len(caplog.records) == 1
    assert "request_id=req-sse-header" in caplog.records[0].message


@pytest.mark.asyncio
async def test_async_sse_parse_failure_logs_request_id(reset_logging, caplog):
    response = _logging_sse_response("not-json")
    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        with pytest.raises(DashScopeError) as caught:
            async for _ in AsyncStream(response, lambda body: body):
                pass
    assert caught.value.request_id == "req-sse-header"
    assert response.is_closed
    assert len(caplog.records) == 1
    assert "request_id=req-sse-header" in caplog.records[0].message


def test_sse_service_error_logs_body_request_id(reset_logging, caplog):
    response = _logging_sse_response(
        json.dumps({"code": "Throttling", "request_id": "req-sse-body"}),
    )
    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        with pytest.raises(DashScopeError) as caught:
            list(Stream(response, lambda body: body))
    assert caught.value.request_id == "req-sse-body"
    assert len(caplog.records) == 1
    assert "request_id=req-sse-body" in caplog.records[0].message
    assert response.is_closed


def test_sse_result_parse_failure_logs_body_request_id(reset_logging, caplog):
    response = _logging_sse_response(
        json.dumps({"request_id": "req-result-body"}),
    )

    def reject_result(body):
        raise ValueError("result has an invalid shape")

    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        with pytest.raises(ValueError, match="invalid shape"):
            list(Stream(response, reject_result))
    assert len(caplog.records) == 1
    assert "request_id=req-result-body" in caplog.records[0].message
    assert response.is_closed


def test_pre_response_failure_log_has_no_request_id(reset_logging, caplog):
    def fail(request):
        raise httpx.ConnectError("connection refused", request=request)

    with httpx.Client(transport=httpx.MockTransport(fail)) as http_client:
        transport = SyncTransport(
            base_url="https://api.test.local",
            http_client=http_client,
            max_retries=0,
        )
        with caplog.at_level(logging.DEBUG, logger=logging_mod.ROOT_NAME):
            with pytest.raises(APIConnectionError) as caught:
                transport.request("POST", "/chat")
    assert caught.value.request_id is None
    assert len(caplog.records) == 1
    assert "request_id" not in caplog.records[0].message


def test_static_sse_parser_does_not_log_request_id(reset_logging, caplog):
    from dashscope.transport.sse import parse_event_data

    with caplog.at_level(logging.DEBUG, logger=logging_mod.ROOT_NAME):
        with pytest.raises(DashScopeError):
            parse_event_data("not-json")
    assert caplog.records == []


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_sse_later_failure_retains_stream_request_id(
    reset_logging,
    caplog,
    async_mode,
):
    response = _logging_sse_response('{"request_id": "req-first"}')
    response = httpx.Response(
        200,
        text=(
            'data: {"request_id": "req-first"}\n\n'
            'data: {"output": {}}\n\n'
            "data: not-json\n\n"
        ),
        headers=response.headers,
    )
    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        with pytest.raises(DashScopeError) as caught:
            if async_mode:
                async for _ in AsyncStream(response, lambda body: body):
                    pass
            else:
                list(Stream(response, lambda body: body))
    assert caught.value.request_id == "req-first"
    assert len(caplog.records) == 1
    assert "request_id=req-first" in caplog.records[0].message
    assert response.is_closed


def test_job_event_log_uses_header_without_changing_error_contract(
    reset_logging,
    caplog,
):
    from dashscope.resources._sse import _stream_error

    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        error = _stream_error(
            None,
            '{"code": "Throttling", "output": {"request_id": "inner-id"}}',
            {"X-DashScope-Request-ID": "req-job-header"},
        )
    assert error.request_id is None
    assert len(caplog.records) == 1
    assert "request_id=req-job-header" in caplog.records[0].message


class _BrokenSyncResponse(httpx.SyncByteStream):
    def __iter__(self):
        yield b""
        raise httpx.ReadError("read failed")


class _BrokenAsyncResponse(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b""
        raise httpx.ReadError("read failed")


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_response_read_failure_logs_request_id(
    reset_logging,
    caplog,
    async_mode,
    streaming,
):
    response = httpx.Response(
        400,
        stream=_BrokenAsyncResponse() if async_mode else _BrokenSyncResponse(),
        headers={"X-DashScope-Request-ID": "req-read-error"},
    )
    mock = httpx.MockTransport(lambda request: response)
    with caplog.at_level(logging.WARNING, logger=logging_mod.ROOT_NAME):
        with pytest.raises(APIConnectionError) as caught:
            if async_mode:
                async with httpx.AsyncClient(transport=mock) as http_client:
                    transport = AsyncTransport(
                        base_url="https://api.test.local",
                        http_client=http_client,
                        max_retries=0,
                    )
                    request = (
                        transport.open_stream
                        if streaming
                        else transport.request
                    )
                    await request("POST", "/chat")
            else:
                with httpx.Client(transport=mock) as http_client:
                    transport = SyncTransport(
                        base_url="https://api.test.local",
                        http_client=http_client,
                        max_retries=0,
                    )
                    request = (
                        transport.open_stream
                        if streaming
                        else transport.request
                    )
                    request("POST", "/chat")
    assert caught.value.request_id == "req-read-error"
    assert response.is_closed
    assert len(caplog.records) == 1
    assert "request_id=req-read-error" in caplog.records[0].message


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_sse_protocol_failure_logs_request_id(
    reset_logging,
    caplog,
    async_mode,
):
    from httpx_sse import SSEError

    response = httpx.Response(
        200,
        text="not an event stream",
        headers={
            "Content-Type": "application/json",
            "X-DashScope-Request-ID": "req-sse-protocol",
        },
    )
    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        with pytest.raises(SSEError):
            if async_mode:
                async for _ in AsyncStream(response, lambda body: body):
                    pass
            else:
                list(Stream(response, lambda body: body))
    assert response.is_closed
    assert len(caplog.records) == 1
    assert "request_id=req-sse-protocol" in caplog.records[0].message


@pytest.mark.parametrize("resource", ["fine_tunes", "deployments"])
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_nonraising_job_event_keeps_malformed_code(
    reset_logging,
    caplog,
    resource,
    async_mode,
):
    body = {"code": {"unexpected": "object"}}
    response = httpx.Response(
        200,
        text="event:error\ndata: " + json.dumps(body) + "\n\n",
        headers={
            "Content-Type": "text/event-stream",
            "X-DashScope-RequestId": "req-nonraising-job",
        },
    )
    mock = httpx.MockTransport(lambda request: response)
    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        if async_mode:
            async with httpx.AsyncClient(transport=mock) as http_client:
                async with AsyncClient(
                    base_url="https://api.test.local",
                    http_client=http_client,
                    raise_on_error=False,
                ) as client:
                    events = getattr(client, resource).stream_events("job-1")
                    result = [event async for event in events]
        else:
            with httpx.Client(transport=mock) as http_client:
                with Client(
                    base_url="https://api.test.local",
                    http_client=http_client,
                    raise_on_error=False,
                ) as client:
                    result = list(
                        getattr(client, resource).stream_events("job-1"),
                    )
    assert len(result) == 1
    assert result[0].is_error
    assert result[0].output == body
    assert response.is_closed
    assert len(caplog.records) == 1
    assert "request_id=req-nonraising-job" in caplog.records[0].message


@pytest.mark.parametrize("parse", [check_response, response_body])
@pytest.mark.parametrize("raise_on_error", [False, True])
@pytest.mark.parametrize("header_id", [None, "safe-header-id"])
def test_http_error_log_never_stringifies_object_request_id(
    reset_logging,
    caplog,
    parse,
    raise_on_error,
    header_id,
):
    invalid_id = {"credential": "private-payload"}
    response = httpx.Response(
        400,
        json={"code": "InvalidParameter", "request_id": invalid_id},
        headers={"X-DashScope-RequestId": header_id} if header_id else {},
    )
    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        if raise_on_error:
            with pytest.raises(DashScopeError) as caught:
                parse(response)
            assert caught.value.request_id == invalid_id
        else:
            parse(response, raise_on_error=False)
    (record,) = caplog.records
    assert f"request_id={header_id or 'unknown'}" in record.message
    assert "private-payload" not in record.message
    assert "credential" not in record.message


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("fault", ["business", "parse", "result"])
@pytest.mark.asyncio
async def test_sse_invalid_id_preserves_safe_log_context(
    reset_logging,
    caplog,
    async_mode,
    fault,
):
    invalid_id = {"credential": "private-payload"}
    body = {"request_id": invalid_id}
    if fault == "business":
        body["code"] = "InvalidParameter"
    data = (
        'data: {"request_id": "first-id"}\n\n'
        + "data: "
        + json.dumps(body)
        + "\n\n"
    )
    if fault == "parse":
        data += "data: not-json\n\n"
    response = httpx.Response(
        200,
        text=data,
        headers={
            "Content-Type": "text/event-stream",
            "X-Request-ID": "header-id",
        },
    )

    def parse(body):
        if fault == "result" and body.get("request_id") == invalid_id:
            raise ValueError("rejected")
        return body

    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        with pytest.raises(
            ValueError if fault == "result" else DashScopeError,
        ):
            if async_mode:
                async for _ in AsyncStream(response, parse):
                    pass
            else:
                list(Stream(response, parse))
    (record,) = caplog.records
    assert "request_id=first-id" in record.message
    assert "private-payload" not in record.message
    assert "credential" not in record.message
    assert response.is_closed


@pytest.mark.parametrize("resource", ["fine_tunes", "deployments"])
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_job_event_status_parse_failure_logs_and_closes(
    reset_logging,
    caplog,
    resource,
    async_mode,
):
    response = httpx.Response(
        200,
        text="status: private-payload\n",
        headers={
            "Content-Type": "text/event-stream",
            "X-DashScope-RequestId": "job-header-id",
        },
    )
    mock = httpx.MockTransport(lambda request: response)
    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        with pytest.raises(ValueError):
            if async_mode:
                async with httpx.AsyncClient(transport=mock) as http_client:
                    async with AsyncClient(
                        base_url="https://api.test.local",
                        http_client=http_client,
                    ) as client:
                        events = getattr(client, resource).stream_events("job")
                        async for _ in events:
                            pass
            else:
                with httpx.Client(transport=mock) as http_client:
                    with Client(
                        base_url="https://api.test.local",
                        http_client=http_client,
                    ) as client:
                        list(getattr(client, resource).stream_events("job"))
    (record,) = caplog.records
    assert "request_id=job-header-id" in record.message
    assert "private-payload" not in record.message
    assert response.is_closed


def test_job_business_log_never_stringifies_object_request_id(
    reset_logging,
    caplog,
):
    from dashscope.resources._sse import _log_stream_error

    body = {"code": "Throttling", "request_id": {"secret": "private-payload"}}
    with caplog.at_level(logging.ERROR, logger=logging_mod.ROOT_NAME):
        _log_stream_error(json.dumps(body), {"X-Request-ID": "header-id"})
    (record,) = caplog.records
    assert "request_id=header-id" in record.message
    assert "private-payload" not in record.message
