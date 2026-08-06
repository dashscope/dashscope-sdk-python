# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

"""Unit tests for UTF-8 encoding in HTTP request bodies.

Verifies that:
1. Content-Type / Accept headers include charset=utf-8
2. JSON body is serialized with ensure_ascii=False and encoded as UTF-8
3. Body bytes are smaller than ASCII-escaped equivalent
4. Both sync (HttpRequest) and async (AioHttpRequest) paths behave identically
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from dashscope.api_entities.http_request import HttpRequest
from dashscope.api_entities.aiohttp_request import AioHttpRequest
from dashscope.common.constants import HTTPMethod


NON_ASCII_PAYLOAD = {
    "model": "qwen-turbo",
    "input": {
        "messages": [
            {"role": "user", "content": "你好世界"},
            {"role": "user", "content": "Оцени идею"},
            {"role": "user", "content": "こんにちは"},
            {"role": "user", "content": "emoji: 🎉🚀💡"},
        ],
    },
    "parameters": {"result_format": "message"},
}

EXPECTED_BODY = json.dumps(NON_ASCII_PAYLOAD, ensure_ascii=False).encode(
    "utf-8",
)


def _make_mock_data(payload):
    mock_data = MagicMock()
    mock_data.get_http_payload.return_value = (False, None, payload)
    mock_data.get_aiohttp_payload.return_value = (False, payload)
    mock_data.parameters = {}
    return mock_data


def _make_mock_sync_session():
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.iter_content.return_value = [b'{"output":{"text":"ok"}}']
    mock_session.post.return_value = mock_response
    return mock_session


class _MockAioResponse:
    status = 200
    content_type = "application/json"
    headers = {"Content-Type": "application/json; charset=utf-8"}

    async def json(self):
        return {"output": {"text": "ok"}, "request_id": "test-123"}

    async def read(self):
        return b'{"output":{"text":"ok"},"request_id":"test-123"}'

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _MockAioSession:
    def __init__(self):
        self.captured = {}

    # pylint: disable=unused-argument
    async def request(
        self,
        method,
        url,
        data=None,
        headers=None,
        timeout=None,
    ):
        self.captured["data"] = data
        self.captured["headers"] = headers
        return _MockAioResponse()

    # pylint: disable=unused-argument
    async def post(
        self,
        url,
        data=None,
        headers=None,
        timeout=None,
    ):
        self.captured["data"] = data
        self.captured["headers"] = headers
        return _MockAioResponse()

    async def close(self):
        pass


# ---- Header tests ----


class TestHttpRequestUtf8Headers:
    def test_content_type_has_charset(self):
        req = HttpRequest(
            url="https://example.com/api/v1/test",
            api_key="test-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )
        assert req.headers["Content-Type"] == "application/json; charset=utf-8"

    def test_accept_has_charset(self):
        req = HttpRequest(
            url="https://example.com/api/v1/test",
            api_key="test-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )
        assert req.headers["Accept"] == "application/json; charset=utf-8"

    def test_stream_overrides_accept(self):
        req = HttpRequest(
            url="https://example.com/api/v1/test",
            api_key="test-key",
            http_method=HTTPMethod.POST,
            stream=True,
        )
        assert "text/event-stream" in req.headers["Accept"]


class TestAioHttpRequestUtf8Headers:
    def test_content_type_has_charset(self):
        req = AioHttpRequest(
            url="https://example.com/api/v1/test",
            api_key="test-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )
        assert req.headers["Content-Type"] == "application/json; charset=utf-8"

    def test_accept_has_charset(self):
        req = AioHttpRequest(
            url="https://example.com/api/v1/test",
            api_key="test-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )
        assert req.headers["Accept"] == "application/json; charset=utf-8"


# ---- Sync body tests ----


class TestHttpRequestUtf8Body:
    # pylint: disable=protected-access

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_body_is_utf8_bytes(self, mock_session_cls):
        mock_session = _make_mock_sync_session()
        mock_session_cls.return_value = mock_session

        req = HttpRequest(
            url="https://example.com/api/v1/test",
            api_key="test-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )
        req.data = _make_mock_data(NON_ASCII_PAYLOAD)

        try:
            list(req._handle_request())
        except Exception:
            pass

        mock_session.post.assert_called_once()
        sent_body = mock_session.post.call_args.kwargs.get("data")

        assert isinstance(sent_body, bytes), "Body must be bytes, not str"
        assert sent_body == EXPECTED_BODY

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_no_unicode_escapes_in_body(self, mock_session_cls):
        mock_session = _make_mock_sync_session()
        mock_session_cls.return_value = mock_session

        req = HttpRequest(
            url="https://example.com/api/v1/test",
            api_key="test-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )
        req.data = _make_mock_data(NON_ASCII_PAYLOAD)

        try:
            list(req._handle_request())
        except Exception:
            pass

        sent_body = mock_session.post.call_args.kwargs.get("data")
        body_str = sent_body.decode("utf-8")

        assert "\\u" not in body_str, "Body must not contain \\uXXXX escapes"
        assert "你好世界" in body_str
        assert "Оцени идею" in body_str
        assert "こんにちは" in body_str

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_body_is_smaller_than_ascii_escaped(self, mock_session_cls):
        mock_session = _make_mock_sync_session()
        mock_session_cls.return_value = mock_session

        req = HttpRequest(
            url="https://example.com/api/v1/test",
            api_key="test-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )
        req.data = _make_mock_data(NON_ASCII_PAYLOAD)

        try:
            list(req._handle_request())
        except Exception:
            pass

        sent_body = mock_session.post.call_args.kwargs.get("data")
        ascii_escaped_size = len(
            json.dumps(NON_ASCII_PAYLOAD, ensure_ascii=True).encode("utf-8"),
        )
        assert len(sent_body) < ascii_escaped_size


# ---- Async body tests ----


@pytest.mark.asyncio
class TestAioHttpRequestUtf8Body:
    # pylint: disable=protected-access

    async def test_body_is_utf8_bytes(self):
        session = _MockAioSession()
        req = AioHttpRequest(
            url="https://example.com/api/v1/test",
            api_key="test-key",
            http_method=HTTPMethod.POST,
            stream=False,
            session=session,
        )
        req.data = _make_mock_data(NON_ASCII_PAYLOAD)

        async for _ in req._handle_request():
            pass

        sent_body = session.captured.get("data")
        assert isinstance(sent_body, bytes), "Body must be bytes, not str"
        assert sent_body == EXPECTED_BODY

    async def test_no_unicode_escapes_in_body(self):
        session = _MockAioSession()
        req = AioHttpRequest(
            url="https://example.com/api/v1/test",
            api_key="test-key",
            http_method=HTTPMethod.POST,
            stream=False,
            session=session,
        )
        req.data = _make_mock_data(NON_ASCII_PAYLOAD)

        async for _ in req._handle_request():
            pass

        sent_body = session.captured.get("data")
        body_str = sent_body.decode("utf-8")

        assert "\\u" not in body_str, "Body must not contain \\uXXXX escapes"
        assert "你好世界" in body_str
        assert "Оцени идею" in body_str

    async def test_body_is_smaller_than_ascii_escaped(self):
        session = _MockAioSession()
        req = AioHttpRequest(
            url="https://example.com/api/v1/test",
            api_key="test-key",
            http_method=HTTPMethod.POST,
            stream=False,
            session=session,
        )
        req.data = _make_mock_data(NON_ASCII_PAYLOAD)

        async for _ in req._handle_request():
            pass

        sent_body = session.captured.get("data")
        ascii_escaped_size = len(
            json.dumps(NON_ASCII_PAYLOAD, ensure_ascii=True).encode("utf-8"),
        )
        assert len(sent_body) < ascii_escaped_size


# ---- WebSocket tests ----


class TestWebSocketRequestUtf8:
    # pylint: disable=protected-access

    def test_build_up_message_no_unicode_escapes(self):
        from dashscope.api_entities.websocket_request import WebSocketRequest

        req = WebSocketRequest(
            url="wss://example.com/api-ws/v1/test",
            api_key="test-key",
            stream=False,
        )
        headers = {"task_id": "test-123", "action": "start"}
        payload = {
            "input": {"messages": [{"role": "user", "content": "你好世界"}]},
        }

        message = req._build_up_message(headers, payload)

        assert isinstance(message, str), "Message must be str"
        assert "\\u" not in message, "Message must not contain \\uXXXX escapes"
        assert "你好世界" in message

    def test_build_up_message_smaller_than_ascii_escaped(self):
        from dashscope.api_entities.websocket_request import WebSocketRequest

        req = WebSocketRequest(
            url="wss://example.com/api-ws/v1/test",
            api_key="test-key",
            stream=False,
        )
        headers = {"task_id": "test-123", "action": "start"}
        payload = NON_ASCII_PAYLOAD

        message = req._build_up_message(headers, payload)
        message_bytes = message.encode("utf-8")

        ascii_escaped = json.dumps(
            {"header": headers, "payload": payload},
            ensure_ascii=True,
        ).encode("utf-8")

        assert len(message_bytes) < len(ascii_escaped)
