# -*- coding: utf-8 -*-
"""Error paths must keep the detail that explains the failure.

Four places reduced a specific exception to a generic message: the agentstudio
transports wrapped httpx timeouts, whose ``str()`` is usually empty;
``iter_over_async`` logged the real exception but reported only the registry
text; and the WebSocket connect/handshake handlers dropped the host, port and
server reason. The 503 detection also matched a bare "503" anywhere in the
message, which fires on addresses such as ``host:5030``.
"""
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from dashscope.agentstudio import exceptions
from dashscope.agentstudio.transport import (
    AsyncTransport,
    SyncTransport,
    _describe_exception,
)
from dashscope.api_entities.websocket_request import (
    _internal_error_message,
    _is_service_unavailable,
)
from dashscope.common.error_registry import INTERNAL_ERROR
from dashscope.common.utils import iter_over_async


class TestDescribeException:
    def test_messageless_exception_falls_back_to_type_name(self):
        assert _describe_exception(httpx.ReadTimeout("")) == "ReadTimeout"

    def test_whitespace_only_message_is_treated_as_empty(self):
        assert _describe_exception(httpx.ReadTimeout("   ")) == "ReadTimeout"

    def test_message_is_prefixed_with_type_name(self):
        assert _describe_exception(ValueError("bad")) == "ValueError: bad"


def _transport_kwargs(client):
    return {
        "base_url": "https://example.com",
        "api_key": "key",
        "workspace": None,
        "uid": None,
        "user_agent": "ua",
        "timeout": 1.0,
        "max_retries": 0,
        "http_client": client,
    }


class TestSyncTransportKeepsDetail:
    def _transport(self, side_effect):
        client = MagicMock()
        client.send.side_effect = side_effect
        return SyncTransport(**_transport_kwargs(client))

    def test_timeout_message_is_not_empty(self):
        transport = self._transport(httpx.ReadTimeout(""))

        with pytest.raises(exceptions.APITimeoutError) as exc_info:
            transport.request("GET", "/v1/agents")

        assert exc_info.value.message.strip()
        assert "ReadTimeout" in exc_info.value.message

    def test_timeout_keeps_its_cause(self):
        original = httpx.ReadTimeout("")
        transport = self._transport(original)

        with pytest.raises(exceptions.APITimeoutError) as exc_info:
            transport.request("GET", "/v1/agents")

        assert exc_info.value.__cause__ is original

    def test_connection_error_message_is_not_empty(self):
        transport = self._transport(httpx.ConnectError(""))

        with pytest.raises(exceptions.APIConnectionError) as exc_info:
            transport.request("GET", "/v1/agents")

        assert "ConnectError" in exc_info.value.message


class TestAsyncTransportKeepsDetail:
    def _transport(self, side_effect):
        client = AsyncMock()
        client.build_request = MagicMock()
        client.send.side_effect = side_effect
        return AsyncTransport(**_transport_kwargs(client))

    @pytest.mark.asyncio
    async def test_timeout_message_is_not_empty(self):
        transport = self._transport(httpx.ReadTimeout(""))

        with pytest.raises(exceptions.APITimeoutError) as exc_info:
            await transport.request("GET", "/v1/agents")

        assert "ReadTimeout" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_connection_error_message_is_not_empty(self):
        transport = self._transport(httpx.ConnectError(""))

        with pytest.raises(exceptions.APIConnectionError) as exc_info:
            await transport.request("GET", "/v1/agents")

        assert "ConnectError" in exc_info.value.message


class TestIterOverAsyncReportsTheFailure:
    def test_message_names_the_underlying_exception(self):
        async def failing():
            yield 1
            raise RuntimeError("generator exploded")

        yielded = list(iter_over_async(failing()))

        assert yielded[0] == 1
        response = yielded[-1]
        assert response.status_code == INTERNAL_ERROR.status_code
        assert response.code == INTERNAL_ERROR.error_code
        assert "RuntimeError" in response.message
        assert "generator exploded" in response.message

    def test_registry_text_is_still_present(self):
        async def failing():
            raise ValueError("bad")
            yield  # pragma: no cover

        response = list(iter_over_async(failing()))[-1]

        assert response.message.startswith(INTERNAL_ERROR.format_msg())


class TestInternalErrorMessage:
    def test_includes_registry_text_and_exception_detail(self):
        message = _internal_error_message(OSError(22, "Invalid argument"))

        assert message.startswith(INTERNAL_ERROR.error_msg)
        assert "OSError" in message
        assert "Invalid argument" in message

    def test_explicit_detail_wins_over_str(self):
        message = _internal_error_message(
            OSError("ignored"),
            "handshake rejected: model not found",
        )

        assert "handshake rejected: model not found" in message
        assert "ignored" not in message

    def test_empty_detail_falls_back_to_the_exception(self):
        message = _internal_error_message(ValueError("real reason"), "")

        assert "real reason" in message


class TestServiceUnavailableDetection:
    def test_explicit_phrase_is_treated_as_503(self):
        assert _is_service_unavailable(OSError("Service Unavailable"))

    def test_port_containing_503_is_not_treated_as_503(self):
        assert not _is_service_unavailable(
            OSError("Cannot connect to host 10.50.30.1:5030"),
        )

    def test_plain_connection_refusal_is_not_treated_as_503(self):
        assert not _is_service_unavailable(
            OSError("Cannot connect to host dashscope.aliyuncs.com:443"),
        )
