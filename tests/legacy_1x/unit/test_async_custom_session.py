# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

"""
Unit tests for async HTTP custom session support.

Scope:
1. HttpRequest accepts a custom aiohttp.ClientSession parameter
2. Usage and resource management of a custom aio_session
3. Creation and cleanup of temporary aio_session instances
4. Session priority logic
5. Async session behavior across different scenarios

Note: none of the tests rely on a real API key.
"""

# pylint: disable=protected-access,unused-argument,unused-variable
# pylint: disable=broad-exception-raised

import ssl
from unittest.mock import patch, AsyncMock

import aiohttp
import certifi
import pytest

from dashscope.legacy_1x.api_entities.http_request import HttpRequest
from dashscope.legacy_1x.api_entities.api_request_data import ApiRequestData
from dashscope.legacy_1x.common.constants import ApiProtocol, HTTPMethod


class TestAsyncSessionBasics:
    """Basic tests for async sessions"""

    @pytest.mark.asyncio
    async def test_http_request_accepts_aio_session_parameter(self):
        """Test that HttpRequest accepts the aio_session parameter"""
        connector = aiohttp.TCPConnector()
        custom_session = aiohttp.ClientSession(connector=connector)

        try:
            http_request = HttpRequest(
                url="http://example.com/api",
                api_key="fake-api-key",
                http_method=HTTPMethod.POST,
                session=custom_session,
            )

            assert http_request._external_aio_session is custom_session
            assert http_request._external_aio_session is not None
        finally:
            await custom_session.close()

    @pytest.mark.asyncio
    async def test_http_request_without_aio_session_parameter(self):
        """Test HttpRequest without the aio_session parameter"""
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
        )

        assert http_request._external_aio_session is None

    @pytest.mark.asyncio
    async def test_aio_session_parameter_is_optional(self):
        """Test that the aio_session parameter is optional"""
        # Works without the aio_session parameter
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        assert http_request._external_aio_session is None
        assert http_request.url == "http://example.com/api"


class TestAsyncSessionUsage:
    """Tests for actual async session usage"""

    @pytest.mark.asyncio
    async def test_custom_aio_session_is_used_for_request(self):
        """Test that the custom aio_session is actually used for the request"""
        # Create a mock session
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Make request() return an awaitable
        async def mock_request(*_args, **_kwargs):
            return mock_response

        mock_session.request = mock_request

        # Create an HttpRequest with the custom session
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
            session=mock_session,
        )

        # Add request data
        request_data = ApiRequestData(
            model="test-model",
            task_group="test",
            task="test",
            function="test",
            input_data={"test": "data"},
            form=None,
            is_binary_input=False,
            api_protocol=ApiProtocol.HTTPS,
        )
        http_request.data = request_data

        # Execute the request
        async def mock_handle_response(_response):
            yield mock_response

        with patch.object(
            http_request,
            "_handle_aio_response",
            side_effect=mock_handle_response,
        ):
            _ = await http_request.aio_call()

        # Verify the custom session was not closed
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_shared_aio_session_is_used_when_no_custom_session(
        self,
    ):
        """Test that the shared session is used (and not closed) when no
        custom aio_session is given"""
        # Create a mock shared session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.request.return_value = mock_response

        # Create an HttpRequest without aio_session
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        # Add request data
        request_data = ApiRequestData(
            model="test-model",
            task_group="test",
            task="test",
            function="test",
            input_data={"test": "data"},
            form=None,
            is_binary_input=False,
            api_protocol=ApiProtocol.HTTPS,
        )
        http_request.data = request_data

        # Execute the request
        async def mock_handle_response(_response):
            yield mock_response

        with patch(
            "dashscope.legacy_1x.api_entities.http_request.get_shared_aio_session",
            return_value=mock_session,
        ):
            with patch.object(
                http_request,
                "_handle_aio_response",
                side_effect=mock_handle_response,
            ):
                _ = await http_request.aio_call()

        # The shared session must not be closed (lifecycle is managed
        # by the aio_session module)
        mock_session.close.assert_not_called()


class TestAsyncSessionResourceManagement:
    """Tests for async session resource management"""

    @pytest.mark.asyncio
    async def test_custom_aio_session_not_closed_by_http_request(self):
        """Test that HttpRequest does not close the custom aio_session"""
        custom_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Make request() return an awaitable
        async def mock_request(*_args, **_kwargs):
            return mock_response

        custom_session.request = mock_request

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
            session=custom_session,
        )

        request_data = ApiRequestData(
            model="test-model",
            task_group="test",
            task="test",
            function="test",
            input_data={"test": "data"},
            form=None,
            is_binary_input=False,
            api_protocol=ApiProtocol.HTTPS,
        )
        http_request.data = request_data

        async def mock_handle_response(_response):
            yield mock_response

        with patch.object(
            http_request,
            "_handle_aio_response",
            side_effect=mock_handle_response,
        ):
            _ = await http_request.aio_call()

        # Verify the custom aio_session was not closed
        custom_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_shared_aio_session_not_closed_on_success(self):
        """Test that the shared aio_session is not closed on success
        (lifecycle managed by the module)"""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.request.return_value = mock_response

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        request_data = ApiRequestData(
            model="test-model",
            task_group="test",
            task="test",
            function="test",
            input_data={"test": "data"},
            form=None,
            is_binary_input=False,
            api_protocol=ApiProtocol.HTTPS,
        )
        http_request.data = request_data

        async def mock_handle_response(_response):
            yield mock_response

        with patch(
            "dashscope.legacy_1x.api_entities.http_request.get_shared_aio_session",
            return_value=mock_session,
        ):
            with patch.object(
                http_request,
                "_handle_aio_response",
                side_effect=mock_handle_response,
            ):
                _ = await http_request.aio_call()

        # The shared session must not be closed
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_shared_aio_session_not_closed_on_exception(self):
        """Test that the shared aio_session is not closed on exceptions either"""
        mock_session = AsyncMock()

        # Make request() raise an exception
        async def mock_request(*_args, **_kwargs):
            raise Exception("Network error")

        mock_session.request = mock_request

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        request_data = ApiRequestData(
            model="test-model",
            task_group="test",
            task="test",
            function="test",
            input_data={"test": "data"},
            form=None,
            is_binary_input=False,
            api_protocol=ApiProtocol.HTTPS,
        )
        http_request.data = request_data

        # Executing the request should raise
        with patch(
            "dashscope.legacy_1x.api_entities.http_request.get_shared_aio_session",
            return_value=mock_session,
        ):
            with pytest.raises(Exception, match="Network error"):
                _ = await http_request.aio_call()

        # The shared session still must not be closed (lifecycle managed by the module)
        mock_session.close.assert_not_called()


class TestAsyncSessionWithCustomConfiguration:
    """Tests for async sessions with custom configuration"""

    @pytest.mark.asyncio
    async def test_custom_aio_session_with_connector(self):
        """Test an aio_session with a custom connector"""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ssl=ssl.create_default_context(cafile=certifi.where()),
        )
        custom_session = aiohttp.ClientSession(connector=connector)

        try:
            http_request = HttpRequest(
                url="http://example.com/api",
                api_key="fake-api-key",
                http_method=HTTPMethod.POST,
                session=custom_session,
            )

            assert http_request._external_aio_session is custom_session
            # Verify the connector is configured
            assert custom_session.connector is not None
        finally:
            await custom_session.close()

    @pytest.mark.asyncio
    async def test_custom_aio_session_with_headers(self):
        """Test an aio_session with custom headers"""
        custom_headers = {
            "User-Agent": "Custom-Agent/1.0",
            "X-Custom-Header": "custom-value",
        }
        custom_session = aiohttp.ClientSession(headers=custom_headers)

        try:
            http_request = HttpRequest(
                url="http://example.com/api",
                api_key="fake-api-key",
                http_method=HTTPMethod.POST,
                session=custom_session,
            )

            assert http_request._external_aio_session is custom_session
            # Verify the headers are configured
            assert "User-Agent" in custom_session.headers
        finally:
            await custom_session.close()

    @pytest.mark.asyncio
    async def test_custom_aio_session_with_timeout(self):
        """Test an aio_session with a custom timeout"""
        timeout = aiohttp.ClientTimeout(total=60)
        custom_session = aiohttp.ClientSession(timeout=timeout)

        try:
            http_request = HttpRequest(
                url="http://example.com/api",
                api_key="fake-api-key",
                http_method=HTTPMethod.POST,
                session=custom_session,
            )

            assert http_request._external_aio_session is custom_session
            # Verify the timeout is configured
            assert custom_session.timeout is not None
        finally:
            await custom_session.close()


class TestAsyncSessionPriority:
    """Tests for async session priority"""

    @pytest.mark.asyncio
    async def test_custom_aio_session_has_priority(self):
        """Test that a custom aio_session takes priority over a temporary one"""
        connector = aiohttp.TCPConnector()
        custom_session = aiohttp.ClientSession(connector=connector)

        try:
            http_request = HttpRequest(
                url="http://example.com/api",
                api_key="fake-api-key",
                http_method=HTTPMethod.POST,
                session=custom_session,
            )

            # Verify the custom aio_session is stored
            assert http_request._external_aio_session is custom_session
            assert http_request._external_aio_session is not None
        finally:
            await custom_session.close()


class TestAsyncSessionWithDifferentMethods:
    """Tests for async session usage across HTTP methods"""

    @pytest.mark.asyncio
    async def test_custom_aio_session_with_post_request(self):
        """Test that POST requests use the custom session"""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Make request() return an awaitable
        async def mock_request(*_args, **_kwargs):
            return mock_response

        mock_session.request = mock_request

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
            session=mock_session,
        )

        request_data = ApiRequestData(
            model="test-model",
            task_group="test",
            task="test",
            function="test",
            input_data={"test": "data"},
            form=None,
            is_binary_input=False,
            api_protocol=ApiProtocol.HTTPS,
        )
        http_request.data = request_data

        async def mock_handle_response(_response):
            yield mock_response

        with patch.object(
            http_request,
            "_handle_aio_response",
            side_effect=mock_handle_response,
        ):
            _ = await http_request.aio_call()

        # Test passed if no exception

    @pytest.mark.asyncio
    async def test_custom_aio_session_with_get_request(self):
        """Test that GET requests use the custom session"""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Make get() return an awaitable
        async def mock_get(*args, **kwargs):
            return mock_response

        mock_session.get = mock_get

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.GET,
            stream=False,
            session=mock_session,
        )

        request_data = ApiRequestData(
            model="test-model",
            task_group="test",
            task="test",
            function="test",
            input_data={"test": "data"},
            form=None,
            is_binary_input=False,
            api_protocol=ApiProtocol.HTTPS,
        )
        http_request.data = request_data

        async def mock_handle_response(_response):
            yield mock_response

        with patch.object(
            http_request,
            "_handle_aio_response",
            side_effect=mock_handle_response,
        ):
            _ = await http_request.aio_call()

        # Test passed if no exception


class TestAsyncBackwardCompatibility:
    """Tests for async backward compatibility"""

    @pytest.mark.asyncio
    async def test_works_without_aio_session_parameter(self):
        """Test that omitting aio_session keeps the original behavior"""
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        # Verify _external_aio_session is None when aio_session is not passed
        assert http_request._external_aio_session is None

        # Verify the other parameters are unaffected
        assert http_request.url == "http://example.com/api"
        assert http_request.method == HTTPMethod.POST

    @pytest.mark.asyncio
    async def test_default_behavior_uses_shared_session(self):
        """Test that the default behavior uses the shared session (not closed)"""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.request.return_value = mock_response

        # Do not pass the aio_session parameter
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        request_data = ApiRequestData(
            model="test-model",
            task_group="test",
            task="test",
            function="test",
            input_data={"test": "data"},
            form=None,
            is_binary_input=False,
            api_protocol=ApiProtocol.HTTPS,
        )
        http_request.data = request_data

        async def mock_handle_response(_response):
            yield mock_response

        with patch(
            "dashscope.legacy_1x.api_entities.http_request.get_shared_aio_session",
            return_value=mock_session,
        ):
            with patch.object(
                http_request,
                "_handle_aio_response",
                side_effect=mock_handle_response,
            ):
                _ = await http_request.aio_call()

        # The shared session must not be closed (lifecycle managed
        # by the aio_session module)
        mock_session.close.assert_not_called()


class TestAsyncSessionLifecycle:
    """Tests for the async session lifecycle"""

    @pytest.mark.asyncio
    async def test_multiple_requests_with_same_custom_session(self):
        """Test multiple requests with the same custom session"""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Make request() return an awaitable
        async def mock_request(*_args, **_kwargs):
            return mock_response

        mock_session.request = mock_request

        # First request
        http_request1 = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
            session=mock_session,
        )

        request_data1 = ApiRequestData(
            model="test-model",
            task_group="test",
            task="test",
            function="test",
            input_data={"test": "data1"},
            form=None,
            is_binary_input=False,
            api_protocol=ApiProtocol.HTTPS,
        )
        http_request1.data = request_data1

        async def mock_handle_response(_response):
            yield mock_response

        with patch.object(
            http_request1,
            "_handle_aio_response",
            side_effect=mock_handle_response,
        ):
            _ = await http_request1.aio_call()

        # Second request
        http_request2 = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
            session=mock_session,
        )

        request_data2 = ApiRequestData(
            model="test-model",
            task_group="test",
            task="test",
            function="test",
            input_data={"test": "data2"},
            form=None,
            is_binary_input=False,
            api_protocol=ApiProtocol.HTTPS,
        )
        http_request2.data = request_data2

        with patch.object(
            http_request2,
            "_handle_aio_response",
            side_effect=mock_handle_response,
        ):
            _ = await http_request2.aio_call()

        # Verify the session was not closed (it is an external session)
        mock_session.close.assert_not_called()


class TestAsyncConnectionRetry:
    """Tests for retry behavior after a pooled async connection is closed
    by the remote peer"""

    def _build_post_request(self):
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )
        http_request.data = ApiRequestData(
            model="test-model",
            task_group="test",
            task="test",
            function="test",
            input_data={"test": "data"},
            form=None,
            is_binary_input=False,
            api_protocol=ApiProtocol.HTTPS,
        )
        return http_request

    @staticmethod
    def _ok_response():
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        return mock_response

    @pytest.mark.asyncio
    async def test_retry_once_on_connection_error(self):
        """Retries once and succeeds when the connection is closed by the remote peer"""
        mock_session = AsyncMock()
        mock_response = self._ok_response()
        mock_session.request.side_effect = [
            aiohttp.ClientConnectionError("Server disconnected"),
            mock_response,
        ]

        http_request = self._build_post_request()

        async def mock_handle_response(_response):
            yield mock_response

        with patch(
            "dashscope.legacy_1x.api_entities.http_request.get_shared_aio_session",
            return_value=mock_session,
        ):
            with patch.object(
                http_request,
                "_handle_aio_response",
                side_effect=mock_handle_response,
            ):
                _ = await http_request.aio_call()

        assert mock_session.request.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        """Raises after two consecutive ClientConnectionErrors"""
        mock_session = AsyncMock()
        mock_session.request.side_effect = aiohttp.ClientConnectionError(
            "Server disconnected",
        )

        http_request = self._build_post_request()

        with patch(
            "dashscope.legacy_1x.api_entities.http_request.get_shared_aio_session",
            return_value=mock_session,
        ):
            with pytest.raises(aiohttp.ClientConnectionError):
                _ = await http_request.aio_call()

        assert mock_session.request.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_other_exceptions(self):
        """Does not retry non-connection errors"""
        mock_session = AsyncMock()
        mock_session.request.side_effect = ValueError("bad request")

        http_request = self._build_post_request()

        with patch(
            "dashscope.legacy_1x.api_entities.http_request.get_shared_aio_session",
            return_value=mock_session,
        ):
            with pytest.raises(ValueError, match="bad request"):
                _ = await http_request.aio_call()

        assert mock_session.request.call_count == 1
