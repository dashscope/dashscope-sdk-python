# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

"""
Unit tests for custom sync HTTP Session support.

Scope:
1. HttpRequest accepts a custom Session parameter
2. Usage and resource management of custom Sessions
3. Creation and cleanup of the SDK-managed shared Session
4. Session priority logic
5. Session behavior in different scenarios

Note: no test depends on a real API key.
"""

# pylint: disable=protected-access,unused-argument,unused-variable
# pylint: disable=broad-exception-raised

import socket
from unittest.mock import Mock, patch

import pytest
import requests
from requests.adapters import HTTPAdapter

from dashscope.api_entities import http_request as http_request_module
from dashscope.api_entities.http_request import HttpRequest
from dashscope.api_entities.api_request_data import ApiRequestData
from dashscope.common.constants import ApiProtocol, HTTPMethod


class TestSyncSessionBasics:
    """Basic sync Session functionality"""

    def test_http_request_accepts_session_parameter(self):
        """HttpRequest accepts the session parameter"""
        custom_session = requests.Session()

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            session=custom_session,
        )

        assert http_request._external_session is custom_session
        assert http_request._external_session is not None

    def test_http_request_without_session_parameter(self):
        """HttpRequest without the session parameter"""
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
        )

        assert http_request._external_session is None

    def test_session_parameter_is_optional(self):
        """The session parameter is optional"""
        # omitting the session parameter should work
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        assert http_request._external_session is None
        assert http_request.url == "http://example.com/api"


class TestSyncSessionUsage:
    """Actual usage of sync Sessions"""

    @patch("requests.Session")
    def test_custom_session_is_used_for_request(self, _mock_session_class):
        """Custom session is actually used for the request"""
        # create mock session
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"status": "success"}'
        mock_session.post.return_value = mock_response

        # create HttpRequest with the custom session
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
            session=mock_session,
        )

        # add request data
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

        # execute the request
        with patch.object(
            http_request,
            "_handle_response",
            return_value=iter([mock_response]),
        ):
            _ = http_request.call()

        # verify the custom session was used
        mock_session.post.assert_called_once()

        # verify the custom session was not closed
        mock_session.close.assert_not_called()

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_shared_session_is_used_when_no_custom_session(
        self,
        mock_get_session,
    ):
        """Shared session is used without a custom session"""
        # create mock session
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"status": "success"}'
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        # create HttpRequest without a session
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        # add request data
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

        # execute the request
        with patch.object(
            http_request,
            "_handle_response",
            return_value=iter([mock_response]),
        ):
            _ = http_request.call()

        # verify the shared session was used
        mock_get_session.assert_called_once()

        # verify the shared session was not closed
        mock_session.close.assert_not_called()


class TestSyncSessionResourceManagement:
    """Sync Session resource management"""

    def test_custom_session_not_closed_by_http_request(self):
        """Custom session is not closed by HttpRequest"""
        custom_session = Mock(spec=requests.Session)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"status": "success"}'
        custom_session.post.return_value = mock_response

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

        with patch.object(
            http_request,
            "_handle_response",
            return_value=iter([mock_response]),
        ):
            _ = http_request.call()

        # verify the custom session was not closed
        custom_session.close.assert_not_called()

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_shared_session_not_closed_on_success(self, mock_get_session):
        """Shared session is not closed after success"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"status": "success"}'
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

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

        with patch.object(
            http_request,
            "_handle_response",
            return_value=iter([mock_response]),
        ):
            _ = http_request.call()

        # verify the shared session was not closed
        mock_session.close.assert_not_called()

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_shared_session_not_closed_on_exception(self, mock_get_session):
        """Shared session is not closed on exception"""
        mock_session = Mock()
        mock_session.post.side_effect = Exception("Network error")
        mock_get_session.return_value = mock_session

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

        # the request should raise
        with pytest.raises(Exception, match="Network error"):
            _ = http_request.call()

        # verify the shared session was not closed
        mock_session.close.assert_not_called()


class TestSyncSessionWithCustomConfiguration:
    """Sessions with custom configuration"""

    def test_custom_session_with_connection_pool(self):
        """Custom session with connection pool configuration"""
        custom_session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3,
        )
        custom_session.mount("http://", adapter)
        custom_session.mount("https://", adapter)

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            session=custom_session,
        )

        assert http_request._external_session is custom_session
        # verify the adapter is configured
        assert "http://" in custom_session.adapters
        assert "https://" in custom_session.adapters

    def test_custom_session_with_headers(self):
        """Custom session with custom headers"""
        custom_session = requests.Session()
        custom_session.headers.update(
            {
                "User-Agent": "Custom-Agent/1.0",
                "X-Custom-Header": "custom-value",
            },
        )

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            session=custom_session,
        )

        assert http_request._external_session is custom_session
        assert "User-Agent" in custom_session.headers
        assert custom_session.headers["User-Agent"] == "Custom-Agent/1.0"

    def test_custom_session_with_proxies(self):
        """Custom session with proxy configuration"""
        custom_session = requests.Session()
        custom_session.proxies = {
            "http": "http://proxy.example.com:8080",
            "https": "https://proxy.example.com:8080",
        }

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            session=custom_session,
        )

        assert http_request._external_session is custom_session
        assert (
            custom_session.proxies["http"] == "http://proxy.example.com:8080"
        )


class TestSyncSessionPriority:
    """Session priority"""

    def test_custom_session_has_priority(self):
        """Custom session takes priority over the shared session"""
        custom_session = requests.Session()

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            session=custom_session,
        )

        # verify the custom session is stored
        assert http_request._external_session is custom_session
        assert http_request._external_session is not None


class TestSyncSessionWithDifferentMethods:
    """Session usage with different HTTP methods"""

    @patch("requests.Session")
    def test_custom_session_with_post_request(self, _mock_session_class):
        """POST request uses the custom session"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.post.return_value = mock_response

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

        with patch.object(
            http_request,
            "_handle_response",
            return_value=iter([mock_response]),
        ):
            _ = http_request.call()

        # verify POST was used
        mock_session.post.assert_called_once()

    @patch("requests.Session")
    def test_custom_session_with_get_request(self, _mock_session_class):
        """GET request uses the custom session"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response

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

        with patch.object(
            http_request,
            "_handle_response",
            return_value=iter([mock_response]),
        ):
            _ = http_request.call()

        # verify GET was used
        mock_session.get.assert_called_once()


class TestSyncBackwardCompatibility:
    """Backward compatibility"""

    def test_works_without_session_parameter(self):
        """Behavior is unchanged without the session parameter"""
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        # _external_session is None without a session
        assert http_request._external_session is None

        # other parameters are set normally
        assert http_request.url == "http://example.com/api"
        assert http_request.method == HTTPMethod.POST

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_default_behavior_unchanged(self, mock_get_session):
        """Default behavior: use the shared session"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"status": "success"}'
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        # no session parameter
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

        with patch.object(
            http_request,
            "_handle_response",
            return_value=iter([mock_response]),
        ):
            _ = http_request.call()

        # verify the shared session was used
        mock_get_session.assert_called_once()
        # verify the shared session was not closed
        mock_session.close.assert_not_called()


class TestSyncConnectionRetry:
    """Retry behavior when a pooled connection is dropped"""

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
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"status": "success"}'
        return mock_response

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_retry_once_on_connection_error(self, mock_get_session):
        """Retry once and succeed when the peer drops the connection"""
        mock_session = Mock()
        mock_response = self._ok_response()
        mock_session.post.side_effect = [
            requests.exceptions.ConnectionError(
                "Connection aborted.",
            ),
            mock_response,
        ]
        mock_get_session.return_value = mock_session

        http_request = self._build_post_request()

        with patch.object(
            http_request,
            "_handle_response",
            return_value=iter([mock_response]),
        ):
            _ = http_request.call()

        assert mock_session.post.call_count == 2

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_retry_exhausted_raises(self, mock_get_session):
        """Raise after two consecutive ConnectionErrors"""
        mock_session = Mock()
        mock_session.post.side_effect = requests.exceptions.ConnectionError(
            "Connection aborted.",
        )
        mock_get_session.return_value = mock_session

        http_request = self._build_post_request()

        with pytest.raises(requests.exceptions.ConnectionError):
            _ = http_request.call()

        assert mock_session.post.call_count == 2

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_no_retry_on_other_exceptions(self, mock_get_session):
        """Non-connection exceptions are not retried"""
        mock_session = Mock()
        mock_session.post.side_effect = ValueError("bad request")
        mock_get_session.return_value = mock_session

        http_request = self._build_post_request()

        with pytest.raises(ValueError, match="bad request"):
            _ = http_request.call()

        assert mock_session.post.call_count == 1


class TestSharedSyncSessionPool:
    """Keepalive configuration and lifecycle of the shared session"""

    def teardown_method(self):
        """Reset the global shared session after each test"""
        http_request_module.close_shared_sync_session()

    def test_shared_session_enables_tcp_keepalive(self):
        """The shared session's connection pool enables TCP keepalive"""
        session = http_request_module._get_shared_sync_session()
        assert isinstance(session, requests.Session)

        adapter = session.get_adapter("https://dashscope.aliyuncs.com")
        socket_options = adapter.poolmanager.connection_pool_kw[
            "socket_options"
        ]

        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in socket_options
        # urllib3's default TCP_NODELAY must be preserved
        assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in socket_options

    def test_keepalive_options_platform_fallback(self):
        """SO_KEEPALIVE and TCP_NODELAY are enabled on all platforms"""
        options = http_request_module._tcp_keepalive_socket_options()
        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in options
        assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in options

    def test_shared_session_reused_across_calls(self):
        """The shared session is reused across calls"""
        session1 = http_request_module._get_shared_sync_session()
        session2 = http_request_module._get_shared_sync_session()
        assert session1 is session2

    def test_close_shared_sync_session_resets_pool(self):
        """Closing the shared session resets the pool"""
        session1 = http_request_module._get_shared_sync_session()
        http_request_module.close_shared_sync_session()
        assert http_request_module._shared_sync_session is None

        session2 = http_request_module._get_shared_sync_session()
        assert session2 is not session1

    def test_close_shared_sync_session_closes_session(self):
        """Closing the shared session releases its connection pool"""
        mock_session = Mock()
        http_request_module._shared_sync_session = mock_session
        http_request_module.close_shared_sync_session()
        mock_session.close.assert_called_once()
        assert http_request_module._shared_sync_session is None

    def test_close_shared_sync_session_without_session(self):
        """Closing before any shared session exists is safe"""
        http_request_module._shared_sync_session = None
        http_request_module.close_shared_sync_session()
        assert http_request_module._shared_sync_session is None
