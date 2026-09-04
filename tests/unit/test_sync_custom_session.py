# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

"""
同步 HTTP 自定义 Session 功能单元测试

测试范围：
1. HttpRequest 接受自定义 Session 参数
2. 自定义 Session 的使用和资源管理
3. 临时 Session 的创建和清理
4. Session 优先级逻辑
5. 不同场景下的 Session 行为

注意：所有测试都不依赖真实的 API Key
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
    """测试同步 Session 基本功能"""

    def test_http_request_accepts_session_parameter(self):
        """测试 HttpRequest 接受 session 参数"""
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
        """测试 HttpRequest 不传 session 参数"""
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
        )

        assert http_request._external_session is None

    def test_session_parameter_is_optional(self):
        """测试 session 参数是可选的"""
        # 不传 session 参数应该正常工作
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        assert http_request._external_session is None
        assert http_request.url == "http://example.com/api"


class TestSyncSessionUsage:
    """测试同步 Session 的实际使用"""

    @patch("requests.Session")
    def test_custom_session_is_used_for_request(self, _mock_session_class):
        """测试自定义 session 被实际用于请求"""
        # 创建 mock session
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"status": "success"}'
        mock_session.post.return_value = mock_response

        # 创建 HttpRequest 并传入自定义 session
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
            session=mock_session,
        )

        # 添加请求数据
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

        # 执行请求
        with patch.object(
            http_request,
            "_handle_response",
            return_value=iter([mock_response]),
        ):
            _ = http_request.call()

        # 验证自定义 session 被使用
        mock_session.post.assert_called_once()

        # 验证自定义 session 没有被关闭
        mock_session.close.assert_not_called()

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_shared_session_is_used_when_no_custom_session(
        self,
        mock_get_session,
    ):
        """测试没有自定义 session 时使用共享 session"""
        # 创建 mock session
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"status": "success"}'
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        # 创建 HttpRequest 不传 session
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        # 添加请求数据
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

        # 执行请求
        with patch.object(
            http_request,
            "_handle_response",
            return_value=iter([mock_response]),
        ):
            _ = http_request.call()

        # 验证共享 session 被使用
        mock_get_session.assert_called_once()

        # 验证共享 session 没有被关闭
        mock_session.close.assert_not_called()


class TestSyncSessionResourceManagement:
    """测试同步 Session 资源管理"""

    def test_custom_session_not_closed_by_http_request(self):
        """测试自定义 session 不会被 HttpRequest 关闭"""
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

        # 验证自定义 session 没有被关闭
        custom_session.close.assert_not_called()

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_shared_session_not_closed_on_success(self, mock_get_session):
        """测试共享 session 在成功后不被关闭"""
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

        # 验证共享 session 没有被关闭
        mock_session.close.assert_not_called()

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_shared_session_not_closed_on_exception(self, mock_get_session):
        """测试共享 session 在异常时不被关闭"""
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

        # 执行请求应该抛出异常
        with pytest.raises(Exception, match="Network error"):
            _ = http_request.call()

        # 验证共享 session 没有被关闭
        mock_session.close.assert_not_called()


class TestSyncSessionWithCustomConfiguration:
    """测试自定义配置的 Session"""

    def test_custom_session_with_connection_pool(self):
        """测试带连接池配置的自定义 session"""
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
        # 验证 adapter 已配置
        assert "http://" in custom_session.adapters
        assert "https://" in custom_session.adapters

    def test_custom_session_with_headers(self):
        """测试带自定义 headers 的 session"""
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
        """测试带代理配置的 session"""
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
    """测试 Session 优先级"""

    def test_custom_session_has_priority(self):
        """测试自定义 session 优先于临时 session"""
        custom_session = requests.Session()

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            session=custom_session,
        )

        # 验证存储了自定义 session
        assert http_request._external_session is custom_session
        assert http_request._external_session is not None


class TestSyncSessionWithDifferentMethods:
    """测试不同 HTTP 方法的 Session 使用"""

    @patch("requests.Session")
    def test_custom_session_with_post_request(self, _mock_session_class):
        """测试 POST 请求使用自定义 session"""
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

        # 验证使用了 POST 方法
        mock_session.post.assert_called_once()

    @patch("requests.Session")
    def test_custom_session_with_get_request(self, _mock_session_class):
        """测试 GET 请求使用自定义 session"""
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

        # 验证使用了 GET 方法
        mock_session.get.assert_called_once()


class TestSyncBackwardCompatibility:
    """测试向后兼容性"""

    def test_works_without_session_parameter(self):
        """测试不传 session 参数时保持原有行为"""
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        # 验证不传 session 时，_external_session 为 None
        assert http_request._external_session is None

        # 验证其他参数正常
        assert http_request.url == "http://example.com/api"
        assert http_request.method == HTTPMethod.POST

    @patch("dashscope.api_entities.http_request._get_shared_sync_session")
    def test_default_behavior_unchanged(self, mock_get_session):
        """测试默认行为：使用共享 session"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"status": "success"}'
        mock_session.post.return_value = mock_response
        mock_get_session.return_value = mock_session

        # 不传 session 参数
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

        # 验证共享 session 被使用
        mock_get_session.assert_called_once()
        # 验证共享 session 没有被关闭
        mock_session.close.assert_not_called()


class TestSyncConnectionRetry:
    """测试池化连接被远端关闭后的重试行为"""

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
        """连接被远端关闭时重试一次并成功"""
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
        """连续两次 ConnectionError 后向上抛出"""
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
        """非连接类异常不重试"""
        mock_session = Mock()
        mock_session.post.side_effect = ValueError("bad request")
        mock_get_session.return_value = mock_session

        http_request = self._build_post_request()

        with pytest.raises(ValueError, match="bad request"):
            _ = http_request.call()

        assert mock_session.post.call_count == 1


class TestSharedSyncSessionPool:
    """测试共享同步 session 的连接保活配置与生命周期管理"""

    def teardown_method(self):
        """每个测试后重置全局共享 session，避免用例间相互影响"""
        http_request_module.close_shared_sync_session()

    def test_shared_session_enables_tcp_keepalive(self):
        """共享 session 的连接池应启用 TCP keepalive"""
        session = http_request_module._get_shared_sync_session()
        assert isinstance(session, requests.Session)

        adapter = session.get_adapter("https://dashscope.aliyuncs.com")
        socket_options = adapter.poolmanager.connection_pool_kw[
            "socket_options"
        ]

        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in socket_options
        # 不能丢失 urllib3 默认的 TCP_NODELAY
        assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in socket_options

    def test_keepalive_options_platform_fallback(self):
        """所有平台都至少启用 SO_KEEPALIVE 与 TCP_NODELAY"""
        options = http_request_module._tcp_keepalive_socket_options()
        assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in options
        assert (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) in options

    def test_shared_session_reused_across_calls(self):
        """共享 session 在多次调用间复用"""
        session1 = http_request_module._get_shared_sync_session()
        session2 = http_request_module._get_shared_sync_session()
        assert session1 is session2

    def test_close_shared_sync_session_resets_pool(self):
        """关闭共享 session 后，下次请求创建新 session"""
        session1 = http_request_module._get_shared_sync_session()
        http_request_module.close_shared_sync_session()
        assert http_request_module._shared_sync_session is None

        session2 = http_request_module._get_shared_sync_session()
        assert session2 is not session1

    def test_close_shared_sync_session_closes_session(self):
        """关闭共享 session 会释放其连接池"""
        mock_session = Mock()
        http_request_module._shared_sync_session = mock_session
        http_request_module.close_shared_sync_session()
        mock_session.close.assert_called_once()
        assert http_request_module._shared_sync_session is None

    def test_close_shared_sync_session_without_session(self):
        """未创建共享 session 时关闭操作是安全的"""
        http_request_module._shared_sync_session = None
        http_request_module.close_shared_sync_session()
        assert http_request_module._shared_sync_session is None
