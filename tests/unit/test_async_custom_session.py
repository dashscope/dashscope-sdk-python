# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

"""
异步 HTTP 自定义 Session 功能单元测试

测试范围：
1. HttpRequest 接受自定义 aiohttp.ClientSession 参数
2. 自定义 aio_session 的使用和资源管理
3. 临时 aio_session 的创建和清理
4. Session 优先级逻辑
5. 不同场景下的异步 Session 行为

注意：所有测试都不依赖真实的 API Key
"""

# pylint: disable=protected-access,unused-argument,unused-variable
# pylint: disable=broad-exception-raised

import ssl
from unittest.mock import patch, AsyncMock

import aiohttp
import certifi
import pytest

from dashscope.api_entities.http_request import HttpRequest
from dashscope.api_entities.api_request_data import ApiRequestData
from dashscope.common.constants import ApiProtocol, HTTPMethod


class TestAsyncSessionBasics:
    """测试异步 Session 基本功能"""

    @pytest.mark.asyncio
    async def test_http_request_accepts_aio_session_parameter(self):
        """测试 HttpRequest 接受 aio_session 参数"""
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
        """测试 HttpRequest 不传 aio_session 参数"""
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
        )

        assert http_request._external_aio_session is None

    @pytest.mark.asyncio
    async def test_aio_session_parameter_is_optional(self):
        """测试 aio_session 参数是可选的"""
        # 不传 aio_session 参数应该正常工作
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        assert http_request._external_aio_session is None
        assert http_request.url == "http://example.com/api"


class TestAsyncSessionUsage:
    """测试异步 Session 的实际使用"""

    @pytest.mark.asyncio
    async def test_custom_aio_session_is_used_for_request(self):
        """测试自定义 aio_session 被实际用于请求"""
        # 创建 mock session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Make request() return an awaitable
        async def mock_request(*_args, **_kwargs):
            return mock_response

        mock_session.request = mock_request

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
        async def mock_handle_response(_response):
            yield mock_response

        with patch.object(
            http_request,
            "_handle_aio_response",
            side_effect=mock_handle_response,
        ):
            _ = await http_request.aio_call()

        # 验证自定义 session 没有被关闭
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_temporary_aio_session_is_created_when_no_custom_session(
        self,
    ):
        """测试没有自定义 aio_session 时会创建临时 aio_session"""
        # 创建 mock session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.request.return_value = mock_response

        # 创建 HttpRequest 不传 aio_session
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
        async def mock_handle_response(_response):
            yield mock_response

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(
                http_request,
                "_handle_aio_response",
                side_effect=mock_handle_response,
            ):
                _ = await http_request.aio_call()

        # 验证临时 aio_session 被关闭
        mock_session.close.assert_called_once()


class TestAsyncSessionResourceManagement:
    """测试异步 Session 资源管理"""

    @pytest.mark.asyncio
    async def test_custom_aio_session_not_closed_by_http_request(self):
        """测试自定义 aio_session 不会被 HttpRequest 关闭"""
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

        # 验证自定义 aio_session 没有被关闭
        custom_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_temporary_aio_session_closed_on_success(self):
        """测试临时 aio_session 在成功后被关闭"""
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

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(
                http_request,
                "_handle_aio_response",
                side_effect=mock_handle_response,
            ):
                _ = await http_request.aio_call()

        # 验证临时 aio_session 被关闭
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_temporary_aio_session_closed_on_exception(self):
        """测试临时 aio_session 在异常时也被关闭"""
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

        # 执行请求应该抛出异常
        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match="Network error"):
                _ = await http_request.aio_call()

        # 验证临时 aio_session 仍然被关闭
        mock_session.close.assert_called_once()


class TestAsyncSessionWithCustomConfiguration:
    """测试自定义配置的异步 Session"""

    @pytest.mark.asyncio
    async def test_custom_aio_session_with_connector(self):
        """测试带自定义 connector 的 aio_session"""
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
            # 验证 connector 已配置
            assert custom_session.connector is not None
        finally:
            await custom_session.close()

    @pytest.mark.asyncio
    async def test_custom_aio_session_with_headers(self):
        """测试带自定义 headers 的 aio_session"""
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
            # 验证 headers 已配置
            assert "User-Agent" in custom_session.headers
        finally:
            await custom_session.close()

    @pytest.mark.asyncio
    async def test_custom_aio_session_with_timeout(self):
        """测试带自定义 timeout 的 aio_session"""
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
            # 验证 timeout 已配置
            assert custom_session.timeout is not None
        finally:
            await custom_session.close()


class TestAsyncSessionPriority:
    """测试异步 Session 优先级"""

    @pytest.mark.asyncio
    async def test_custom_aio_session_has_priority(self):
        """测试自定义 aio_session 优先于临时 aio_session"""
        connector = aiohttp.TCPConnector()
        custom_session = aiohttp.ClientSession(connector=connector)

        try:
            http_request = HttpRequest(
                url="http://example.com/api",
                api_key="fake-api-key",
                http_method=HTTPMethod.POST,
                session=custom_session,
            )

            # 验证存储了自定义 aio_session
            assert http_request._external_aio_session is custom_session
            assert http_request._external_aio_session is not None
        finally:
            await custom_session.close()


class TestAsyncSessionWithDifferentMethods:
    """测试不同 HTTP 方法的异步 Session 使用"""

    @pytest.mark.asyncio
    async def test_custom_aio_session_with_post_request(self):
        """测试 POST 请求使用自定义 session"""
        mock_session = AsyncMock()
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
        """测试 GET 请求使用自定义 session"""
        mock_session = AsyncMock()
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
    """测试异步向后兼容性"""

    @pytest.mark.asyncio
    async def test_works_without_aio_session_parameter(self):
        """测试不传 aio_session 参数时保持原有行为"""
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="fake-api-key",
            http_method=HTTPMethod.POST,
            stream=False,
        )

        # 验证不传 aio_session 时，_external_aio_session 为 None
        assert http_request._external_aio_session is None

        # 验证其他参数正常
        assert http_request.url == "http://example.com/api"
        assert http_request.method == HTTPMethod.POST

    @pytest.mark.asyncio
    async def test_default_behavior_unchanged(self):
        """测试默认行为未改变"""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        mock_session.request.return_value = mock_response

        # 不传 aio_session 参数
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

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(
                http_request,
                "_handle_aio_response",
                side_effect=mock_handle_response,
            ):
                _ = await http_request.aio_call()

        # 验证临时 aio_session 被关闭（原有行为）
        mock_session.close.assert_called_once()


class TestAsyncSessionLifecycle:
    """测试异步 Session 生命周期"""

    @pytest.mark.asyncio
    async def test_multiple_requests_with_same_custom_session(self):
        """测试使用同一个自定义 session 进行多次请求"""
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Make request() return an awaitable
        async def mock_request(*_args, **_kwargs):
            return mock_response

        mock_session.request = mock_request

        # 第一次请求
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

        # 第二次请求
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

        # 验证 session 没有被关闭（因为是外部 session）
        mock_session.close.assert_not_called()
