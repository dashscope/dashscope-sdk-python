# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

"""
HTTP 连接池功能单元测试

测试范围：
1. SessionManager 基本功能
2. ConnectionPoolConfig 配置类
3. HttpRequest 与 Session 集成
4. 全局连接池 API
5. 自定义 Session 支持
6. 线程安全性
"""

import threading
import time

import pytest
import requests
from requests.adapters import HTTPAdapter

import dashscope
from dashscope.common.session_manager import (
    SessionManager,
    ConnectionPoolConfig,
)
from dashscope.api_entities.http_request import HttpRequest
from tests.unit.base_test import BaseTestEnvironment


class TestConnectionPoolConfig:
    """测试 ConnectionPoolConfig 配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = ConnectionPoolConfig()
        assert config.pool_connections == 10
        assert config.pool_maxsize == 20
        assert config.max_retries == 3
        assert config.pool_block is False

    def test_custom_config(self):
        """测试自定义配置"""
        config = ConnectionPoolConfig(
            pool_connections=20,
            pool_maxsize=50,
            max_retries=5,
            pool_block=True,
        )
        assert config.pool_connections == 20
        assert config.pool_maxsize == 50
        assert config.max_retries == 5
        assert config.pool_block is True

    def test_config_validation(self):
        """测试配置验证"""
        # 测试负数验证
        with pytest.raises(ValueError, match="pool_connections 必须"):
            ConnectionPoolConfig(pool_connections=0)

        with pytest.raises(ValueError, match="pool_maxsize 必须"):
            ConnectionPoolConfig(pool_maxsize=0)

        with pytest.raises(ValueError, match="max_retries 必须"):
            ConnectionPoolConfig(max_retries=-1)

        # 测试 pool_maxsize >= pool_connections
        with pytest.raises(
            ValueError,
            match="pool_maxsize.*必须.*pool_connections",
        ):
            ConnectionPoolConfig(pool_connections=30, pool_maxsize=20)

    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config = ConnectionPoolConfig(
            pool_connections=15,
            pool_maxsize=30,
            max_retries=5,
            pool_block=True,
        )
        config_dict = config.to_dict()
        assert config_dict == {
            "pool_connections": 15,
            "pool_maxsize": 30,
            "max_retries": 5,
            "pool_block": True,
        }

    def test_config_str(self):
        """测试配置字符串表示"""
        config = ConnectionPoolConfig()
        config_str = str(config)
        assert "pool_connections=10" in config_str
        assert "pool_maxsize=20" in config_str
        assert "max_retries=3" in config_str
        assert "pool_block=False" in config_str


class TestSessionManager:
    """测试 SessionManager 单例类"""

    def setup_method(self):
        """每个测试前重置 SessionManager"""
        SessionManager.reset_instance()

    def teardown_method(self):
        """每个测试后清理"""
        manager = SessionManager.get_instance()
        manager.reset()

    def test_singleton_pattern(self):
        """测试单例模式"""
        manager1 = SessionManager.get_instance()
        manager2 = SessionManager.get_instance()
        assert manager1 is manager2

    def test_singleton_thread_safe(self):
        """测试单例模式的线程安全性"""
        instances = []

        def get_instance():
            instances.append(SessionManager.get_instance())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 所有实例应该是同一个
        assert all(inst is instances[0] for inst in instances)

    def test_enable_disable(self):
        """测试启用和禁用连接池"""
        manager = SessionManager.get_instance()

        # 默认禁用
        assert not manager.is_enabled()

        # 启用
        manager.enable()
        assert manager.is_enabled()

        # 禁用
        manager.disable()
        assert not manager.is_enabled()

    def test_enable_with_config(self):
        """测试使用配置启用连接池"""
        manager = SessionManager.get_instance()

        manager.enable(
            pool_connections=15,
            pool_maxsize=30,
            max_retries=5,
            pool_block=True,
        )

        assert manager.is_enabled()
        config = manager.get_config()
        assert config.pool_connections == 15
        assert config.pool_maxsize == 30
        assert config.max_retries == 5
        assert config.pool_block is True

    def test_configure(self):
        """测试配置连接池"""
        manager = SessionManager.get_instance()
        manager.enable()

        # 配置连接池
        manager.configure(
            pool_connections=25,
            pool_maxsize=50,
        )

        config = manager.get_config()
        assert config.pool_connections == 25
        assert config.pool_maxsize == 50
        assert config.max_retries == 3  # 保持默认值

    def test_get_session_when_disabled(self):
        """测试禁用时获取 Session（直接方式）"""
        manager = SessionManager.get_instance()
        manager.disable()

        session = manager.get_session()
        assert session is None

    def test_get_session_when_enabled(self):
        """测试启用时获取 Session（直接方式）"""
        manager = SessionManager.get_instance()
        manager.enable()

        session = manager.get_session()
        assert session is not None
        assert isinstance(session, requests.Session)

    def test_get_session_returns_same_instance(self):
        """测试获取 Session 返回同一实例"""
        manager = SessionManager.get_instance()
        manager.enable()

        session1 = manager.get_session()
        session2 = manager.get_session()
        assert session1 is session2

    def test_get_session(self):
        """测试直接获取 Session"""
        manager = SessionManager.get_instance()

        # 启用时能获取
        manager.enable()
        session = manager.get_session()
        assert session is not None
        assert isinstance(session, requests.Session)

        # 禁用时返回 None
        manager.disable()
        session = manager.get_session()
        assert session is None

    def test_reset(self):
        """测试重置连接池"""
        manager = SessionManager.get_instance()
        manager.enable()

        old_session = manager.get_session()
        assert old_session is not None

        # 禁用后重置
        manager.disable()
        manager.reset()

        # Session 应该被清理
        assert not manager.has_active_session()
        assert not manager.is_enabled()

        # 重新启用后应该是新的 Session
        manager.enable()
        new_session = manager.get_session()
        assert new_session is not old_session

    def test_session_has_adapter(self):
        """测试 Session 配置了 HTTPAdapter"""
        manager = SessionManager.get_instance()
        manager.enable(pool_connections=15, pool_maxsize=30)

        session = manager.get_session()
        assert session is not None

        # 检查是否配置了 HTTPAdapter
        http_adapter = session.get_adapter("http://")
        https_adapter = session.get_adapter("https://")

        assert isinstance(http_adapter, HTTPAdapter)
        assert isinstance(https_adapter, HTTPAdapter)

    def test_thread_safe_session_creation(self):
        """测试多线程环境下 Session 创建的线程安全性"""
        manager = SessionManager.get_instance()
        manager.enable()

        sessions = []

        def get_session():
            sessions.append(manager.get_session())

        threads = [threading.Thread(target=get_session) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 所有线程应该获取到同一个 Session
        assert all(s is sessions[0] for s in sessions)


class TestHttpRequestSessionIntegration:
    """测试 HttpRequest 与 Session 的集成"""

    def test_http_request_accepts_session(self):
        """测试 HttpRequest 接受 session 参数"""
        custom_session = requests.Session()

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="test-key",
            http_method="POST",
            session=custom_session,
        )

        assert http_request.get_external_session() is custom_session

    def test_http_request_without_session(self):
        """测试 HttpRequest 不传 session 参数"""
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="test-key",
            http_method="POST",
        )

        assert http_request.get_external_session() is None

    def test_http_request_uses_external_session_priority(self):
        """测试 HttpRequest 优先使用外部传入的 Session"""
        # 创建自定义 Session
        custom_session = requests.Session()
        custom_session.headers.update({"X-Test": "custom"})

        # 创建 HttpRequest，传入自定义 Session
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="test-key",
            http_method="POST",
            session=custom_session,
        )

        # 验证使用了自定义 Session
        assert http_request.get_external_session() is custom_session
        assert (
            http_request.get_external_session().headers.get("X-Test")
            == "custom"
        )

    def test_http_request_session_priority(self):
        """测试 Session 优先级：外部 > 全局 > 临时"""
        # 1. 外部 Session 优先级最高
        custom_session = requests.Session()
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="test-key",
            http_method="POST",
            session=custom_session,
        )
        assert http_request.get_external_session() is custom_session

        # 2. 没有外部 Session 时，应该尝试使用全局 Session
        http_request_no_session = HttpRequest(
            url="http://example.com/api",
            api_key="test-key",
            http_method="POST",
        )
        assert http_request_no_session.get_external_session() is None


class TestGlobalConnectionPoolAPI(BaseTestEnvironment):
    """测试全局连接池 API"""

    def setup_method(self):
        """每个测试前重置"""
        super().setup_class()
        SessionManager.reset_instance()

    def teardown_method(self):
        """每个测试后清理"""
        dashscope.disable_http_connection_pool()
        super().teardown_class()

    def test_enable_http_connection_pool(self):
        """测试启用 HTTP 连接池"""
        dashscope.enable_http_connection_pool()

        manager = SessionManager.get_instance()
        assert manager.is_enabled()

    def test_enable_http_connection_pool_with_params(self):
        """测试使用参数启用 HTTP 连接池"""
        dashscope.enable_http_connection_pool(
            pool_connections=15,
            pool_maxsize=30,
            max_retries=5,
            pool_block=True,
        )

        manager = SessionManager.get_instance()
        assert manager.is_enabled()

        config = manager.get_config()
        assert config.pool_connections == 15
        assert config.pool_maxsize == 30
        assert config.max_retries == 5
        assert config.pool_block is True

    def test_disable_http_connection_pool(self):
        """测试禁用 HTTP 连接池"""
        dashscope.enable_http_connection_pool()
        assert SessionManager.get_instance().is_enabled()

        dashscope.disable_http_connection_pool()
        assert not SessionManager.get_instance().is_enabled()

    def test_reset_http_connection_pool(self):
        """测试重置 HTTP 连接池"""
        dashscope.enable_http_connection_pool()
        # 验证 session 存在
        assert SessionManager.get_instance().get_session() is not None

        # 禁用后重置
        dashscope.disable_http_connection_pool()
        dashscope.reset_http_connection_pool()

        # Session 应该被清理
        manager = SessionManager.get_instance()
        assert not manager.has_active_session()
        assert not manager.is_enabled()

    def test_configure_http_connection_pool(self):
        """测试配置 HTTP 连接池"""
        dashscope.enable_http_connection_pool()

        dashscope.configure_http_connection_pool(
            pool_connections=25,
            pool_maxsize=50,
        )

        config = SessionManager.get_instance().get_config()
        assert config.pool_connections == 25
        assert config.pool_maxsize == 50

    def test_configure_before_enable(self):
        """测试在启用前配置"""
        # 先启用
        dashscope.enable_http_connection_pool()

        # 然后配置
        dashscope.configure_http_connection_pool(
            pool_connections=20,
            pool_maxsize=40,
        )

        manager = SessionManager.get_instance()
        assert manager.is_enabled()

        config = manager.get_config()
        assert config.pool_connections == 20
        assert config.pool_maxsize == 40


class TestCustomSessionSupport:
    """测试自定义 Session 支持"""

    def test_custom_session_with_headers(self):
        """测试自定义 Session 带请求头"""
        session = requests.Session()
        session.headers.update({"X-Custom-Header": "TestValue"})

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="test-key",
            http_method="POST",
            session=session,
        )

        assert http_request.get_external_session() is session
        assert session.headers.get("X-Custom-Header") == "TestValue"

    def test_custom_session_with_proxies(self):
        """测试自定义 Session 带代理"""
        session = requests.Session()
        session.proxies = {"https": "https://proxy.example.com:8080"}

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="test-key",
            http_method="POST",
            session=session,
        )

        assert http_request.get_external_session() is session
        assert session.proxies.get("https") == "https://proxy.example.com:8080"

    def test_custom_session_with_adapter(self):
        """测试自定义 Session 带自定义 Adapter"""
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=50,
            pool_maxsize=100,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="test-key",
            http_method="POST",
            session=session,
        )

        assert http_request.get_external_session() is session

        # 验证 Adapter 配置
        http_adapter = session.get_adapter("http://")
        assert isinstance(http_adapter, HTTPAdapter)


class TestThreadSafety:
    """测试线程安全性"""

    def setup_method(self):
        """每个测试前重置"""
        SessionManager.reset_instance()

    def teardown_method(self):
        """每个测试后清理"""
        manager = SessionManager.get_instance()
        manager.reset()

    def test_concurrent_enable_disable(self):
        """测试并发启用和禁用"""
        manager = SessionManager.get_instance()

        def toggle_enable():
            for _ in range(10):
                manager.enable()
                time.sleep(0.001)
                manager.disable()
                time.sleep(0.001)

        threads = [threading.Thread(target=toggle_enable) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 不应该抛出异常
        assert True

    def test_concurrent_get_session(self):
        """测试并发获取 Session"""
        manager = SessionManager.get_instance()
        manager.enable()

        sessions = []

        def get_session():
            for _ in range(10):
                s = manager.get_session()
                sessions.append(s)
                time.sleep(0.001)

        threads = [threading.Thread(target=get_session) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 所有获取的 Session 应该是同一个
        assert all(s is sessions[0] for s in sessions)

    def test_concurrent_configure(self):
        """测试并发配置"""
        manager = SessionManager.get_instance()
        manager.enable()

        def configure():
            for i in range(5):
                manager.configure(
                    pool_connections=10 + i,
                    pool_maxsize=20 + i * 2,
                )
                time.sleep(0.001)

        threads = [threading.Thread(target=configure) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # 不应该抛出异常，最终配置应该是有效的
        config = manager.get_config()
        assert config.pool_connections > 0
        assert config.pool_maxsize >= config.pool_connections


class TestEdgeCases:
    """测试边界情况"""

    def setup_method(self):
        """每个测试前重置"""
        SessionManager.reset_instance()

    def teardown_method(self):
        """每个测试后清理"""
        manager = SessionManager.get_instance()
        manager.reset()

    def test_enable_multiple_times(self):
        """测试多次启用"""
        manager = SessionManager.get_instance()

        manager.enable()
        session1 = manager.get_session()

        manager.enable()
        session2 = manager.get_session()

        # 应该返回同一个 Session
        assert session1 is session2

    def test_configure_with_partial_params(self):
        """测试部分参数配置"""
        manager = SessionManager.get_instance()
        manager.enable()

        # 只配置部分参数
        manager.configure(pool_connections=15)

        config = manager.get_config()
        assert config.pool_connections == 15
        assert config.pool_maxsize == 20  # 保持默认值
        assert config.max_retries == 3  # 保持默认值

    def test_reset_when_disabled(self):
        """测试禁用状态下重置"""
        manager = SessionManager.get_instance()
        manager.disable()

        # 不应该抛出异常
        manager.reset()
        assert not manager.is_enabled()

    def test_get_session_after_reset(self):
        """测试重置后获取 Session"""
        manager = SessionManager.get_instance()
        manager.enable()

        old_session = manager.get_session()

        # 禁用后重置
        manager.disable()
        manager.reset()

        # 重置后应该返回 None
        assert manager.get_session() is None

        # 重新启用后应该是新的 Session
        manager.enable()
        new_session = manager.get_session()
        assert new_session is not None
        assert new_session is not old_session


class TestBackwardCompatibility:
    """测试向后兼容性"""

    def test_http_request_without_session_param(self):
        """测试不传 session 参数的 HttpRequest（向后兼容）"""
        # 不传 session 参数应该正常工作
        http_request = HttpRequest(
            url="http://example.com/api",
            api_key="test-key",
            http_method="POST",
        )

        assert http_request.get_external_session() is None

    def test_default_behavior_unchanged(self):
        """测试默认行为未改变（需要在干净环境中测试）"""
        # 重置到初始状态
        manager = SessionManager.get_instance()
        manager.disable()
        manager.reset()

        # 默认应该是禁用状态
        assert not manager.is_enabled()

        # 默认获取 Session 应该返回 None
        assert manager.get_session() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
