# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

"""异步连接池单元测试"""

import asyncio

import aiohttp
import pytest

from dashscope.common.aio_session_manager import (
    AioConnectionPoolConfig,
    AioSessionManager,
)


class TestAioConnectionPoolConfig:
    """测试 AioConnectionPoolConfig 类"""

    def test_default_config(self):
        """测试默认配置"""
        config = AioConnectionPoolConfig()
        assert config.limit == 100
        assert config.limit_per_host == 30
        assert config.ttl_dns_cache == 300
        assert config.keepalive_timeout == 30
        assert config.force_close is False

    def test_custom_config(self):
        """测试自定义配置"""
        config = AioConnectionPoolConfig(
            limit=200,
            limit_per_host=50,
            ttl_dns_cache=600,
            keepalive_timeout=60,
            force_close=True,
        )
        assert config.limit == 200
        assert config.limit_per_host == 50
        assert config.ttl_dns_cache == 600
        assert config.keepalive_timeout == 60
        assert config.force_close is True

    def test_config_validation(self):
        """测试配置参数验证"""
        # limit 必须 > 0
        with pytest.raises(ValueError, match=r"limit.*必须 > 0"):
            AioConnectionPoolConfig(limit=0)

        # limit_per_host 必须 > 0
        with pytest.raises(ValueError, match=r"limit_per_host.*必须 > 0"):
            AioConnectionPoolConfig(limit_per_host=0)

        # limit_per_host 必须 <= limit
        with pytest.raises(ValueError, match=r"limit_per_host.*必须 <="):
            AioConnectionPoolConfig(limit=50, limit_per_host=100)

        # ttl_dns_cache 必须 >= 0
        with pytest.raises(ValueError, match=r"ttl_dns_cache.*必须 >= 0"):
            AioConnectionPoolConfig(ttl_dns_cache=-1)

        # keepalive_timeout 必须 >= 0
        with pytest.raises(ValueError, match=r"keepalive_timeout.*必须 >= 0"):
            AioConnectionPoolConfig(keepalive_timeout=-1)

    def test_config_repr(self):
        """测试配置的字符串表示"""
        config = AioConnectionPoolConfig(limit=200, limit_per_host=50)
        repr_str = repr(config)
        assert "AioConnectionPoolConfig" in repr_str
        assert "limit=200" in repr_str
        assert "limit_per_host=50" in repr_str


class TestAioSessionManager:
    """测试 AioSessionManager 类"""

    @pytest.fixture(autouse=True)
    async def cleanup(self):
        """每个测试后清理单例实例"""
        yield
        await AioSessionManager.reset_instance()

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """测试单例模式"""
        manager1 = await AioSessionManager.get_instance()
        manager2 = await AioSessionManager.get_instance()
        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_default_state(self):
        """测试默认状态"""
        manager = await AioSessionManager.get_instance()
        assert not manager.is_enabled()
        assert not await manager.has_active_session()
        config = manager.get_config()
        assert config.limit == 100
        assert config.limit_per_host == 30

    @pytest.mark.asyncio
    async def test_enable(self):
        """测试启用连接池"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()
        assert manager.is_enabled()
        assert await manager.has_active_session()

    @pytest.mark.asyncio
    async def test_enable_with_config(self):
        """测试启用时配置参数"""
        manager = await AioSessionManager.get_instance()
        await manager.enable(limit=200, limit_per_host=50)
        config = manager.get_config()
        assert config.limit == 200
        assert config.limit_per_host == 50

    @pytest.mark.asyncio
    async def test_disable(self):
        """测试禁用连接池"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()
        assert manager.is_enabled()

        await manager.disable()
        assert not manager.is_enabled()
        assert not await manager.has_active_session()

    @pytest.mark.asyncio
    async def test_get_session(self):
        """测试获取 Session"""
        manager = await AioSessionManager.get_instance()

        # 禁用时返回 None
        session = await manager.get_session()
        assert session is None

        # 启用后返回 Session
        await manager.enable()
        session = await manager.get_session()
        assert session is not None
        assert isinstance(session, aiohttp.ClientSession)
        assert not session.closed

    @pytest.mark.asyncio
    async def test_get_session_direct(self):
        """测试直接获取 Session"""
        manager = await AioSessionManager.get_instance()

        # 禁用时返回 None
        session = await manager.get_session_direct()
        assert session is None

        # 启用后返回 Session
        await manager.enable()
        session = await manager.get_session_direct()
        assert session is not None

        # 禁用后 Session 被关闭
        await manager.disable()
        session = await manager.get_session_direct()
        assert session is None

    @pytest.mark.asyncio
    async def test_configure(self):
        """测试配置连接池"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()

        # 配置参数
        await manager.configure(limit=200, limit_per_host=50)
        config = manager.get_config()
        assert config.limit == 200
        assert config.limit_per_host == 50

    @pytest.mark.asyncio
    async def test_configure_before_enable(self):
        """测试启用前配置"""
        manager = await AioSessionManager.get_instance()

        # 启用前配置不会创建 Session
        await manager.configure(limit=200)
        assert not await manager.has_active_session()

        # 启用后使用配置的参数
        await manager.enable()
        config = manager.get_config()
        assert config.limit == 200

    @pytest.mark.asyncio
    async def test_reset(self):
        """测试重置连接池"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()

        old_session = await manager.get_session_direct()
        assert old_session is not None

        # 重置后创建新 Session
        await manager.reset()
        new_session = await manager.get_session_direct()
        assert new_session is not None
        assert new_session is not old_session

    @pytest.mark.asyncio
    async def test_reset_when_disabled(self):
        """测试禁用状态下重置"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()
        await manager.disable()

        # 禁用状态下重置不会创建 Session
        await manager.reset()
        assert not await manager.has_active_session()

    @pytest.mark.asyncio
    async def test_reset_instance(self):
        """测试重置单例实例"""
        manager1 = await AioSessionManager.get_instance()
        await manager1.enable()

        await AioSessionManager.reset_instance()

        manager2 = await AioSessionManager.get_instance()
        assert not manager2.is_enabled()
        assert not await manager2.has_active_session()

    @pytest.mark.asyncio
    async def test_session_reuse(self):
        """测试 Session 复用"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()

        session1 = await manager.get_session()
        session2 = await manager.get_session()
        assert session1 is session2

    @pytest.mark.asyncio
    async def test_session_recreation_on_configure(self):
        """测试配置变更时重新创建 Session"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()

        old_session = await manager.get_session_direct()

        # 配置变更后 Session 被重新创建
        await manager.configure(limit=200)
        new_session = await manager.get_session_direct()
        assert new_session is not old_session

    @pytest.mark.asyncio
    async def test_concurrent_enable(self):
        """测试并发启用"""
        manager = await AioSessionManager.get_instance()

        # 并发启用
        await asyncio.gather(
            manager.enable(),
            manager.enable(),
            manager.enable(),
        )

        assert manager.is_enabled()
        assert await manager.has_active_session()

    @pytest.mark.asyncio
    async def test_concurrent_get_session(self):
        """测试并发获取 Session"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()

        # 并发获取 Session
        sessions = await asyncio.gather(
            manager.get_session(),
            manager.get_session(),
            manager.get_session(),
        )

        # 所有 Session 应该是同一个实例
        assert all(s is sessions[0] for s in sessions)

    @pytest.mark.asyncio
    async def test_concurrent_enable_disable(self):
        """测试并发启用和禁用"""
        manager = await AioSessionManager.get_instance()

        async def enable_disable():
            await manager.enable()
            await asyncio.sleep(0.01)
            await manager.disable()

        # 并发执行启用和禁用
        await asyncio.gather(
            enable_disable(),
            enable_disable(),
            enable_disable(),
        )

        # 最终状态应该是禁用
        assert not manager.is_enabled()

    @pytest.mark.asyncio
    async def test_session_closed_detection(self):
        """测试 Session 关闭检测"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()

        session = await manager.get_session_direct()
        assert not session.closed

        # 手动关闭 Session
        await session.close()

        # get_session 应该创建新的 Session
        new_session = await manager.get_session()
        assert new_session is not session
        assert not new_session.closed


class TestAioConnectionPoolIntegration:
    """测试异步连接池集成"""

    @pytest.fixture(autouse=True)
    async def cleanup(self):
        """每个测试后清理"""
        yield
        await AioSessionManager.reset_instance()

    @pytest.mark.asyncio
    async def test_default_behavior_unchanged(self):
        """测试默认行为不变"""
        manager = await AioSessionManager.get_instance()

        # 默认禁用，不影响现有代码
        session = await manager.get_session()
        assert session is None

    @pytest.mark.asyncio
    async def test_enable_affects_all_requests(self):
        """测试启用后影响所有请求"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()

        # 所有请求应该使用同一个 Session
        session1 = await manager.get_session()
        session2 = await manager.get_session()
        assert session1 is session2

    @pytest.mark.asyncio
    async def test_disable_stops_reuse(self):
        """测试禁用后停止复用"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()

        session_before = await manager.get_session()
        assert session_before is not None

        await manager.disable()

        session_after = await manager.get_session()
        assert session_after is None

    @pytest.mark.asyncio
    async def test_multiple_enable_disable_cycles(self):
        """测试多次启用/禁用循环"""
        manager = await AioSessionManager.get_instance()

        for _ in range(3):
            await manager.enable()
            assert manager.is_enabled()
            session = await manager.get_session()
            assert session is not None

            await manager.disable()
            assert not manager.is_enabled()
            session = await manager.get_session()
            assert session is None


class TestAioCustomSession:
    """测试自定义异步 Session"""

    @pytest.fixture(autouse=True)
    async def cleanup(self):
        """每个测试后清理"""
        yield
        await AioSessionManager.reset_instance()

    @pytest.mark.asyncio
    async def test_external_session_priority(self):
        """测试外部 Session 优先级最高"""
        from dashscope.api_entities.http_request import HttpRequest

        # 创建外部 Session
        external_session = aiohttp.ClientSession()

        # 创建 HttpRequest（传入外部 Session）
        http_request = HttpRequest(
            url="https://example.com",
            api_key="test_key",
            http_method="POST",
            aio_session=external_session,
        )

        # 验证外部 Session 被存储
        assert http_request.get_external_aio_session() is external_session

        await external_session.close()

    @pytest.mark.asyncio
    async def test_external_session_overrides_global(self):
        """测试外部 Session 覆盖全局连接池"""
        from dashscope.api_entities.http_request import HttpRequest

        # 启用全局连接池
        manager = await AioSessionManager.get_instance()
        await manager.enable()

        # 创建外部 Session
        external_session = aiohttp.ClientSession()

        # 创建 HttpRequest（传入外部 Session）
        http_request = HttpRequest(
            url="https://example.com",
            api_key="test_key",
            http_method="POST",
            aio_session=external_session,
        )

        # 验证使用外部 Session
        assert http_request.get_external_aio_session() is external_session

        await external_session.close()


class TestAioConnectionPoolEdgeCases:
    """测试异步连接池边界情况"""

    @pytest.fixture(autouse=True)
    async def cleanup(self):
        """每个测试后清理"""
        yield
        await AioSessionManager.reset_instance()

    @pytest.mark.asyncio
    async def test_configure_partial_params(self):
        """测试部分配置参数"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()

        # 只配置部分参数
        await manager.configure(limit=200)
        config = manager.get_config()
        assert config.limit == 200
        assert config.limit_per_host == 30  # 保持默认值

    @pytest.mark.asyncio
    async def test_enable_multiple_times(self):
        """测试多次启用"""
        manager = await AioSessionManager.get_instance()

        await manager.enable()
        session1 = await manager.get_session_direct()

        await manager.enable()
        session2 = await manager.get_session_direct()

        # 多次启用不会重新创建 Session
        assert session1 is session2

    @pytest.mark.asyncio
    async def test_disable_multiple_times(self):
        """测试多次禁用"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()

        await manager.disable()
        await manager.disable()  # 不应该报错

        assert not manager.is_enabled()

    @pytest.mark.asyncio
    async def test_reset_multiple_times(self):
        """测试多次重置"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()

        await manager.reset()
        await manager.reset()  # 不应该报错

        assert manager.is_enabled()
        assert await manager.has_active_session()

    @pytest.mark.asyncio
    async def test_configure_with_no_params(self):
        """测试无参数配置"""
        manager = await AioSessionManager.get_instance()
        await manager.enable()

        old_config = manager.get_config()
        await manager.configure()  # 不传参数
        new_config = manager.get_config()

        # 配置应该保持不变
        assert old_config.limit == new_config.limit
        assert old_config.limit_per_host == new_config.limit_per_host


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
