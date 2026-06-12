# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

"""
Shared aiohttp session pool unit tests.

Tests the connection reuse and SSL context caching in aio_session module.
"""

# pylint: disable=protected-access

import asyncio
import ssl
from unittest.mock import patch, MagicMock

import aiohttp
import pytest

from dashscope.api_entities import aio_session


class TestSSLContextCaching:
    """Test SSL context is created once and reused."""

    def setup_method(self):
        aio_session._shared_ssl_context = None

    def test_get_ssl_context_returns_ssl_context(self):
        ctx = aio_session.get_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)

    def test_get_ssl_context_cached(self):
        ctx1 = aio_session.get_ssl_context()
        ctx2 = aio_session.get_ssl_context()
        assert ctx1 is ctx2

    def test_get_ssl_context_calls_create_default_context_once(self):
        with patch(
            "ssl.create_default_context",
            wraps=ssl.create_default_context,
        ) as mock_create:
            aio_session._shared_ssl_context = None
            aio_session.get_ssl_context()
            aio_session.get_ssl_context()
            aio_session.get_ssl_context()
            assert mock_create.call_count == 1


class TestSharedAioSession:
    """Test shared session creation and reuse."""

    def setup_method(self):
        aio_session._shared_ssl_context = None
        aio_session._aio_sessions.clear()

    @pytest.mark.asyncio
    async def test_get_shared_session_returns_client_session(self):
        session = await aio_session.get_shared_aio_session()
        try:
            assert isinstance(session, aiohttp.ClientSession)
            assert not session.closed
        finally:
            await aio_session.close_shared_aio_session()

    @pytest.mark.asyncio
    async def test_get_shared_session_reuses_same_session(self):
        s1 = await aio_session.get_shared_aio_session()
        s2 = await aio_session.get_shared_aio_session()
        try:
            assert s1 is s2
        finally:
            await aio_session.close_shared_aio_session()

    @pytest.mark.asyncio
    async def test_shared_session_has_tcp_connector(self):
        session = await aio_session.get_shared_aio_session()
        try:
            assert isinstance(session.connector, aiohttp.TCPConnector)
        finally:
            await aio_session.close_shared_aio_session()

    @pytest.mark.asyncio
    async def test_shared_session_uses_cached_ssl(self):
        session = await aio_session.get_shared_aio_session()
        try:
            ssl_ctx = aio_session.get_ssl_context()
            assert session.connector._ssl is ssl_ctx
        finally:
            await aio_session.close_shared_aio_session()

    @pytest.mark.asyncio
    async def test_close_shared_session(self):
        session = await aio_session.get_shared_aio_session()
        assert not session.closed
        await aio_session.close_shared_aio_session()
        assert session.closed

    @pytest.mark.asyncio
    async def test_new_session_after_close(self):
        s1 = await aio_session.get_shared_aio_session()
        await aio_session.close_shared_aio_session()
        assert s1.closed

        s2 = await aio_session.get_shared_aio_session()
        try:
            assert s2 is not s1
            assert not s2.closed
        finally:
            await aio_session.close_shared_aio_session()

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        await aio_session.close_shared_aio_session()
        await aio_session.close_shared_aio_session()


class TestSessionPerLoop:
    """Test that different event loops get different sessions."""

    def setup_method(self):
        aio_session._shared_ssl_context = None
        aio_session._aio_sessions.clear()

    def test_different_loops_get_different_sessions(self):
        sessions = []

        def run_in_loop():
            loop = asyncio.new_event_loop()
            try:
                session = loop.run_until_complete(
                    aio_session.get_shared_aio_session(),
                )
                sessions.append(session)
                # Keep loop open so session stays valid
                loop.run_until_complete(
                    aio_session.close_shared_aio_session(),
                )
            finally:
                loop.close()

        run_in_loop()
        run_in_loop()

        # Each loop should have gotten its own session
        assert len(sessions) == 2
        assert sessions[0] is not sessions[1]
