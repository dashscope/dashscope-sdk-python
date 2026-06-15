# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared aiohttp session pool with cached SSL context.

Provides connection reuse across async API calls. Each event loop gets
its own ClientSession (aiohttp sessions are loop-bound). The SSL context
is created once and shared across all sessions.
"""
import asyncio
import ssl
import weakref
from typing import Optional

import aiohttp
import certifi

_shared_ssl_context: Optional[ssl.SSLContext] = None
_aio_sessions: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def get_ssl_context() -> ssl.SSLContext:
    global _shared_ssl_context
    if _shared_ssl_context is None:
        _shared_ssl_context = ssl.create_default_context(
            cafile=certifi.where(),
        )
    return _shared_ssl_context


async def get_shared_aio_session() -> aiohttp.ClientSession:
    """Return a shared aiohttp.ClientSession bound to the running event loop.

    The session is lazily created on first use and reused for all
    subsequent calls on the same event loop. Connection pooling (keep-alive)
    is handled by the underlying TCPConnector.
    """
    loop = asyncio.get_running_loop()

    session = _aio_sessions.get(loop)
    if session is not None and not session.closed:
        return session

    connector = aiohttp.TCPConnector(ssl=get_ssl_context())
    session = aiohttp.ClientSession(connector=connector)
    _aio_sessions[loop] = session
    return session


async def close_shared_aio_session() -> None:
    """Close the shared session for the current event loop."""
    loop = asyncio.get_running_loop()
    session = _aio_sessions.pop(loop, None)
    if session is not None and not session.closed:
        await session.close()
