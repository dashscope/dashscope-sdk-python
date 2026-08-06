# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared aiohttp session pool with cached SSL context.

Provides connection reuse across async API calls. Each event loop gets
its own ClientSession (aiohttp sessions are loop-bound). The SSL context
is created once and shared across all sessions.
"""
import asyncio
import atexit
import ssl
import threading
import weakref
from typing import Optional

import aiohttp
import certifi

_shared_ssl_context: Optional[ssl.SSLContext] = None
_aio_sessions: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_lock = threading.RLock()


def get_ssl_context() -> ssl.SSLContext:
    global _shared_ssl_context
    with _lock:
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

    with _lock:
        session = _aio_sessions.get(loop)
        if session is not None and not session.closed:
            return session

        connector = aiohttp.TCPConnector(ssl=get_ssl_context())
        session = aiohttp.ClientSession(connector=connector, trust_env=True)

        _aio_sessions[loop] = session
        # Register GC-safe finalizer to close session when loop is collected
        weakref.finalize(session, _sync_close_session, id(session))
    return session


def _sync_close_session(session_id: int) -> None:
    """GC callback to safely close a session outside async context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            return
        if loop.is_running():
            asyncio.ensure_future(_async_close_by_id(session_id))
        else:
            loop.run_until_complete(_async_close_by_id(session_id))
    except RuntimeError:
        pass


async def _async_close_by_id(session_id: int) -> None:
    """Close session by ID, safe against already-collected sessions."""
    with _lock:
        for loop, session in list(_aio_sessions.items()):
            if id(session) == session_id and not session.closed:
                await session.close()
                _aio_sessions.pop(loop, None)
                return


def _atexit_cleanup() -> None:
    """Cleanup all sessions at interpreter exit."""
    with _lock:
        sessions = list(_aio_sessions.items())
        _aio_sessions.clear()

    for loop, session in sessions:
        if not session.closed:
            try:
                if not loop.is_closed() and not loop.is_running():
                    loop.run_until_complete(session.close())
            except Exception:
                pass


# Register atexit handler
atexit.register(_atexit_cleanup)


async def close_shared_aio_session() -> None:
    """Close the shared session for the current event loop."""
    loop = asyncio.get_running_loop()
    with _lock:
        session = _aio_sessions.pop(loop, None)
    if session is not None and not session.closed:
        await session.close()
