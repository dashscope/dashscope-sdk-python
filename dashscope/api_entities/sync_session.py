# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared requests.Session pool with configurable connection pool size.

Provides connection reuse across synchronous API calls. A single process-wide
Session is shared across all threads (requests.Session is thread-safe).
"""
import atexit
import ssl
import threading
from typing import Optional

import certifi
import requests
from requests.adapters import HTTPAdapter

import dashscope

_shared_sync_session: Optional[requests.Session] = None
_shared_ssl_context: Optional[ssl.SSLContext] = None
_lock = threading.RLock()


def _get_ssl_context() -> ssl.SSLContext:
    """Get or create shared SSL context."""
    global _shared_ssl_context
    with _lock:
        if _shared_ssl_context is None:
            _shared_ssl_context = ssl.create_default_context(
                cafile=certifi.where(),
            )
    return _shared_ssl_context


def _create_session() -> requests.Session:
    """Create a new session with configured connection pool."""
    session = requests.Session()

    # Configure connection pool size
    pool_size = getattr(dashscope, "http_connection_pool_size", 20)

    # Create adapter with custom pool size and retry strategy
    adapter = HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=0,  # Don't retry by default, let user handle retries
    )

    # Mount adapter for both HTTP and HTTPS
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set SSL context
    session.verify = certifi.where()

    return session


def get_shared_sync_session() -> requests.Session:
    """Return a process-wide shared requests.Session.

    The session is lazily created on first use and reused for all
    subsequent calls. Connection pooling (keep-alive) is handled
    by the underlying urllib3 connection pool.

    The session is thread-safe and can be shared across threads.
    """
    global _shared_sync_session
    with _lock:
        if _shared_sync_session is None:
            _shared_sync_session = _create_session()
    return _shared_sync_session


def close_shared_sync_session() -> None:
    """Close the shared synchronous session.

    This is optional - the session will be closed automatically at
    interpreter exit. Call this if you need to explicitly release
    resources.
    """
    global _shared_sync_session
    with _lock:
        if _shared_sync_session is not None:
            _shared_sync_session.close()
            _shared_sync_session = None


def _atexit_cleanup() -> None:
    """Cleanup session at interpreter exit."""
    close_shared_sync_session()


# Register atexit handler
atexit.register(_atexit_cleanup)
