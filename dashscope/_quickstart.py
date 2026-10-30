# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Module-level quickstart entry points backed by a managed Client."""

from __future__ import annotations

import atexit
import threading
from typing import Any, Dict, Optional, Set

from dashscope._config import Config, resolve_config
from dashscope.client import Client
from dashscope.types.completions import ChatCompletion

__all__ = ["chat"]

_lock = threading.Lock()
_client: Optional[Client] = None
_active_clients: Dict[Client, int] = {}
_retired_clients: Set[Client] = set()


def _new_client(config: Config) -> Client:
    """Construct the quickstart client from one resolved config snapshot."""
    return Client(
        api_key=config.api_key,
        base_url=config.configured_base_url,
        workspace=config.workspace,
        region=config.region,
        protocol=config.protocol,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def _safe_close(client: Client) -> None:
    try:
        client.close()
    except Exception:
        # Cleanup must never hide the result of the user's chat call or make
        # interpreter shutdown noisy.
        pass


def _select_client(*, acquire: bool) -> Client:
    """Return a client for the current settings, replacing stale instances."""
    global _client
    desired = resolve_config()
    close_after_unlock: Optional[Client] = None
    with _lock:
        current = _client
        if current is None or current.config != desired:
            replacement = _new_client(desired)
            _client = replacement
            if current is not None:
                if _active_clients.get(current, 0):
                    _retired_clients.add(current)
                else:
                    close_after_unlock = current
            current = replacement
        if acquire:
            _active_clients[current] = _active_clients.get(current, 0) + 1
    if close_after_unlock is not None:
        _safe_close(close_after_unlock)
    return current


def _get_client() -> Client:
    """Return the configuration-aware singleton (private test hook)."""
    return _select_client(acquire=False)


def _release_client(client: Client) -> None:
    close_after_unlock = False
    with _lock:
        active = _active_clients.get(client, 0)
        if active <= 1:
            _active_clients.pop(client, None)
            if client in _retired_clients:
                _retired_clients.remove(client)
                close_after_unlock = True
        else:
            _active_clients[client] = active - 1
    if close_after_unlock:
        _safe_close(client)


def _close_clients() -> None:
    """Best-effort cleanup for the hidden quickstart connection pools."""
    global _client
    with _lock:
        clients = set(_active_clients) | _retired_clients
        if _client is not None:
            clients.add(_client)
        _client = None
        _active_clients.clear()
        _retired_clients.clear()
    for client in clients:
        _safe_close(client)


def chat(model: str, prompt: str, **kwargs: Any) -> ChatCompletion:
    """One-call chat entry: ``dashscope.chat("qwen-plus", "Tell me a joke")``.

    Extra keyword arguments are forwarded to
    ``client.chat.completions.create``. Global settings and environment
    changes are re-resolved before each call; an in-flight call keeps its
    previous client alive until that call finishes.
    """
    client = _select_client(acquire=True)
    try:
        return client.chat.completions.create(
            model=model,
            messages=[prompt],
            **kwargs,
        )
    finally:
        _release_client(client)


atexit.register(_close_clients)
