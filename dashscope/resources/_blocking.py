# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Run the blocking upload path on behalf of an async resource.

Local-file → OSS upload is written once, synchronously: it fetches the
upload policy over the client transport, then posts the file with a
blocking httpx client. Async resources offload that work with
``asyncio.to_thread``, but a worker thread cannot ``await`` the
``AsyncTransport`` — calling it there returns an un-awaited coroutine
instead of a response.

So the worker gets a *bridge* client: same client for everything except the
transport, whose ``request`` schedules the coroutine back onto the running
loop with ``run_coroutine_threadsafe`` and blocks on the future. The upload
keeps the async client's connection pool, retries and per-call credentials,
and the event loop stays free to run the coroutine it was handed.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

__all__ = ["bridge_client", "run_blocking"]


class _BridgeTransport:
    def __init__(self, transport, loop: asyncio.AbstractEventLoop) -> None:
        self._transport = transport
        self._loop = loop

    def request(self, *args, **kwargs):
        return asyncio.run_coroutine_threadsafe(
            self._transport.request(*args, **kwargs),
            self._loop,
        ).result()


class _BridgeClient:
    """Stand-in client whose ``transport`` answers synchronously."""

    def __init__(self, client, loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self.transport = _BridgeTransport(client.transport, loop)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def bridge_client(client) -> Any:
    """Wrap ``client`` so its transport can be used from a worker thread.

    Must be called from the event loop that owns the client's transport;
    the loop is captured here because a worker thread has no way to find it.
    """
    return _BridgeClient(client, asyncio.get_running_loop())


async def run_blocking(
    client,
    fn: Callable[..., Any],
    *args,
    **kwargs,
) -> Any:
    """``asyncio.to_thread(fn, <bridged client>, *args, **kwargs)``.

    ``fn`` takes the client as its first argument, as every blocking helper
    on the upload path does.
    """
    return await asyncio.to_thread(fn, bridge_client(client), *args, **kwargs)
