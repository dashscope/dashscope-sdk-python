# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""SSE (Server-Sent Events) stream iterators backed by ``httpx-sse``.

The AgentStudio stream is documented as::

    event: message
    data: <json>

    : keepalive

The ``event:`` field is always the literal ``"message"`` and never used
for routing – the actual event type lives in ``data.type``. The server
does not emit ``id:`` lines; reconnection is performed via
``GET /sessions/{id}/events`` with ``created_at[gt]=...`` plus
client-side dedup by ``data.id``.

This module wraps ``httpx_sse.EventSource`` in iterator classes that
yield parsed JSON dicts for consumption by the typed event stream layer.

A wall-clock ``idle_timeout`` (default 120 s) guards against the server
silently dropping business events while still sending ``: keepalive``
frames — ``httpx``'s read timeout gets reset by every keepalive, so
without an application-level watchdog the iterator would block forever.
When no business event arrives for ``idle_timeout`` seconds, the
underlying response is forcibly closed so ``iter_sse()`` unblocks.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import httpx
from httpx_sse import EventSource

from dashscope.agentstudio import exceptions
from dashscope.common.logging import logger

_DEFAULT_IDLE_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# Iterators
# ---------------------------------------------------------------------------


@dataclass
class EventStream:
    """Synchronous iterator over an SSE response backed by ``httpx-sse``.

    Emits dict payloads (the parsed ``data`` JSON of each frame). Comment
    / keepalive frames are silently skipped. The underlying response is
    closed when iteration completes or :meth:`close` is invoked.

    A wall-clock ``idle_timeout`` guards against the server silently
    dropping business events while still sending ``: keepalive`` frames.
    ``httpx``'s read timeout gets reset by every keepalive, and
    ``response.close()`` does not unblock a blocking ``iter_sse()``
    call, so the iterator is drained on a background thread while the
    foreground loop pulls events through a queue with timeout.
    """

    response: httpx.Response
    idle_timeout: Optional[float] = _DEFAULT_IDLE_TIMEOUT
    _event_source: EventSource = field(init=False)
    _closed: bool = False
    _producer: Optional[threading.Thread] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._event_source = EventSource(self.response)

    def __enter__(self) -> "EventStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self._iter()

    def _iter(self) -> Iterator[Dict[str, Any]]:
        if self._closed:
            raise exceptions.StreamClosedError("stream already closed")
        q: "queue.Queue[Any]" = queue.Queue()

        def _produce() -> None:
            try:
                for sse in self._event_source.iter_sse():
                    if sse.data:
                        try:
                            payload = json.loads(sse.data)
                        except json.JSONDecodeError:
                            logger.warning("SSE frame contains invalid JSON")
                            q.put({"_raw": sse.data})
                            continue
                        if payload:
                            q.put(payload)
            except Exception as exc:  # noqa: W0718
                q.put(exc)
            finally:
                q.put(None)

        self._producer = threading.Thread(target=_produce, daemon=True)
        self._producer.start()

        try:
            while True:
                try:
                    item = q.get(timeout=self.idle_timeout)
                except queue.Empty:
                    logger.warning(
                        "SSE stream idle for %ss, closing response",
                        self.idle_timeout,
                    )
                    self.close()
                    return
                if item is None:
                    return
                if isinstance(item, Exception):
                    if isinstance(item, exceptions.AgentStudioError):
                        raise item
                    raise exceptions.StreamError(
                        exceptions.describe_exception(item),
                    ) from item
                yield item
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.response.close()
        except Exception:  # pragma: no cover
            pass


@dataclass
class AsyncEventStream:
    """Async iterator over an SSE response backed by ``httpx-sse``.

    See :class:`EventStream` for the ``idle_timeout`` rationale — the
    async variant drains ``aiter_sse()`` on a background task and pulls
    events through an :class:`asyncio.Queue` with ``wait_for`` timeout.
    """

    response: httpx.Response
    idle_timeout: Optional[float] = _DEFAULT_IDLE_TIMEOUT
    _event_source: EventSource = field(init=False)
    _closed: bool = False
    _producer: Optional[asyncio.Task] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._event_source = EventSource(self.response)

    async def __aenter__(self) -> "AsyncEventStream":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[Dict[str, Any]]:
        if self._closed:
            raise exceptions.StreamClosedError("stream already closed")
        q: "asyncio.Queue[Any]" = asyncio.Queue()

        async def _produce() -> None:
            try:
                async for sse in self._event_source.aiter_sse():
                    if sse.data:
                        try:
                            payload = json.loads(sse.data)
                        except json.JSONDecodeError:
                            logger.warning("SSE frame contains invalid JSON")
                            await q.put({"_raw": sse.data})
                            continue
                        if payload:
                            await q.put(payload)
            except Exception as exc:  # noqa: W0718
                await q.put(exc)
            finally:
                await q.put(None)

        self._producer = asyncio.create_task(_produce())

        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        q.get(),
                        timeout=self.idle_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "SSE stream idle for %ss, closing response",
                        self.idle_timeout,
                    )
                    await self.aclose()
                    return
                if item is None:
                    return
                if isinstance(item, Exception):
                    if isinstance(item, exceptions.AgentStudioError):
                        raise item
                    raise exceptions.StreamError(
                        exceptions.describe_exception(item),
                    ) from item
                yield item
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._producer is not None:
            self._producer.cancel()
            try:
                await self._producer
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await self.response.aclose()
        except Exception:  # pragma: no cover
            pass
