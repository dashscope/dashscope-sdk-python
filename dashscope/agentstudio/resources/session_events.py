# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""SessionEvents and AsyncSessionEvents resource classes."""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Sequence,
)

from dashscope.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
    build_page,
)
from dashscope.agentstudio.resources._helpers import (
    _coerce_event,
    _events_path,
    _stream_path,
)
from dashscope.agentstudio.streaming import AsyncEventStream, EventStream
from dashscope.agentstudio.types import ServerEvent
from dashscope.agentstudio.types.params import (
    SessionEventListParams,
    SessionEventSendParams,
)
from dashscope.agentstudio.constants import (
    AGENTSTUDIO_DEFAULT_TIMEOUT,
    BlockType,
    SessionStatus,
    SSEEventType,
)


class SessionEvents:
    """Session event send / list / stream."""

    def __init__(self, client) -> None:
        self._client = client

    def send(
        self,
        session_id: str,
        events: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Send events to a session.

        Returns the server response dict::

            result = client.sessions.events.send(
                session.id, [user_message("hello")],
            )
        """
        if not events:
            raise ValueError("events must contain at least 1 entry")
        body = SessionEventSendParams(input=events).to_dict()
        resp = self._client.transport.request(
            "POST",
            _events_path(session_id),
            json=body,
        )
        return resp.data

    def list(
        self,
        session_id: str,
        *,
        types: Optional[Sequence[str]] = None,
        created_at_gt: Optional[str] = None,
        created_at_gte: Optional[str] = None,
        created_at_lt: Optional[str] = None,
        created_at_lte: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        page: Optional[str] = None,
    ) -> CursorPage[ServerEvent]:
        params = SessionEventListParams(
            types=types,
            created_at_gt=created_at_gt,
            created_at_gte=created_at_gte,
            created_at_lt=created_at_lt,
            created_at_lte=created_at_lte,
            limit=limit,
            order=order,
            page=page,
        ).to_dict()
        resp = self._client.transport.request(
            "GET",
            _events_path(session_id),
            params=params,
        )

        def fetch_next(nxt: str) -> CursorPage[ServerEvent]:
            return self.list(
                session_id,
                types=types,
                created_at_gt=created_at_gt,
                created_at_gte=created_at_gte,
                created_at_lt=created_at_lt,
                created_at_lte=created_at_lte,
                limit=limit,
                order=order,
                page=nxt,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_event,
            request_id=resp.request_id,
            fetch_next=fetch_next,
        )

    def stream(
        self,
        session_id: str,
        *,
        event_deltas: Optional[Sequence[str]] = None,
        timeout: Optional[float] = None,
    ) -> "_TypedEventStream":
        """Open the SSE stream and return an iterator of typed events.

        ``event_deltas`` opts into incremental text streaming for the given
        event types (``"message"`` and/or ``"reasoning"``; aliases
        ``"agent.message"`` / ``"agent.thinking"``). When set, the stream
        additionally emits ``event_start`` / ``event_delta`` frames carrying
        partial text — consume them via :attr:`_TypedEventStream.text_deltas`.
        Other values are rejected by the server (HTTP 400).
        """
        params: Optional[Dict[str, Any]] = None
        if event_deltas:
            params = {"event_deltas[]": list(event_deltas)}
        resp = self._client.transport.request(
            "GET",
            _stream_path(session_id),
            extra_headers={"Accept": "text/event-stream"},
            params=params,
            stream=True,
            timeout=timeout or AGENTSTUDIO_DEFAULT_TIMEOUT,
        )
        return _TypedEventStream(
            EventStream(response=resp),
        )


class _TypedEventStream:
    """Wrap :class:`EventStream` so iterating yields :class:`ServerEvent`."""

    def __init__(self, stream: EventStream) -> None:
        self._stream = stream

    def __enter__(self) -> "_TypedEventStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[ServerEvent]:
        for payload in self._stream:
            yield _coerce_event(payload)

    @property
    def text_stream(self):
        """Iterate over text chunks from agent messages.

        Automatically stops when the session reaches ``idle`` or
        ``terminated`` status, so callers don't need to handle
        ``session_status`` events manually.
        """
        for event in self:
            if getattr(event, "type", None) == SSEEventType.MESSAGE:
                for block in event.content or []:
                    if getattr(block, "type", None) == BlockType.TEXT:
                        text = getattr(block, "text", "")
                        if text:
                            yield text
            elif getattr(event, "type", None) == SSEEventType.SESSION_STATUS:
                block = event.content[0] if event.content else None
                d = getattr(block, "data", None) or {}
                if d.get("session_status") in (
                    SessionStatus.IDLE,
                    SessionStatus.TERMINATED,
                    SessionStatus.RESCHEDULED,
                    SessionStatus.DELETED,
                ):
                    return

    @property
    def text_deltas(self):
        """Iterate over incremental text chunks from ``event_delta`` frames.

        Requires the stream to be opened with ``event_deltas`` (e.g.
        ``event_deltas=["message"]``); otherwise the server emits only
        terminal messages and this yields nothing (use
        :attr:`text_stream` for terminal full text). Stops on terminal
        ``session_status`` like :attr:`text_stream`.
        """
        for event in self:
            etype = getattr(event, "type", None)
            if etype == "event_delta":
                text = event.delta_text
                if text:
                    yield text
            elif etype == SSEEventType.SESSION_STATUS:
                block = event.content[0] if event.content else None
                d = getattr(block, "data", None) or {}
                if d.get("session_status") in (
                    SessionStatus.IDLE,
                    SessionStatus.TERMINATED,
                    SessionStatus.RESCHEDULED,
                    SessionStatus.DELETED,
                ):
                    return

    def close(self) -> None:
        self._stream.close()


class AsyncSessionEvents:
    """Async session event operations."""

    def __init__(self, client) -> None:
        self._client = client

    async def send(
        self,
        session_id: str,
        events: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Send events to a session.

        Returns the server response dict::

            result = await client.sessions.events.send(
                session.id, [user_message("hello")],
            )
        """
        if not events:
            raise ValueError("events must contain at least 1 entry")
        body = SessionEventSendParams(input=events).to_dict()
        resp = await self._client.transport.request(
            "POST",
            _events_path(session_id),
            json=body,
        )
        return resp.data

    async def list(
        self,
        session_id: str,
        *,
        types: Optional[Sequence[str]] = None,
        created_at_gt: Optional[str] = None,
        created_at_gte: Optional[str] = None,
        created_at_lt: Optional[str] = None,
        created_at_lte: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[ServerEvent]:
        params = SessionEventListParams(
            types=types,
            created_at_gt=created_at_gt,
            created_at_gte=created_at_gte,
            created_at_lt=created_at_lt,
            created_at_lte=created_at_lte,
            limit=limit,
            order=order,
            page=page,
        ).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _events_path(session_id),
            params=params,
        )

        async def fetch_next(
            nxt: str,
        ) -> AsyncCursorPage[ServerEvent]:
            return await self.list(
                session_id,
                types=types,
                created_at_gt=created_at_gt,
                created_at_gte=created_at_gte,
                created_at_lt=created_at_lt,
                created_at_lte=created_at_lte,
                limit=limit,
                order=order,
                page=nxt,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_event,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def stream(
        self,
        session_id: str,
        *,
        event_deltas: Optional[Sequence[str]] = None,
        timeout: Optional[float] = None,
    ) -> "_AioTypedEventStream":
        """Open the SSE stream and return an async iterator of typed events.

        See :meth:`SessionEvents.stream` for the ``event_deltas`` opt-in
        for incremental text streaming.
        """
        params: Optional[Dict[str, Any]] = None
        if event_deltas:
            params = {"event_deltas[]": list(event_deltas)}
        resp = await self._client.transport.request(
            "GET",
            _stream_path(session_id),
            extra_headers={"Accept": "text/event-stream"},
            params=params,
            stream=True,
            timeout=timeout or AGENTSTUDIO_DEFAULT_TIMEOUT,
        )
        return _AioTypedEventStream(
            AsyncEventStream(response=resp),
        )


class _AioTypedEventStream:
    def __init__(self, stream: AsyncEventStream) -> None:
        self._stream = stream

    async def __aenter__(self) -> "_AioTypedEventStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __aiter__(self) -> AsyncIterator[ServerEvent]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[ServerEvent]:
        async for payload in self._stream:
            yield _coerce_event(payload)

    @property
    def text_stream(self):
        """Async iterator over text chunks from agent messages."""
        return self._text_stream()

    async def _text_stream(self):
        """Async iterator over text chunks from agent messages.

        Automatically stops when the session reaches ``idle`` or
        ``terminated`` status, so callers don't need to handle
        ``session_status`` events manually.
        """
        async for event in self:
            if getattr(event, "type", None) == SSEEventType.MESSAGE:
                for block in event.content or []:
                    if getattr(block, "type", None) == BlockType.TEXT:
                        text = getattr(block, "text", "")
                        if text:
                            yield text
            elif getattr(event, "type", None) == SSEEventType.SESSION_STATUS:
                block = event.content[0] if event.content else None
                d = getattr(block, "data", None) or {}
                if d.get("session_status") in (
                    SessionStatus.IDLE,
                    SessionStatus.TERMINATED,
                    SessionStatus.RESCHEDULED,
                    SessionStatus.DELETED,
                ):
                    return

    @property
    def text_deltas(self):
        """Async iterator over incremental text from ``event_delta`` frames.

        Requires the stream to be opened with ``event_deltas``; otherwise
        yields nothing (use :attr:`text_stream` for terminal full text).
        """
        return self._text_deltas()

    async def _text_deltas(self):
        async for event in self:
            etype = getattr(event, "type", None)
            if etype == "event_delta":
                text = event.delta_text
                if text:
                    yield text
            elif etype == SSEEventType.SESSION_STATUS:
                block = event.content[0] if event.content else None
                d = getattr(block, "data", None) or {}
                if d.get("session_status") in (
                    SessionStatus.IDLE,
                    SessionStatus.TERMINATED,
                    SessionStatus.RESCHEDULED,
                    SessionStatus.DELETED,
                ):
                    return

    async def aclose(self) -> None:
        await self._stream.aclose()
