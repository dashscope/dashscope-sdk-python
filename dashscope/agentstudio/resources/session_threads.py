# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Session Threads sub-resource.

List, get, and archive sub-agent threads within a session, and list the
events scoped to one thread. Thread event streaming is not exposed.
"""

from __future__ import annotations

from typing import Optional, Sequence

from dashscope.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
    build_page,
)
from dashscope.agentstudio.resources._helpers import (
    _coerce_event,
    _coerce_session_thread,
    _session_thread_archive_path,
    _session_thread_events_path,
    _session_thread_item_path,
    _session_threads_path,
)
from dashscope.agentstudio.types import ServerEvent, SessionThread
from dashscope.agentstudio.types.params import (
    SessionEventListParams,
    SessionThreadListParams,
)


class SessionThreadEvents:
    """Events scoped to a single sub-agent thread (list only)."""

    def __init__(self, client) -> None:
        self._client = client

    def list(
        self,
        session_id: str,
        thread_id: str,
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
            _session_thread_events_path(session_id, thread_id),
            params=params,
        )

        def fetch_next(nxt: str) -> CursorPage[ServerEvent]:
            return self.list(
                session_id,
                thread_id,
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


class AsyncSessionThreadEvents:
    """Async events scoped to a single sub-agent thread (list only)."""

    def __init__(self, client) -> None:
        self._client = client

    async def list(
        self,
        session_id: str,
        thread_id: str,
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
            _session_thread_events_path(session_id, thread_id),
            params=params,
        )

        async def fetch_next(nxt: str) -> AsyncCursorPage[ServerEvent]:
            return await self.list(
                session_id,
                thread_id,
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


class SessionThreads:
    """Sub-agent threads within a session."""

    def __init__(self, client) -> None:
        self._client = client
        self.events = SessionThreadEvents(client)

    def list(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> CursorPage[SessionThread]:
        params = SessionThreadListParams(limit=limit, page=page).to_dict()
        resp = self._client.transport.request(
            "GET",
            _session_threads_path(session_id),
            params=params,
        )

        def fetch_next(token: str) -> CursorPage[SessionThread]:
            return self.list(session_id, limit=limit, page=token)

        return build_page(
            payload=resp.data,
            item_factory=_coerce_session_thread,
            request_id=resp.request_id,
            fetch_next=fetch_next,
        )

    def retrieve(self, session_id: str, thread_id: str) -> SessionThread:
        resp = self._client.transport.request(
            "GET",
            _session_thread_item_path(session_id, thread_id),
        )
        return _coerce_session_thread(resp.data)

    # Alias: get() delegates to retrieve().
    get = retrieve  # type: ignore[assignment]

    def archive(self, session_id: str, thread_id: str) -> SessionThread:
        resp = self._client.transport.request(
            "POST",
            _session_thread_archive_path(session_id, thread_id),
        )
        return _coerce_session_thread(resp.data)


class AsyncSessionThreads:
    """Async sub-agent threads within a session."""

    def __init__(self, client) -> None:
        self._client = client
        self.events = AsyncSessionThreadEvents(client)

    async def list(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[SessionThread]:
        params = SessionThreadListParams(limit=limit, page=page).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _session_threads_path(session_id),
            params=params,
        )

        async def fetch_next(token: str) -> AsyncCursorPage[SessionThread]:
            return await self.list(session_id, limit=limit, page=token)

        return build_page(
            payload=resp.data,
            item_factory=_coerce_session_thread,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def retrieve(
        self,
        session_id: str,
        thread_id: str,
    ) -> SessionThread:
        resp = await self._client.transport.request(
            "GET",
            _session_thread_item_path(session_id, thread_id),
        )
        return _coerce_session_thread(resp.data)

    # Alias: get() delegates to retrieve().
    get = retrieve  # type: ignore[assignment]

    async def archive(self, session_id: str, thread_id: str) -> SessionThread:
        resp = await self._client.transport.request(
            "POST",
            _session_thread_archive_path(session_id, thread_id),
        )
        return _coerce_session_thread(resp.data)
