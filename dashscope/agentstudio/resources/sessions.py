# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Sessions resource class."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashscope.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
    build_page,
)
from dashscope.agentstudio.resources._helpers import (
    _coerce_session,
)
from dashscope.agentstudio.resources.session_events import (
    SessionEvents,
    AsyncSessionEvents,
)
from dashscope.agentstudio.types import DeleteResponse, Session
from dashscope.agentstudio.types.params import (
    SessionCreateParams,
    SessionListParams,
    SessionUpdateParams,
)


_PATH_SESSIONS = "/sessions"


class Sessions:
    """Session CRUD — instance methods backed by a shared client."""

    def __init__(self, client) -> None:
        self._client = client
        self.events = SessionEvents(client)

    def create(
        self,
        *,
        agent: str,
        environment_id: Optional[str] = None,
        title: Optional[str] = None,
        resources: Optional[Sequence[Mapping[str, Any]]] = None,
        vault_ids: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Session:
        body = SessionCreateParams(
            agent=agent,
            environment_id=environment_id,
            title=title,
            resources=resources,
            vault_ids=vault_ids,
            metadata=metadata,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            _PATH_SESSIONS,
            json=body,
        )
        return _coerce_session(resp.data)

    def retrieve(self, session_id: str) -> Session:
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_SESSIONS}/{session_id}",
        )
        return _coerce_session(resp.data)

    # Alias: get() delegates to retrieve()
    get = retrieve  # type: ignore[assignment]

    # Alias: call() delegates to create()
    call = create  # type: ignore[assignment]

    def update(
        self,
        session_id: str,
        *,
        title: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Session:
        body = SessionUpdateParams(
            title=title,
            metadata=metadata,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_SESSIONS}/{session_id}",
            json=body,
        )
        return _coerce_session(resp.data)

    def list(
        self,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        agent_id: Optional[str] = None,
        statuses: Optional[Sequence[str]] = None,
        created_at_gt: Optional[str] = None,
        created_at_gte: Optional[str] = None,
        created_at_lt: Optional[str] = None,
        created_at_lte: Optional[str] = None,
    ) -> CursorPage[Session]:
        params = SessionListParams(
            limit=limit,
            page=page,
            agent_id=agent_id,
            statuses=statuses,
            created_at_gt=created_at_gt,
            created_at_gte=created_at_gte,
            created_at_lt=created_at_lt,
            created_at_lte=created_at_lte,
        ).to_dict()
        resp = self._client.transport.request(
            "GET",
            _PATH_SESSIONS,
            params=params,
        )

        def fetch_next(nxt: str) -> CursorPage[Session]:
            return self.list(
                limit=limit,
                page=nxt,
                agent_id=agent_id,
                statuses=statuses,
                created_at_gt=created_at_gt,
                created_at_gte=created_at_gte,
                created_at_lt=created_at_lt,
                created_at_lte=created_at_lte,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_session,
            request_id=resp.request_id,
            fetch_next=fetch_next,
        )

    def archive(self, session_id: str) -> Session:
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_SESSIONS}/{session_id}/archive",
        )
        return _coerce_session(resp.data)

    def delete(self, session_id: str) -> DeleteResponse:
        resp = self._client.transport.request(
            "DELETE",
            f"{_PATH_SESSIONS}/{session_id}",
        )
        return DeleteResponse(**resp.data)


class AsyncSessions:
    """Async Session CRUD — instance methods backed by a shared client."""

    def __init__(self, client) -> None:
        self._client = client
        self.events = AsyncSessionEvents(client)

    async def create(
        self,
        *,
        agent: str,
        environment_id: Optional[str] = None,
        title: Optional[str] = None,
        resources: Optional[Sequence[Mapping[str, Any]]] = None,
        vault_ids: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Session:
        body = SessionCreateParams(
            agent=agent,
            environment_id=environment_id,
            title=title,
            resources=resources,
            vault_ids=vault_ids,
            metadata=metadata,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            _PATH_SESSIONS,
            json=body,
        )
        return _coerce_session(resp.data)

    async def retrieve(self, session_id: str) -> Session:
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_SESSIONS}/{session_id}",
        )
        return _coerce_session(resp.data)

    # Alias: get() delegates to retrieve()
    get = retrieve  # type: ignore[assignment]

    # Alias: call() delegates to create()
    call = create  # type: ignore[assignment]

    async def update(
        self,
        session_id: str,
        *,
        title: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Session:
        body = SessionUpdateParams(
            title=title,
            metadata=metadata,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_SESSIONS}/{session_id}",
            json=body,
        )
        return _coerce_session(resp.data)

    async def list(
        self,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        agent_id: Optional[str] = None,
        statuses: Optional[Sequence[str]] = None,
        created_at_gt: Optional[str] = None,
        created_at_gte: Optional[str] = None,
        created_at_lt: Optional[str] = None,
        created_at_lte: Optional[str] = None,
    ) -> AsyncCursorPage[Session]:
        params = SessionListParams(
            limit=limit,
            page=page,
            agent_id=agent_id,
            statuses=statuses,
            created_at_gt=created_at_gt,
            created_at_gte=created_at_gte,
            created_at_lt=created_at_lt,
            created_at_lte=created_at_lte,
        ).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _PATH_SESSIONS,
            params=params,
        )

        async def fetch_next(nxt: str) -> AsyncCursorPage[Session]:
            return await self.list(
                limit=limit,
                page=nxt,
                agent_id=agent_id,
                statuses=statuses,
                created_at_gt=created_at_gt,
                created_at_gte=created_at_gte,
                created_at_lt=created_at_lt,
                created_at_lte=created_at_lte,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_session,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def archive(self, session_id: str) -> Session:
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_SESSIONS}/{session_id}/archive",
        )
        return _coerce_session(resp.data)

    async def delete(self, session_id: str) -> DeleteResponse:
        resp = await self._client.transport.request(
            "DELETE",
            f"{_PATH_SESSIONS}/{session_id}",
        )
        return DeleteResponse(**resp.data)
