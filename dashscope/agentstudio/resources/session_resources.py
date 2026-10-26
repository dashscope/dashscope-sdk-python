# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Session Resources runtime CRUD.

Mount (add), list, get, and delete files mounted into a running session.
Uses ``add`` rather than ``create``; ``update`` is not exposed.
"""

from __future__ import annotations

from typing import Optional

from dashscope.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
    build_page,
)
from dashscope.agentstudio.resources._helpers import (
    _coerce_session_resource,
    _session_resource_item_path,
    _session_resources_path,
)
from dashscope.agentstudio.types import DeleteResponse, SessionResource
from dashscope.agentstudio.types.params import (
    SessionResourceAddParams,
    SessionResourceListParams,
)


class SessionResources:
    """Runtime resource mount/unmount on a session."""

    def __init__(self, client) -> None:
        self._client = client

    def add(
        self,
        session_id: str,
        *,
        resource_type: str = "file",
        file_id: str,
        mount_path: Optional[str] = None,
    ) -> SessionResource:
        """Mount a file into a running session (``POST /sessions/{id}/resources``).

        ``mount_path`` must be an absolute path under ``/uploads/``. The
        returned ``file_id`` is the session-scoped copy id and may differ
        from the source ``file_id`` passed in.
        """
        body = SessionResourceAddParams(
            type=resource_type,
            file_id=file_id,
            mount_path=mount_path,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            _session_resources_path(session_id),
            json=body,
        )
        return _coerce_session_resource(resp.data)

    def retrieve(
        self,
        session_id: str,
        resource_id: str,
    ) -> SessionResource:
        resp = self._client.transport.request(
            "GET",
            _session_resource_item_path(session_id, resource_id),
        )
        return _coerce_session_resource(resp.data)

    # Alias: get() delegates to retrieve(), matching the SDK convention.
    get = retrieve  # type: ignore[assignment]

    def list(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> CursorPage[SessionResource]:
        params = SessionResourceListParams(limit=limit, page=page).to_dict()
        resp = self._client.transport.request(
            "GET",
            _session_resources_path(session_id),
            params=params,
        )

        def fetch_next(token: str) -> CursorPage[SessionResource]:
            return self.list(session_id, limit=limit, page=token)

        return build_page(
            payload=resp.data,
            item_factory=_coerce_session_resource,
            request_id=resp.request_id,
            fetch_next=fetch_next,
        )

    def delete(
        self,
        session_id: str,
        resource_id: str,
    ) -> DeleteResponse:
        resp = self._client.transport.request(
            "DELETE",
            _session_resource_item_path(session_id, resource_id),
        )
        return DeleteResponse(**resp.data)


class AsyncSessionResources:
    """Async runtime resource mount/unmount on a session."""

    def __init__(self, client) -> None:
        self._client = client

    async def add(
        self,
        session_id: str,
        *,
        resource_type: str = "file",
        file_id: str,
        mount_path: Optional[str] = None,
    ) -> SessionResource:
        body = SessionResourceAddParams(
            type=resource_type,
            file_id=file_id,
            mount_path=mount_path,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            _session_resources_path(session_id),
            json=body,
        )
        return _coerce_session_resource(resp.data)

    async def retrieve(
        self,
        session_id: str,
        resource_id: str,
    ) -> SessionResource:
        resp = await self._client.transport.request(
            "GET",
            _session_resource_item_path(session_id, resource_id),
        )
        return _coerce_session_resource(resp.data)

    # Alias: get() delegates to retrieve().
    get = retrieve  # type: ignore[assignment]

    async def list(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[SessionResource]:
        params = SessionResourceListParams(limit=limit, page=page).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _session_resources_path(session_id),
            params=params,
        )

        async def fetch_next(token: str) -> AsyncCursorPage[SessionResource]:
            return await self.list(session_id, limit=limit, page=token)

        return build_page(
            payload=resp.data,
            item_factory=_coerce_session_resource,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def delete(
        self,
        session_id: str,
        resource_id: str,
    ) -> DeleteResponse:
        resp = await self._client.transport.request(
            "DELETE",
            _session_resource_item_path(session_id, resource_id),
        )
        return DeleteResponse(**resp.data)
