# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Environments resource class."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from dashscope.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
    build_page,
)
from dashscope.agentstudio.resources._helpers import (
    _coerce_env,
)
from dashscope.agentstudio.types import DeleteResponse, Environment
from dashscope.agentstudio.types.params import (
    EnvironmentCreateParams,
    EnvironmentListParams,
    EnvironmentUpdateParams,
)


_PATH_ENVIRONMENTS = "/environments"


class Environments:
    """Environment CRUD — instance methods backed by a shared client."""

    def __init__(self, client) -> None:
        self._client = client

    def create(
        self,
        *,
        name: str,
        config: Mapping[str, Any],
        description: Optional[str] = None,
        scope: Optional[str] = "organization",
        metadata: Optional[Mapping[str, Any]] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> Environment:
        body = EnvironmentCreateParams(
            name=name,
            config=config,
            description=description,
            scope=scope,
            metadata=metadata,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            _PATH_ENVIRONMENTS,
            json=body,
            extra_headers=extra_headers,
        )
        return _coerce_env(resp.data)

    def retrieve(self, environment_id: str) -> Environment:
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_ENVIRONMENTS}/{environment_id}",
        )
        return _coerce_env(resp.data)

    # Alias: get() delegates to retrieve()
    get = retrieve  # type: ignore[assignment]

    def update(
        self,
        environment_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Environment:
        body = EnvironmentUpdateParams(
            name=name,
            description=description,
            config=config,
            metadata=metadata,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_ENVIRONMENTS}/{environment_id}",
            json=body,
        )
        return _coerce_env(resp.data)

    def list(
        self,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        include_archived: Optional[bool] = None,
    ) -> CursorPage[Environment]:
        params = EnvironmentListParams(
            limit=limit,
            page=page,
            include_archived=include_archived,
        ).to_dict()
        resp = self._client.transport.request(
            "GET",
            _PATH_ENVIRONMENTS,
            params=params,
        )
        return build_page(
            payload=resp.data,
            item_factory=_coerce_env,
            request_id=resp.request_id,
            fetch_next=lambda nxt: self.list(
                limit=limit,
                page=nxt,
                include_archived=include_archived,
            ),
        )

    def archive(self, environment_id: str) -> Environment:
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_ENVIRONMENTS}/{environment_id}/archive",
        )
        return _coerce_env(resp.data)

    def delete(self, environment_id: str) -> DeleteResponse:
        resp = self._client.transport.request(
            "DELETE",
            f"{_PATH_ENVIRONMENTS}/{environment_id}",
        )
        return DeleteResponse(**resp.data)


class AsyncEnvironments:
    """Async Environment CRUD — instance methods backed by a shared client."""

    def __init__(self, client) -> None:
        self._client = client

    async def create(
        self,
        *,
        name: str,
        config: Mapping[str, Any],
        description: Optional[str] = None,
        scope: Optional[str] = "organization",
        metadata: Optional[Mapping[str, Any]] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> Environment:
        body = EnvironmentCreateParams(
            name=name,
            config=config,
            description=description,
            scope=scope,
            metadata=metadata,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            _PATH_ENVIRONMENTS,
            json=body,
            extra_headers=extra_headers,
        )
        return _coerce_env(resp.data)

    async def retrieve(self, environment_id: str) -> Environment:
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_ENVIRONMENTS}/{environment_id}",
        )
        return _coerce_env(resp.data)

    # Alias: get() delegates to retrieve()
    get = retrieve  # type: ignore[assignment]

    async def update(
        self,
        environment_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Environment:
        body = EnvironmentUpdateParams(
            name=name,
            description=description,
            config=config,
            metadata=metadata,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_ENVIRONMENTS}/{environment_id}",
            json=body,
        )
        return _coerce_env(resp.data)

    async def list(
        self,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        include_archived: Optional[bool] = None,
    ) -> AsyncCursorPage[Environment]:
        params = EnvironmentListParams(
            limit=limit,
            page=page,
            include_archived=include_archived,
        ).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _PATH_ENVIRONMENTS,
            params=params,
        )

        async def fetch_next(nxt: str) -> AsyncCursorPage[Environment]:
            return await self.list(
                limit=limit,
                page=nxt,
                include_archived=include_archived,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_env,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def archive(self, environment_id: str) -> Environment:
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_ENVIRONMENTS}/{environment_id}/archive",
        )
        return _coerce_env(resp.data)

    async def delete(self, environment_id: str) -> DeleteResponse:
        resp = await self._client.transport.request(
            "DELETE",
            f"{_PATH_ENVIRONMENTS}/{environment_id}",
        )
        return DeleteResponse(**resp.data)
