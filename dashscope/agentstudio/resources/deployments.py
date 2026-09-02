# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Deployment and deployment-run resource classes."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashscope.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
    build_page,
)
from dashscope.agentstudio.resources._helpers import (
    _coerce_deployment,
    _coerce_deployment_run,
)
from dashscope.agentstudio.types import Deployment, DeploymentRun
from dashscope.agentstudio.types.params import (
    _NOT_GIVEN,
    DeploymentCreateParams,
    DeploymentListParams,
    DeploymentRunListParams,
    DeploymentUpdateParams,
)

_PATH_DEPLOYMENTS = "/deployments"
_PATH_DEPLOYMENT_RUNS = "/deployment_runs"


class Deployments:
    """Managed Agent deployment lifecycle and execution."""

    def __init__(self, client) -> None:
        self._client = client

    def create(
        self,
        *,
        name: str,
        agent: Any,
        initial_events: Sequence[Mapping[str, Any]],
        description: Optional[str] = None,
        environment_id: Optional[str] = None,
        schedule: Any = None,
        resources: Optional[Sequence[Any]] = None,
        vault_ids: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> Deployment:
        body = DeploymentCreateParams(
            name=name,
            agent=agent,
            initial_events=initial_events,
            description=description,
            environment_id=environment_id,
            schedule=schedule,
            resources=resources,
            vault_ids=vault_ids,
            metadata=metadata,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            _PATH_DEPLOYMENTS,
            json=body,
        )
        return _coerce_deployment(resp.data)

    def retrieve(self, deployment_id: str) -> Deployment:
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_DEPLOYMENTS}/{deployment_id}",
        )
        return _coerce_deployment(resp.data)

    get = retrieve  # type: ignore[assignment]

    def update(
        self,
        deployment_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        agent: Any = None,
        environment_id: Any = _NOT_GIVEN,
        schedule: Any = _NOT_GIVEN,
        initial_events: Optional[Sequence[Mapping[str, Any]]] = None,
        resources: Optional[Sequence[Any]] = None,
        vault_ids: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> Deployment:
        body = DeploymentUpdateParams(
            name=name,
            description=description,
            agent=agent,
            environment_id=environment_id,
            schedule=schedule,
            initial_events=initial_events,
            resources=resources,
            vault_ids=vault_ids,
            metadata=metadata,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_DEPLOYMENTS}/{deployment_id}",
            json=body,
        )
        return _coerce_deployment(resp.data)

    def list(
        self,
        *,
        agent_id: Optional[str] = None,
        keyword: Optional[str] = None,
        status: Optional[str] = None,
        include_archived: Optional[bool] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        created_at_gte: Optional[str] = None,
        created_at_lte: Optional[str] = None,
    ) -> CursorPage[Deployment]:
        params = DeploymentListParams(
            agent_id=agent_id,
            keyword=keyword,
            status=status,
            include_archived=include_archived,
            limit=limit,
            page=page,
            created_at_gte=created_at_gte,
            created_at_lte=created_at_lte,
        ).to_dict()
        resp = self._client.transport.request(
            "GET",
            _PATH_DEPLOYMENTS,
            params=params,
        )
        return build_page(
            payload=resp.data,
            item_factory=_coerce_deployment,
            request_id=resp.request_id,
            fetch_next=lambda nxt: self.list(
                agent_id=agent_id,
                keyword=keyword,
                status=status,
                include_archived=include_archived,
                limit=limit,
                page=nxt,
                created_at_gte=created_at_gte,
                created_at_lte=created_at_lte,
            ),
        )

    def pause(self, deployment_id: str) -> Deployment:
        return self._lifecycle(deployment_id, "pause")

    def unpause(self, deployment_id: str) -> Deployment:
        return self._lifecycle(deployment_id, "unpause")

    def archive(self, deployment_id: str) -> Deployment:
        return self._lifecycle(deployment_id, "archive")

    def _lifecycle(self, deployment_id: str, action: str) -> Deployment:
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_DEPLOYMENTS}/{deployment_id}/{action}",
        )
        return _coerce_deployment(resp.data)

    def run(self, deployment_id: str) -> DeploymentRun:
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_DEPLOYMENTS}/{deployment_id}/run",
        )
        return _coerce_deployment_run(resp.data)

    def list_runs(
        self,
        deployment_id: str,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> CursorPage[DeploymentRun]:
        params = DeploymentRunListParams(limit=limit, page=page).to_dict()
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_DEPLOYMENTS}/{deployment_id}/runs",
            params=params,
        )
        return build_page(
            payload=resp.data,
            item_factory=_coerce_deployment_run,
            request_id=resp.request_id,
            fetch_next=lambda nxt: self.list_runs(
                deployment_id,
                limit=limit,
                page=nxt,
            ),
        )


class DeploymentRuns:
    """Workspace-wide deployment run history."""

    def __init__(self, client) -> None:
        self._client = client

    def retrieve(self, deployment_run_id: str) -> DeploymentRun:
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_DEPLOYMENT_RUNS}/{deployment_run_id}",
        )
        return _coerce_deployment_run(resp.data)

    get = retrieve  # type: ignore[assignment]

    def list(
        self,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> CursorPage[DeploymentRun]:
        params = DeploymentRunListParams(limit=limit, page=page).to_dict()
        resp = self._client.transport.request(
            "GET",
            _PATH_DEPLOYMENT_RUNS,
            params=params,
        )
        return build_page(
            payload=resp.data,
            item_factory=_coerce_deployment_run,
            request_id=resp.request_id,
            fetch_next=lambda nxt: self.list(limit=limit, page=nxt),
        )


class AsyncDeployments:
    """Asynchronous Managed Agent deployment lifecycle and execution."""

    def __init__(self, client) -> None:
        self._client = client

    async def create(
        self,
        *,
        name: str,
        agent: Any,
        initial_events: Sequence[Mapping[str, Any]],
        description: Optional[str] = None,
        environment_id: Optional[str] = None,
        schedule: Any = None,
        resources: Optional[Sequence[Any]] = None,
        vault_ids: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> Deployment:
        body = DeploymentCreateParams(
            name=name,
            agent=agent,
            initial_events=initial_events,
            description=description,
            environment_id=environment_id,
            schedule=schedule,
            resources=resources,
            vault_ids=vault_ids,
            metadata=metadata,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            _PATH_DEPLOYMENTS,
            json=body,
        )
        return _coerce_deployment(resp.data)

    async def retrieve(self, deployment_id: str) -> Deployment:
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_DEPLOYMENTS}/{deployment_id}",
        )
        return _coerce_deployment(resp.data)

    get = retrieve  # type: ignore[assignment]

    async def update(
        self,
        deployment_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        agent: Any = None,
        environment_id: Any = _NOT_GIVEN,
        schedule: Any = _NOT_GIVEN,
        initial_events: Optional[Sequence[Mapping[str, Any]]] = None,
        resources: Optional[Sequence[Any]] = None,
        vault_ids: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> Deployment:
        body = DeploymentUpdateParams(
            name=name,
            description=description,
            agent=agent,
            environment_id=environment_id,
            schedule=schedule,
            initial_events=initial_events,
            resources=resources,
            vault_ids=vault_ids,
            metadata=metadata,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_DEPLOYMENTS}/{deployment_id}",
            json=body,
        )
        return _coerce_deployment(resp.data)

    async def list(
        self,
        *,
        agent_id: Optional[str] = None,
        keyword: Optional[str] = None,
        status: Optional[str] = None,
        include_archived: Optional[bool] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        created_at_gte: Optional[str] = None,
        created_at_lte: Optional[str] = None,
    ) -> AsyncCursorPage[Deployment]:
        params = DeploymentListParams(
            agent_id=agent_id,
            keyword=keyword,
            status=status,
            include_archived=include_archived,
            limit=limit,
            page=page,
            created_at_gte=created_at_gte,
            created_at_lte=created_at_lte,
        ).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _PATH_DEPLOYMENTS,
            params=params,
        )

        async def fetch_next(nxt: str) -> AsyncCursorPage[Deployment]:
            return await self.list(
                agent_id=agent_id,
                keyword=keyword,
                status=status,
                include_archived=include_archived,
                limit=limit,
                page=nxt,
                created_at_gte=created_at_gte,
                created_at_lte=created_at_lte,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_deployment,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def pause(self, deployment_id: str) -> Deployment:
        return await self._lifecycle(deployment_id, "pause")

    async def unpause(self, deployment_id: str) -> Deployment:
        return await self._lifecycle(deployment_id, "unpause")

    async def archive(self, deployment_id: str) -> Deployment:
        return await self._lifecycle(deployment_id, "archive")

    async def _lifecycle(
        self,
        deployment_id: str,
        action: str,
    ) -> Deployment:
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_DEPLOYMENTS}/{deployment_id}/{action}",
        )
        return _coerce_deployment(resp.data)

    async def run(self, deployment_id: str) -> DeploymentRun:
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_DEPLOYMENTS}/{deployment_id}/run",
        )
        return _coerce_deployment_run(resp.data)

    async def list_runs(
        self,
        deployment_id: str,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[DeploymentRun]:
        params = DeploymentRunListParams(limit=limit, page=page).to_dict()
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_DEPLOYMENTS}/{deployment_id}/runs",
            params=params,
        )

        async def fetch_next(nxt: str) -> AsyncCursorPage[DeploymentRun]:
            return await self.list_runs(
                deployment_id,
                limit=limit,
                page=nxt,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_deployment_run,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )


class AsyncDeploymentRuns:
    """Asynchronous workspace-wide deployment run history."""

    def __init__(self, client) -> None:
        self._client = client

    async def retrieve(self, deployment_run_id: str) -> DeploymentRun:
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_DEPLOYMENT_RUNS}/{deployment_run_id}",
        )
        return _coerce_deployment_run(resp.data)

    get = retrieve  # type: ignore[assignment]

    async def list(
        self,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[DeploymentRun]:
        params = DeploymentRunListParams(limit=limit, page=page).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _PATH_DEPLOYMENT_RUNS,
            params=params,
        )

        async def fetch_next(nxt: str) -> AsyncCursorPage[DeploymentRun]:
            return await self.list(limit=limit, page=nxt)

        return build_page(
            payload=resp.data,
            item_factory=_coerce_deployment_run,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )
