# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Agents resource class."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashscope.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
    build_page,
)
from dashscope.agentstudio.resources._helpers import (
    _coerce_agent,
    _coerce_agent_version,
)
from dashscope.agentstudio.types import Agent, AgentVersion
from dashscope.agentstudio.types.params import (
    AgentCreateParams,
    AgentListParams,
    AgentUpdateParams,
    AgentVersionListParams,
)

_PATH_AGENTS = "/agents"


class Agents:
    """Agent CRUD — instance methods backed by a shared client."""

    def __init__(self, client) -> None:
        self._client = client

    def create(
        self,
        *,
        name: str,
        model: str,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        mcp_servers: Optional[Sequence[Mapping[str, Any]]] = None,
        skills: Optional[Sequence[Mapping[str, Any]]] = None,
        multiagent: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Agent:
        body = AgentCreateParams(
            name=name,
            model=model,
            description=description,
            system_prompt=system_prompt,
            tools=tools,
            mcp_servers=mcp_servers,
            skills=skills,
            multiagent=multiagent,
            metadata=metadata,
        ).to_dict()
        resp = self._client.transport.request("POST", _PATH_AGENTS, json=body)
        return _coerce_agent(resp.data)

    def retrieve(
        self,
        agent_id: str,
        *,
        version: Optional[int] = None,
    ) -> Agent:
        params = {"version": version} if version is not None else None
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_AGENTS}/{agent_id}",
            params=params,
        )
        return _coerce_agent(resp.data)

    # Alias: get() delegates to retrieve(), matching Assistants.get() pattern
    get = retrieve  # type: ignore[assignment]

    # Alias: call() delegates to create(), matching Assistants.call() pattern
    call = create  # type: ignore[assignment]

    def update(
        self,
        agent_id: str,
        *,
        version: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        mcp_servers: Optional[Sequence[Mapping[str, Any]]] = None,
        skills: Optional[Sequence[Mapping[str, Any]]] = None,
        multiagent: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Agent:
        """Update the latest version of an agent.

        ``version`` must equal the server's current version;
        the server rejects mismatches with HTTP 409.
        Retrieve the agent first to obtain the current version.
        """
        body = AgentUpdateParams(
            version=version,
            name=name,
            description=description,
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            mcp_servers=mcp_servers,
            skills=skills,
            multiagent=multiagent,
            metadata=metadata,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_AGENTS}/{agent_id}",
            json=body,
        )
        return _coerce_agent(resp.data)

    def archive(self, agent_id: str) -> Agent:
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_AGENTS}/{agent_id}/archive",
        )
        return _coerce_agent(resp.data)

    def list(
        self,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        include_archived: Optional[bool] = None,
    ) -> CursorPage[Agent]:
        params = AgentListParams(
            limit=limit,
            page=page,
            include_archived=include_archived,
        ).to_dict()
        resp = self._client.transport.request(
            "GET",
            _PATH_AGENTS,
            params=params,
        )
        return build_page(
            payload=resp.data,
            item_factory=_coerce_agent,
            request_id=resp.request_id,
            fetch_next=lambda nxt: self.list(
                limit=limit,
                page=nxt,
                include_archived=include_archived,
            ),
        )

    def list_versions(
        self,
        agent_id: str,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> CursorPage[AgentVersion]:
        params = AgentVersionListParams(limit=limit, page=page).to_dict()
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_AGENTS}/{agent_id}/versions",
            params=params,
        )
        return build_page(
            payload=resp.data,
            item_factory=_coerce_agent_version,
            request_id=resp.request_id,
            fetch_next=lambda nxt: self.list_versions(
                agent_id,
                limit=limit,
                page=nxt,
            ),
        )


class AsyncAgents:
    """Async Agent CRUD — instance methods backed by a shared client."""

    def __init__(self, client) -> None:
        self._client = client

    async def create(
        self,
        *,
        name: str,
        model: str,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        mcp_servers: Optional[Sequence[Mapping[str, Any]]] = None,
        skills: Optional[Sequence[Mapping[str, Any]]] = None,
        multiagent: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Agent:
        body = AgentCreateParams(
            name=name,
            model=model,
            description=description,
            system_prompt=system_prompt,
            tools=tools,
            mcp_servers=mcp_servers,
            skills=skills,
            multiagent=multiagent,
            metadata=metadata,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            _PATH_AGENTS,
            json=body,
        )
        return _coerce_agent(resp.data)

    async def retrieve(
        self,
        agent_id: str,
        *,
        version: Optional[int] = None,
    ) -> Agent:
        params = {"version": version} if version is not None else None
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_AGENTS}/{agent_id}",
            params=params,
        )
        return _coerce_agent(resp.data)

    # Alias: get() delegates to retrieve()
    get = retrieve  # type: ignore[assignment]

    # Alias: call() delegates to create()
    call = create  # type: ignore[assignment]

    async def update(
        self,
        agent_id: str,
        *,
        version: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        mcp_servers: Optional[Sequence[Mapping[str, Any]]] = None,
        skills: Optional[Sequence[Mapping[str, Any]]] = None,
        multiagent: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Agent:
        """Update the latest version of an agent.

        ``version`` must equal the server's current version;
        the server rejects mismatches with HTTP 409.
        Retrieve the agent first to obtain the current version.
        """
        body = AgentUpdateParams(
            version=version,
            name=name,
            description=description,
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            mcp_servers=mcp_servers,
            skills=skills,
            multiagent=multiagent,
            metadata=metadata,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_AGENTS}/{agent_id}",
            json=body,
        )
        return _coerce_agent(resp.data)

    async def archive(self, agent_id: str) -> Agent:
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_AGENTS}/{agent_id}/archive",
        )
        return _coerce_agent(resp.data)

    async def list(
        self,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        include_archived: Optional[bool] = None,
    ) -> AsyncCursorPage[Agent]:
        params = AgentListParams(
            limit=limit,
            page=page,
            include_archived=include_archived,
        ).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _PATH_AGENTS,
            params=params,
        )

        async def fetch_next(nxt: str) -> AsyncCursorPage[Agent]:
            return await self.list(
                limit=limit,
                page=nxt,
                include_archived=include_archived,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_agent,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def list_versions(
        self,
        agent_id: str,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[AgentVersion]:
        params = AgentVersionListParams(limit=limit, page=page).to_dict()
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_AGENTS}/{agent_id}/versions",
            params=params,
        )

        async def fetch_next(nxt: str) -> AsyncCursorPage[AgentVersion]:
            return await self.list_versions(
                agent_id,
                limit=limit,
                page=nxt,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_agent_version,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )
