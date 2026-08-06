# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AgentStudio client classes."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Union

import httpx

from dashscope.agentstudio.resources.agents import Agents, AsyncAgents
from dashscope.agentstudio.resources.deployments import (
    AsyncDeploymentRuns,
    AsyncDeployments,
    DeploymentRuns,
    Deployments,
)
from dashscope.agentstudio.resources.environments import (
    Environments,
    AsyncEnvironments,
)
from dashscope.agentstudio.resources.files import Files, AsyncFiles
from dashscope.agentstudio.resources.sessions import Sessions, AsyncSessions
from dashscope.agentstudio.resources.skills import Skills, AsyncSkills
from dashscope.agentstudio.resources.vaults import Vaults, AsyncVaults
from dashscope.agentstudio.constants import (
    AGENTSTUDIO_BASE_URL_TEMPLATE,
    AGENTSTUDIO_DEFAULT_REGION,
    AGENTSTUDIO_DEFAULT_TIMEOUT,
    AGENTSTUDIO_MAX_RETRIES,
)
from dashscope.agentstudio.transport import SyncTransport, AsyncTransport
from dashscope.common.api_key import get_default_api_key


def _resolve_base_url(
    explicit_url: Optional[str],
    workspace: Optional[str],
    region: Optional[str] = None,
) -> str:
    """Resolve the base URL.

    Priority:
    1. explicit base_url parameter (full URL)
    2. DASHSCOPE_AGENTSTUDIO_URL / AGENTSTUDIO_URL env
    3. Build from workspace + region template
    """
    if explicit_url:
        return explicit_url
    env_url = os.environ.get("DASHSCOPE_AGENTSTUDIO_URL") or os.environ.get(
        "AGENTSTUDIO_URL",
    )
    if env_url:
        return env_url
    # Build from workspace + region
    ws = workspace or os.environ.get("DASHSCOPE_WORKSPACE")
    if not ws:
        raise ValueError(
            "workspace is required when base_url is not provided "
            "(pass workspace='ws_xxx' or set DASHSCOPE_WORKSPACE env var "
            "or use base_url=... to override)",
        )
    rgn = region or AGENTSTUDIO_DEFAULT_REGION
    return AGENTSTUDIO_BASE_URL_TEMPLATE.format(
        workspace=ws,
        region=rgn,
    )


def _user_agent(base_url: str) -> str:
    try:
        from dashscope.version import __version__
    except Exception:
        __version__ = "0.0.0"
    return f"dashscope-agentstudio-python/{__version__} (+{base_url})"


class Client:
    """Synchronous AgentStudio client.

    Usage::

        from dashscope.agentstudio import Client

        client = Client(api_key="sk-xxx", workspace="my-ws")
        agent = client.agents.create(name="demo", model="qwen-max")
        session = client.sessions.create(agent=agent.id)
        with client.sessions.events.stream(session.id) as stream:
            for event in stream:
                print(event.type)
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        region: Optional[str] = None,
        base_url: Optional[str] = None,
        uid: Optional[str] = None,
        timeout: Optional[
            Union[float, httpx.Timeout, Tuple[float, float]]
        ] = None,
        max_retries: int = AGENTSTUDIO_MAX_RETRIES,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        resolved_workspace = workspace or os.environ.get("DASHSCOPE_WORKSPACE")
        resolved_base = _resolve_base_url(
            base_url,
            resolved_workspace,
            region,
        )
        self.transport = SyncTransport(
            base_url=resolved_base,
            api_key=api_key or get_default_api_key(),
            workspace=resolved_workspace,
            uid=uid or os.environ.get("DASHSCOPE_UID"),
            user_agent=_user_agent(resolved_base),
            timeout=timeout or AGENTSTUDIO_DEFAULT_TIMEOUT,
            max_retries=max_retries,
            http_client=http_client,
        )
        self.agents = Agents(self)
        self.deployments = Deployments(self)
        self.deployment_runs = DeploymentRuns(self)
        self.sessions = Sessions(self)
        self.environments = Environments(self)
        self.files = Files(self)
        self.skills = Skills(self)
        self.vaults = Vaults(self)

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class AsyncClient:
    """Asynchronous AgentStudio client.

    Usage::

        from dashscope.agentstudio import AsyncClient

        client = AsyncClient(api_key="sk-xxx", workspace="my-ws")
        agent = await client.agents.create(
            name="demo", model="qwen-max",
        )
        session = await client.sessions.create(agent=agent.id)
        async with client.sessions.events.stream(session.id) as stream:
            async for event in stream:
                print(event.type)
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        region: Optional[str] = None,
        base_url: Optional[str] = None,
        uid: Optional[str] = None,
        timeout: Optional[
            Union[float, httpx.Timeout, Tuple[float, float]]
        ] = None,
        max_retries: int = AGENTSTUDIO_MAX_RETRIES,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        resolved_workspace = workspace or os.environ.get("DASHSCOPE_WORKSPACE")
        resolved_base = _resolve_base_url(
            base_url,
            resolved_workspace,
            region,
        )
        self.transport = AsyncTransport(
            base_url=resolved_base,
            api_key=api_key or get_default_api_key(),
            workspace=resolved_workspace,
            uid=uid or os.environ.get("DASHSCOPE_UID"),
            user_agent=_user_agent(resolved_base),
            timeout=timeout or AGENTSTUDIO_DEFAULT_TIMEOUT,
            max_retries=max_retries,
            http_client=http_client,
        )
        self.agents = AsyncAgents(self)
        self.deployments = AsyncDeployments(self)
        self.deployment_runs = AsyncDeploymentRuns(self)
        self.sessions = AsyncSessions(self)
        self.environments = AsyncEnvironments(self)
        self.files = AsyncFiles(self)
        self.skills = AsyncSkills(self)
        self.vaults = AsyncVaults(self)

    async def aclose(self) -> None:
        await self.transport.aclose()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __del__(self) -> None:
        try:
            self.transport.close()
        except Exception:
            pass
