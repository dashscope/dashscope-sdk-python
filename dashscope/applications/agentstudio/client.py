# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AgentStudio client classes."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Union

import httpx

from dashscope.applications.agentstudio import (
    exceptions as _default_exceptions,
)
from dashscope._shared._validation import (
    validate_region,
    validate_workspace_id,
)
from dashscope.applications.agentstudio.resources.agents import (
    Agents,
    AsyncAgents,
)
from dashscope.applications.agentstudio.resources.deployments import (
    AsyncDeploymentRuns,
    AsyncDeployments,
    DeploymentRuns,
    Deployments,
)
from dashscope.applications.agentstudio.resources.environments import (
    Environments,
    AsyncEnvironments,
)
from dashscope.applications.agentstudio.resources.files import (
    Files,
    AsyncFiles,
)
from dashscope.applications.agentstudio.resources.memory_stores import (
    MemoryStores,
    AsyncMemoryStores,
)
from dashscope.applications.agentstudio.resources.security import (
    Security,
    AsyncSecurity,
)
from dashscope.applications.agentstudio.resources.sessions import (
    Sessions,
    AsyncSessions,
)
from dashscope.applications.agentstudio.resources.skills import (
    Skills,
    AsyncSkills,
)
from dashscope.applications.agentstudio.resources.vaults import (
    Vaults,
    AsyncVaults,
)
from dashscope.applications.agentstudio.resources.webhook_endpoints import (
    AsyncWebhookEndpoints,
    WebhookEndpoints,
)
from dashscope.applications.agentstudio.constants import (
    AGENTSTUDIO_BASE_URL_TEMPLATE,
    AGENTSTUDIO_DEFAULT_REGION,
    AGENTSTUDIO_DEFAULT_TIMEOUT,
    AGENTSTUDIO_MAX_RETRIES,
)
from dashscope.applications.agentstudio.transport import (
    SyncTransport,
    AsyncTransport,
)
from dashscope.applications.agentstudio.streaming import (
    AsyncEventStream,
    EventStream,
)
from dashscope._config import _api_key_from_file, settings


def _default_api_key() -> Optional[str]:
    """Resolve the api key when ``Client(api_key=...)`` was not given.

    Same tiers as the 2.0 client: ``settings.api_key`` (explicit value or
    the ``DASHSCOPE_API_KEY`` env var), then the key file written by
    ``save_api_key``. Returns ``None`` rather than raising -- the
    transport turns a missing key into this SDK's own
    ``AuthenticationError`` on the first request, with a message that
    names the AgentStudio entry points.

    This deliberately does not read the 1.x ``dashscope.api_key`` module
    global; set ``dashscope.settings.api_key`` or the environment
    variable instead.
    """
    return settings.api_key or _api_key_from_file()


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

    Raises:
        ValueError: If workspace or region is missing, or is not a valid
            hostname label.
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
    # Both are interpolated into the host below; unvalidated values such as
    # "a@evil.com/#" would move the effective host off *.maas.aliyuncs.com
    # while the transport still sends Authorization: Bearer <api_key>.
    validate_workspace_id(ws)
    validate_region(rgn)
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

        from dashscope.applications.agentstudio import Client, ModelConfig

        client = Client(api_key="sk-xxx", workspace="my-ws")
        agent = client.agents.create(
            name="demo",
            model=ModelConfig(
                id="qwen-max",
                temperature=0.7,
                enable_thinking=True,
                thinking_budget=2048,
            ),
        )
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
        _exceptions_module: Any = _default_exceptions,
        _api_key_resolver: Any = _default_api_key,
        _transport_cls: Any = SyncTransport,
        _event_stream_cls: Any = EventStream,
    ) -> None:
        resolved_workspace = workspace or os.environ.get("DASHSCOPE_WORKSPACE")
        resolved_base = _resolve_base_url(
            base_url,
            resolved_workspace,
            region,
        )
        self._event_stream_cls = _event_stream_cls
        self.transport = _transport_cls(
            base_url=resolved_base,
            api_key=api_key or _api_key_resolver(),
            workspace=resolved_workspace,
            uid=uid or os.environ.get("DASHSCOPE_UID"),
            user_agent=_user_agent(resolved_base),
            timeout=timeout or AGENTSTUDIO_DEFAULT_TIMEOUT,
            max_retries=max_retries,
            http_client=http_client,
            exceptions_module=_exceptions_module,
        )
        self.agents = Agents(self)
        self.deployments = Deployments(self)
        self.deployment_runs = DeploymentRuns(self)
        self.sessions = Sessions(self)
        self.environments = Environments(self)
        self.files = Files(self)
        self.skills = Skills(self)
        self.vaults = Vaults(self)
        self.memory_stores = MemoryStores(self)
        self.webhook_endpoints = WebhookEndpoints(self)
        self.security = Security(self)

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

        from dashscope.applications.agentstudio import AsyncClient, ModelConfig

        client = AsyncClient(api_key="sk-xxx", workspace="my-ws")
        agent = await client.agents.create(
            name="demo",
            model=ModelConfig(id="qwen-max", max_tokens=4096),
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
        _exceptions_module: Any = _default_exceptions,
        _api_key_resolver: Any = _default_api_key,
        _transport_cls: Any = AsyncTransport,
        _event_stream_cls: Any = AsyncEventStream,
    ) -> None:
        resolved_workspace = workspace or os.environ.get("DASHSCOPE_WORKSPACE")
        resolved_base = _resolve_base_url(
            base_url,
            resolved_workspace,
            region,
        )
        self._event_stream_cls = _event_stream_cls
        self.transport = _transport_cls(
            base_url=resolved_base,
            api_key=api_key or _api_key_resolver(),
            workspace=resolved_workspace,
            uid=uid or os.environ.get("DASHSCOPE_UID"),
            user_agent=_user_agent(resolved_base),
            timeout=timeout or AGENTSTUDIO_DEFAULT_TIMEOUT,
            max_retries=max_retries,
            http_client=http_client,
            exceptions_module=_exceptions_module,
        )
        self.agents = AsyncAgents(self)
        self.deployments = AsyncDeployments(self)
        self.deployment_runs = AsyncDeploymentRuns(self)
        self.sessions = AsyncSessions(self)
        self.environments = AsyncEnvironments(self)
        self.files = AsyncFiles(self)
        self.skills = AsyncSkills(self)
        self.vaults = AsyncVaults(self)
        self.memory_stores = AsyncMemoryStores(self)
        self.webhook_endpoints = AsyncWebhookEndpoints(self)
        self.security = AsyncSecurity(self)

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
