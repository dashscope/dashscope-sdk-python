# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 Client shells.

Resources and transport are attached in subsequent M1 tasks;
this module only holds construction and configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import httpx

from dashscope._config import Config, resolve_config
from dashscope.protocols import ChatProtocol, get_protocol
from dashscope.resources.apps import Apps, AsyncApps
from dashscope.resources.audio import AsyncAudio, Audio
from dashscope.resources.chat import AsyncChat, Chat
from dashscope.resources.decision import AsyncDecision, Decision
from dashscope.resources.deployments import AsyncDeployments, Deployments
from dashscope.resources.embeddings import AsyncEmbeddings, Embeddings
from dashscope.resources.files import AsyncFiles, Files
from dashscope.resources.fine_tunes import AsyncFineTunes, FineTunes
from dashscope.resources.images import AsyncImages, Images
from dashscope.resources.models import AsyncModels, Models
from dashscope.resources.multimodal import AsyncMultimodal, Multimodal
from dashscope.resources.rerank import AsyncRerank, Rerank
from dashscope.resources.reinforcement_learning import AsyncRl, Rl
from dashscope.resources.tingwu import AsyncTingWu, TingWu
from dashscope.resources.tokenization import AsyncTokenization, Tokenization
from dashscope.resources.videos import AsyncVideos, Videos
from dashscope.transport import (
    AsyncTransport,
    SyncTransport,
    default_user_agent,
)
from dashscope.transport.ws import Connector
from dashscope.transport.ws_connection import WsConnection

# Historical dashscope.client.* children resolve through the 1.x alias finder.
__path__ = []


def _http_limits(config: Config) -> httpx.Limits:
    """Connection-pool sizes for the client's own httpx client."""
    return httpx.Limits(
        max_connections=config.http_max_connections,
        max_keepalive_connections=config.http_max_keepalive_connections,
        keepalive_expiry=config.http_keepalive_expiry,
    )


def _agentstudio_base_url(config: Config) -> Optional[str]:
    """Derive AgentStudio's service root from an explicit main base URL.

    Default workspace gateways are resolved by the AgentStudio core itself.
    A configured main service root ends at ``/api/v1``; AgentStudio is mounted
    below it at ``/agentstudio``.
    """
    if config.configured_base_url is None:
        return None
    base = config.configured_base_url.rstrip("/")
    if base.endswith("/agentstudio"):
        return base
    return f"{base}/agentstudio"


class Client:
    """Synchronous dashscope 2.0 client.

    Usage::

        import dashscope

        client = dashscope.Client(api_key="sk-xxx")
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        workspace: Optional[str] = None,
        region: Optional[str] = None,
        protocol: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        config_path: Union[str, Path, None] = None,
        http_client: Optional[httpx.Client] = None,
        raise_on_error: bool = True,
        protocol_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._config = resolve_config(
            api_key=api_key,
            base_url=base_url,
            workspace=workspace,
            region=region,
            protocol=protocol,
            timeout=timeout,
            max_retries=max_retries,
            config_path=config_path,
        )
        self._raise_on_error = raise_on_error
        self._protocol_options = protocol_options
        self._protocol = get_protocol(
            self._config.protocol,
            protocol_options=protocol_options,
        )
        self._transport = SyncTransport(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
            workspace=self._config.workspace,
            user_agent=default_user_agent(),
            timeout=self._config.timeout,
            max_retries=self._config.max_retries,
            http_client=http_client,
            limits=None
            if http_client is not None
            else _http_limits(self._config),
        )
        self.chat = Chat(self)
        self.multimodal = Multimodal(self)
        self.embeddings = Embeddings(self)
        self.rerank = Rerank(self)
        self.decision = Decision(self)
        self.tokenization = Tokenization(self)
        self.files = Files(self)
        self.models = Models(self)
        self.images = Images(self)
        self.videos = Videos(self)
        self._audio = None
        self.fine_tunes = FineTunes(self)
        self.deployments = Deployments(self)
        self.rl = Rl(self)
        self.apps = Apps(self)
        self.tingwu = TingWu(self)
        self._agentstudio = None

    @property
    def audio(self) -> "Audio":
        """Audio capabilities (lazily mounted)."""
        if self._audio is None:
            self._audio = Audio(self)
        audio = self._audio
        assert audio is not None
        return audio

    @property
    def reinforcement_learning(self) -> "Rl":
        """Long-form alias for ``client.rl``."""
        return self.rl

    @property
    def agentstudio(self):
        """Lazily mounted AgentStudio sub-SDK (design doc M3 3.8).

        AgentStudio raises ``dashscope.errors`` natively. It ships its
        own transport rather than riding this client's, so its requests
        keep the unmarked module tag (``agentstudio``) instead of the v2
        ``2`` marker -- the backend already reports on that tag, and
        changing it would split the server-side statistics.
        """
        if self._agentstudio is None:
            # pylint: disable=import-outside-toplevel
            from dashscope.applications.agentstudio import (
                Client as AgentStudioClient,
            )

            # agentstudio resolves its own base_url from workspace +
            # region (the v2 base_url does not propagate: the agentstudio
            # path differs), so forward both.
            self._agentstudio = AgentStudioClient(
                api_key=self._config.api_key,
                workspace=self._config.workspace,
                region=self._config.region,
                base_url=_agentstudio_base_url(self._config),
                timeout=self._config.timeout,
                max_retries=self._config.max_retries,
                http_client=self._transport.http_client,
            )
        return self._agentstudio

    @property
    def config(self) -> Config:
        return self._config

    @property
    def protocol(self) -> ChatProtocol:
        return self._protocol

    @property
    def transport(self) -> SyncTransport:
        return self._transport

    @property
    def raise_on_error(self) -> bool:
        return self._raise_on_error

    def with_options(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        workspace: Optional[str] = None,
        region: Optional[str] = None,
        protocol: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        raise_on_error: Optional[bool] = None,
    ) -> "Client":
        """Derive a client with the given options overridden.

        Unmentioned options are inherited. When connection-level options
        (``api_key``/``base_url``/``workspace``/``region``/``timeout``)
        are left unchanged the derived client shares this one's
        connection pool and must not outlive it; otherwise it builds its
        own pool.
        """
        config = self._config
        share_pool = (
            api_key is None
            and base_url is None
            and workspace is None
            and region is None
            and timeout is None
        )
        return type(self)(
            api_key=config.api_key if api_key is None else api_key,
            base_url=(
                config.configured_base_url if base_url is None else base_url
            ),
            workspace=(config.workspace if workspace is None else workspace),
            region=config.region if region is None else region,
            protocol=config.protocol if protocol is None else protocol,
            timeout=config.timeout if timeout is None else timeout,
            max_retries=(
                config.max_retries if max_retries is None else max_retries
            ),
            http_client=(self._transport.http_client if share_pool else None),
            raise_on_error=(
                self._raise_on_error
                if raise_on_error is None
                else raise_on_error
            ),
            protocol_options=self._protocol_options,
        )

    def close(self) -> None:
        self._transport.close()
        if self._agentstudio is not None:
            self._agentstudio.close()
            self._agentstudio = None

    def ws_connection(
        self,
        *,
        url: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        connector: Optional[Connector] = None,
        timeout: Optional[float] = None,
        module: Optional[str] = None,
        task_queue_size: int = 1024,
        ping_interval: Optional[float] = 30.0,
    ) -> WsConnection:
        """Open a shared duplex websocket carrying multiple tasks.

        Credentials, region and timeout come from this client's config.
        Pass the connection as ``connection=`` to the realtime audio
        sessions (``audio.asr.recognize`` / ``audio.tts.connect`` /
        ``tingwu.realtime``) to run tasks concurrently or back-to-back
        on one socket, saving the handshake per task -- the 1.x
        Java-side connection-reuse pattern for latency-sensitive audio
        traffic. The caller owns the connection and closes it via
        ``WsConnection.close``.
        """
        config = self._config
        return WsConnection(
            url=url,
            api_key=config.api_key,
            workspace=config.workspace,
            region=config.region,
            headers=headers,
            connector=connector,
            timeout=config.timeout if timeout is None else timeout,
            module=module,
            task_queue_size=task_queue_size,
            ping_interval=ping_interval,
        )

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncClient:
    """Asynchronous dashscope 2.0 client.

    Usage::

        import dashscope

        client = dashscope.AsyncClient(api_key="sk-xxx")
        resp = await client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        workspace: Optional[str] = None,
        region: Optional[str] = None,
        protocol: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        config_path: Union[str, Path, None] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        raise_on_error: bool = True,
        protocol_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._config = resolve_config(
            api_key=api_key,
            base_url=base_url,
            workspace=workspace,
            region=region,
            protocol=protocol,
            timeout=timeout,
            max_retries=max_retries,
            config_path=config_path,
        )
        self._raise_on_error = raise_on_error
        self._protocol_options = protocol_options
        self._protocol = get_protocol(
            self._config.protocol,
            protocol_options=protocol_options,
        )
        self._transport = AsyncTransport(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
            workspace=self._config.workspace,
            user_agent=default_user_agent(),
            timeout=self._config.timeout,
            max_retries=self._config.max_retries,
            http_client=http_client,
            limits=None
            if http_client is not None
            else _http_limits(self._config),
        )
        self.chat = AsyncChat(self)
        self.multimodal = AsyncMultimodal(self)
        self.embeddings = AsyncEmbeddings(self)
        self.rerank = AsyncRerank(self)
        self.decision = AsyncDecision(self)
        self.tokenization = AsyncTokenization(self)
        self.files = AsyncFiles(self)
        self.models = AsyncModels(self)
        self.images = AsyncImages(self)
        self.videos = AsyncVideos(self)
        self._audio = None
        self.fine_tunes = AsyncFineTunes(self)
        self.deployments = AsyncDeployments(self)
        self.rl = AsyncRl(self)
        self.apps = AsyncApps(self)
        self.tingwu = AsyncTingWu(self)
        self._agentstudio = None

    @property
    def audio(self) -> "AsyncAudio":
        """Audio capabilities (lazily mounted)."""
        if self._audio is None:
            self._audio = AsyncAudio(self)
        audio = self._audio
        assert audio is not None
        return audio

    @property
    def reinforcement_learning(self) -> "AsyncRl":
        """Long-form alias for ``client.rl``."""
        return self.rl

    @property
    def agentstudio(self):
        """Lazily mounted AgentStudio sub-SDK (design doc M3 3.8).

        See :attr:`Client.agentstudio`; errors use the ``dashscope``
        exception family natively.
        """
        if self._agentstudio is None:
            # pylint: disable=import-outside-toplevel
            from dashscope.applications.agentstudio import (
                AsyncClient as AsClient,
            )

            # See Client.agentstudio: agentstudio builds its own base_url
            # from workspace + region, so forward both.
            self._agentstudio = AsClient(
                api_key=self._config.api_key,
                workspace=self._config.workspace,
                region=self._config.region,
                base_url=_agentstudio_base_url(self._config),
                timeout=self._config.timeout,
                max_retries=self._config.max_retries,
                http_client=self._transport.http_client,
            )
        return self._agentstudio

    @property
    def config(self) -> Config:
        return self._config

    @property
    def protocol(self) -> ChatProtocol:
        return self._protocol

    @property
    def transport(self) -> AsyncTransport:
        return self._transport

    @property
    def raise_on_error(self) -> bool:
        return self._raise_on_error

    def with_options(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        workspace: Optional[str] = None,
        region: Optional[str] = None,
        protocol: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        raise_on_error: Optional[bool] = None,
    ) -> "AsyncClient":
        """Derive a client with the given options overridden.

        Unmentioned options are inherited. When connection-level options
        (``api_key``/``base_url``/``workspace``/``region``/``timeout``)
        are left unchanged the derived client shares this one's
        connection pool and must not outlive it; otherwise it builds its
        own pool.
        """
        config = self._config
        share_pool = (
            api_key is None
            and base_url is None
            and workspace is None
            and region is None
            and timeout is None
        )
        return type(self)(
            api_key=config.api_key if api_key is None else api_key,
            base_url=(
                config.configured_base_url if base_url is None else base_url
            ),
            workspace=(config.workspace if workspace is None else workspace),
            region=config.region if region is None else region,
            protocol=config.protocol if protocol is None else protocol,
            timeout=config.timeout if timeout is None else timeout,
            max_retries=(
                config.max_retries if max_retries is None else max_retries
            ),
            http_client=(self._transport.http_client if share_pool else None),
            raise_on_error=(
                self._raise_on_error
                if raise_on_error is None
                else raise_on_error
            ),
            protocol_options=self._protocol_options,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()
        if self._agentstudio is not None:
            await self._agentstudio.aclose()
            self._agentstudio = None

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
