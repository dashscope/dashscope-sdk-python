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

from dashscope.v2._config import Config, resolve_config
from dashscope.v2.protocols import ChatProtocol, get_protocol
from dashscope.v2.resources.apps import Apps, AsyncApps
from dashscope.v2.resources.audio import AsyncAudio, Audio
from dashscope.v2.resources.chat import AsyncChat, Chat
from dashscope.v2.resources.decision import AsyncDecision, Decision
from dashscope.v2.resources.deployments import AsyncDeployments, Deployments
from dashscope.v2.resources.embeddings import AsyncEmbeddings, Embeddings
from dashscope.v2.resources.files import AsyncFiles, Files
from dashscope.v2.resources.fine_tunes import AsyncFineTunes, FineTunes
from dashscope.v2.resources.images import AsyncImages, Images
from dashscope.v2.resources.models import AsyncModels, Models
from dashscope.v2.resources.multimodal import AsyncMultimodal, Multimodal
from dashscope.v2.resources.rerank import AsyncRerank, Rerank
from dashscope.v2.resources.reinforcement_learning import AsyncRl, Rl
from dashscope.v2.resources.tingwu import AsyncTingWu, TingWu
from dashscope.v2.resources.tokenization import AsyncTokenization, Tokenization
from dashscope.v2.resources.videos import AsyncVideos, Videos
from dashscope.v2.transport import (
    AsyncTransport,
    SyncTransport,
    default_user_agent,
)
from dashscope.v2.transport.ws import Connector
from dashscope.v2.transport.ws_connection import WsConnection


def _http_limits(config: Config) -> httpx.Limits:
    """Connection-pool sizes for the client's own httpx client."""
    return httpx.Limits(
        max_connections=config.http_max_connections,
        max_keepalive_connections=config.http_max_keepalive_connections,
        keepalive_expiry=config.http_keepalive_expiry,
    )


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
        self.audio = Audio(self)
        self.fine_tunes = FineTunes(self)
        self.deployments = Deployments(self)
        self.rl = Rl(self)
        self.apps = Apps(self)
        self.tingwu = TingWu(self)
        self._agentstudio = None

    @property
    def reinforcement_learning(self) -> "Rl":
        """Long-form alias for ``client.rl``."""
        return self.rl

    @property
    def agentstudio(self):
        """Lazily mounted 1.x AgentStudio sub-SDK (design doc M3 3.8).

        The client is wrapped in a transparent proxy (design doc M3 3.12)
        that translates 1.x AgentStudio errors into the ``dashscope.v2``
        exception family, so ``except dashscope.v2.errors.DashScopeError``
        covers them. The proxy spoofs ``__class__`` and forwards
        attributes, so ``isinstance(client.agentstudio, AgentStudioClient)``
        and ``.transport`` access keep working; no 1.x code is touched.

        AgentStudio rides the shared 1.x transport, so its requests keep
        the unmarked 1.x module tag (``agentstudio``), not the v2 ``2``
        marker -- intentional, since this is the 1.x code path.
        """
        if self._agentstudio is None:
            # pylint: disable=import-outside-toplevel
            from dashscope.agentstudio import Client as AgentStudioClient
            from dashscope.v2.resources._agentstudio import wrap_agentstudio

            # agentstudio resolves its own base_url from workspace +
            # region (the v2 base_url does not propagate: the agentstudio
            # path differs), so forward both.
            self._agentstudio = wrap_agentstudio(
                AgentStudioClient(
                    api_key=self._config.api_key,
                    workspace=self._config.workspace,
                    region=self._config.region,
                ),
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
    ) -> WsConnection:
        """Open a shared duplex websocket carrying multiple tasks.

        Credentials, region and timeout come from this client's config.
        Pass the connection as ``connection=`` to the realtime audio
        sessions (``audio.asr.recognize`` / ``audio.tts.connect`` /
        ``audio.translation.recognize`` / ``tingwu.realtime``) to run
        tasks concurrently or back-to-back on one socket, saving the
        handshake per task -- the 1.x Java-side connection-reuse pattern
        for latency-sensitive audio traffic. The caller owns the
        connection and closes it via ``WsConnection.close``.
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
        self.audio = AsyncAudio(self)
        self.fine_tunes = AsyncFineTunes(self)
        self.deployments = AsyncDeployments(self)
        self.rl = AsyncRl(self)
        self.apps = AsyncApps(self)
        self.tingwu = AsyncTingWu(self)
        self._agentstudio = None

    @property
    def reinforcement_learning(self) -> "AsyncRl":
        """Long-form alias for ``client.rl``."""
        return self.rl

    @property
    def agentstudio(self):
        """Lazily mounted 1.x AgentStudio sub-SDK (design doc M3 3.8).

        See :attr:`Client.agentstudio`: the returned client is wrapped in
        a transparent proxy that aligns 1.x AgentStudio errors with the
        ``dashscope.v2`` exception family (design doc M3 3.12).
        """
        if self._agentstudio is None:
            # pylint: disable=import-outside-toplevel
            from dashscope.agentstudio import AsyncClient as AsClient
            from dashscope.v2.resources._agentstudio import wrap_agentstudio

            # See Client.agentstudio: agentstudio builds its own base_url
            # from workspace + region, so forward both.
            self._agentstudio = wrap_agentstudio(
                AsClient(
                    api_key=self._config.api_key,
                    workspace=self._config.workspace,
                    region=self._config.region,
                ),
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
