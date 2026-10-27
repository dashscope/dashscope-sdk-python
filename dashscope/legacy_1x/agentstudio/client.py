# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The historical AgentStudio Client facade.

Business resources and HTTP mechanics live in ``dashscope.applications``;
this facade preserves 1.x credential resolution and exception classes.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import httpx

from dashscope.applications.agentstudio.client import (  # noqa: F401
    AsyncClient as _CoreAsyncClient,
)
from dashscope.applications.agentstudio.client import Client as _CoreClient
from dashscope.applications.agentstudio import client as _core_client
from dashscope.applications.agentstudio.constants import (
    AGENTSTUDIO_MAX_RETRIES,
)
from dashscope.legacy_1x.agentstudio import exceptions
from dashscope.legacy_1x.agentstudio.streaming import (
    AsyncEventStream,
    EventStream,
)
from dashscope.legacy_1x.agentstudio.transport import (
    AsyncTransport,
    SyncTransport,
)
from dashscope.legacy_1x.common.api_key import get_default_api_key

# Historical module-level helpers remain importable from this facade.
# pylint: disable=protected-access
_resolve_base_url = _core_client._resolve_base_url
_user_agent = _core_client._user_agent


class Client(_CoreClient):
    """1.x AgentStudio client backed by the shared application core."""

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
        super().__init__(
            api_key=api_key,
            workspace=workspace,
            region=region,
            base_url=base_url,
            uid=uid,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            _exceptions_module=exceptions,
            _api_key_resolver=get_default_api_key,
            _transport_cls=SyncTransport,
            _event_stream_cls=EventStream,
        )


class AsyncClient(_CoreAsyncClient):
    """Async 1.x AgentStudio client backed by the shared core."""

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
        super().__init__(
            api_key=api_key,
            workspace=workspace,
            region=region,
            base_url=base_url,
            uid=uid,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            _exceptions_module=exceptions,
            _api_key_resolver=get_default_api_key,
            _transport_cls=AsyncTransport,
            _event_stream_cls=AsyncEventStream,
        )


__all__ = ["AsyncClient", "Client"]
