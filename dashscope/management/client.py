# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``ManagementClient`` / ``AsyncManagementClient`` -- the RAM OpenAPI client.

The ToB integration entry point: a backend service constructs one client
per tenant credential (AK/SK, STS, RAM role, OIDC, or the default chain)
and reaches workspace / API-key / billing / TokenPlan management without
logging into the aliyun console. This revision is read-only (G1/G2/G5);
write ops are high-risk and deferred.

The client is region-bound (``modelstudio.{region}.aliyuncs.com``);
``with_options(region=...)`` derives a new client with its own connection
pool for cross-region calls. Region strings are validated as safe DNS
labels; an unknown-but-safe region warns rather than hard-fails, leaving
custom endpoints usable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

import httpx

from dashscope.management._credentials import CredentialResolver
from dashscope.management._redaction import get_management_logger
from dashscope.management._transport import AsyncTransport, SyncTransport
from dashscope.management.api_keys import ApiKeys, AsyncApiKeys
from dashscope.management.billing import AsyncBilling, Billing
from dashscope.management.token_plan import AsyncTokenPlan, TokenPlan
from dashscope.management.workspaces import AsyncWorkspaces, Workspaces

__all__ = ["AsyncManagementClient", "ManagementClient", "ManagementConfig"]

logger = get_management_logger("management.client")

DEFAULT_REGION = "cn-beijing"
KNOWN_REGIONS = frozenset(
    {
        "cn-beijing",
        "cn-hongkong",
        "ap-southeast-1",
        "us-east-1",
        "eu-central-1",
    },
)
_REGION_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


def _validate_region(region: str) -> None:
    if not _REGION_LABEL.match(region):
        raise ValueError(
            f"invalid region {region!r}: expected a DNS label like 'cn-beijing'",
        )
    if region not in KNOWN_REGIONS:
        logger.warning(
            "region %r is not a known ModelStudio region; using it as given "
            "(endpoint modelstudio.%s.aliyuncs.com)",
            region,
            region,
        )


@dataclass(frozen=True)
class ManagementConfig:
    """Immutable client settings, read by resources and ``with_options``."""

    region: str = DEFAULT_REGION
    endpoint: Optional[str] = None
    raise_on_error: bool = True
    strict_validation: bool = False
    timeout: Union[float, httpx.Timeout, None] = None
    max_retries: int = 2
    user_agent: Optional[str] = None


class ManagementClient:
    """Synchronous RAM management client (read-only)."""

    def __init__(
        self,
        *,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        security_token: Optional[str] = None,
        region: str = DEFAULT_REGION,
        credential: Optional[Any] = None,
        config: Optional[Mapping[str, Any]] = None,
        endpoint: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        max_retries: int = 2,
        http_client: Optional[httpx.Client] = None,
        user_agent: Optional[str] = None,
        raise_on_error: bool = True,
        strict_validation: bool = False,
        limits: Optional[httpx.Limits] = None,
        _resolver: Optional[CredentialResolver] = None,
    ) -> None:
        _validate_region(region)
        self._resolver = _resolver or CredentialResolver(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            security_token=security_token,
            credential=credential,
            config=config,
        )
        self.config = ManagementConfig(
            region=region,
            endpoint=endpoint,
            raise_on_error=raise_on_error,
            strict_validation=strict_validation,
            timeout=timeout,
            max_retries=max_retries,
            user_agent=user_agent,
        )
        self._transport = SyncTransport(
            region=region,
            resolver=self._resolver,
            endpoint=endpoint,
            user_agent=user_agent,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            limits=limits,
        )
        self.workspaces = Workspaces(self)
        self.api_keys = ApiKeys(self)
        self.billing = Billing(self)
        self.token_plan = TokenPlan(self)

    @property
    def transport(self) -> SyncTransport:
        return self._transport

    @property
    def region(self) -> str:
        return self.config.region

    @property
    def raise_on_error(self) -> bool:
        return self.config.raise_on_error

    def with_options(
        self,
        *,
        region: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        max_retries: Optional[int] = None,
        raise_on_error: Optional[bool] = None,
        strict_validation: Optional[bool] = None,
    ) -> "ManagementClient":
        """Derive a new client (own connection pool) sharing the credential."""
        cfg = self.config
        return ManagementClient(
            _resolver=self._resolver,
            region=region if region is not None else cfg.region,
            endpoint=endpoint if endpoint is not None else cfg.endpoint,
            timeout=timeout if timeout is not None else cfg.timeout,
            max_retries=(
                max_retries if max_retries is not None else cfg.max_retries
            ),
            user_agent=cfg.user_agent,
            raise_on_error=(
                raise_on_error
                if raise_on_error is not None
                else cfg.raise_on_error
            ),
            strict_validation=(
                strict_validation
                if strict_validation is not None
                else cfg.strict_validation
            ),
        )

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "ManagementClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncManagementClient:
    """Asynchronous RAM management client (read-only)."""

    def __init__(
        self,
        *,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        security_token: Optional[str] = None,
        region: str = DEFAULT_REGION,
        credential: Optional[Any] = None,
        config: Optional[Mapping[str, Any]] = None,
        endpoint: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        max_retries: int = 2,
        http_client: Optional[httpx.AsyncClient] = None,
        user_agent: Optional[str] = None,
        raise_on_error: bool = True,
        strict_validation: bool = False,
        limits: Optional[httpx.Limits] = None,
        _resolver: Optional[CredentialResolver] = None,
    ) -> None:
        _validate_region(region)
        self._resolver = _resolver or CredentialResolver(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            security_token=security_token,
            credential=credential,
            config=config,
        )
        self.config = ManagementConfig(
            region=region,
            endpoint=endpoint,
            raise_on_error=raise_on_error,
            strict_validation=strict_validation,
            timeout=timeout,
            max_retries=max_retries,
            user_agent=user_agent,
        )
        self._transport = AsyncTransport(
            region=region,
            resolver=self._resolver,
            endpoint=endpoint,
            user_agent=user_agent,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            limits=limits,
        )
        self.workspaces = AsyncWorkspaces(self)
        self.api_keys = AsyncApiKeys(self)
        self.billing = AsyncBilling(self)
        self.token_plan = AsyncTokenPlan(self)

    @property
    def transport(self) -> AsyncTransport:
        return self._transport

    @property
    def region(self) -> str:
        return self.config.region

    @property
    def raise_on_error(self) -> bool:
        return self.config.raise_on_error

    def with_options(
        self,
        *,
        region: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        max_retries: Optional[int] = None,
        raise_on_error: Optional[bool] = None,
        strict_validation: Optional[bool] = None,
    ) -> "AsyncManagementClient":
        """Derive a new async client (own pool) sharing the credential."""
        cfg = self.config
        return AsyncManagementClient(
            _resolver=self._resolver,
            region=region if region is not None else cfg.region,
            endpoint=endpoint if endpoint is not None else cfg.endpoint,
            timeout=timeout if timeout is not None else cfg.timeout,
            max_retries=(
                max_retries if max_retries is not None else cfg.max_retries
            ),
            user_agent=cfg.user_agent,
            raise_on_error=(
                raise_on_error
                if raise_on_error is not None
                else cfg.raise_on_error
            ),
            strict_validation=(
                strict_validation
                if strict_validation is not None
                else cfg.strict_validation
            ),
        )

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> "AsyncManagementClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
