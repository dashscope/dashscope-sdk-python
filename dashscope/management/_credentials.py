# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Multi-tenant credential chain for the management surface.

Wraps the official ``alibabacloud_credentials`` client so a ToB backend
can hand the SDK any of: an explicit AK/SK (optionally an STS token), a
``Config`` dict for the richer provider types (ECS RAM role, RAM role
ARN, OIDC, credentials URI), a pre-built credentials ``Client``, or
nothing at all to fall through to the library's default chain
(environment, IMDS, ...). Resolution returns a frozen snapshot; the
library owns refresh, so an STS token that expires mid-process is
re-fetched on the next ``resolve()``.

No secret is ever logged here -- the resolved snapshot is handed
straight to the signer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from alibabacloud_credentials.client import Client as _CredentialsClient
from alibabacloud_credentials.models import Config as _CredentialsConfig

__all__ = ["CredentialResolver", "ResolvedCredential"]


@dataclass(frozen=True)
class ResolvedCredential:
    """One credential snapshot used to sign a single request."""

    access_key_id: str
    access_key_secret: str
    security_token: Optional[str] = None
    provider_name: Optional[str] = None


class CredentialResolver:
    """Resolve management credentials through the official chain.

    Precedence: a pre-built ``credential`` client wins, then an explicit
    ``config`` dict, then explicit ``access_key_id`` / ``access_key_secret``
    (with ``security_token`` selecting the STS provider type), then the
    library default chain. A *partial* explicit AK (id without secret, or
    a lone token) raises instead of falling through: silently signing with
    ambient host credentials would swap a tenant's identity.
    """

    def __init__(
        self,
        *,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        security_token: Optional[str] = None,
        credential: Optional[Any] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if credential is not None:
            self._client = credential
        elif config is not None:
            self._client = _CredentialsClient(
                _CredentialsConfig(**dict(config)),
            )
        elif access_key_id or access_key_secret or security_token:
            # Fail loudly on a partial explicit credential: the default
            # chain below picks up the host's env/IMDS identity, which in
            # a multi-tenant backend is a silent cross-tenant swap.
            if not (access_key_id and access_key_secret):
                raise ValueError(
                    "access_key_id and access_key_secret must be provided "
                    "together; refusing to fall back to ambient credentials",
                )
            self._client = _CredentialsClient(
                _CredentialsConfig(
                    type="sts" if security_token else "access_key",
                    access_key_id=access_key_id,
                    access_key_secret=access_key_secret,
                    security_token=security_token,
                ),
            )
        else:
            self._client = _CredentialsClient()

    def resolve(self) -> ResolvedCredential:
        """Fetch a credential snapshot synchronously."""
        return _snapshot(self._client.get_credential())

    async def resolve_async(self) -> ResolvedCredential:
        """Fetch a credential snapshot without blocking the event loop."""
        return _snapshot(await self._client.get_credential_async())


def _snapshot(model: Any) -> ResolvedCredential:
    return ResolvedCredential(
        access_key_id=model.access_key_id,
        access_key_secret=model.access_key_secret,
        security_token=model.security_token,
        provider_name=getattr(model, "provider_name", None),
    )
