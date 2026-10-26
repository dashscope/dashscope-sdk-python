# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Vaults and Credentials resource classes."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from dashscope.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
    build_page,
)
from dashscope.agentstudio.resources._helpers import (
    _coerce_credential,
    _coerce_vault,
)
from dashscope.agentstudio.types import (
    Credential,
    DeleteResponse,
    Vault,
)
from dashscope.agentstudio.types.params import (
    CredentialCreateParams,
    CredentialListParams,
    CredentialUpdateParams,
    VaultCreateParams,
    VaultListParams,
    VaultUpdateParams,
)


_PATH_VAULTS = "/vaults"


# ---------------------------------------------------------------------------
# Credentials (sub-resource of Vaults)
# ---------------------------------------------------------------------------


class Credentials:
    def __init__(self, client) -> None:
        self._client = client

    def create(
        self,
        vault_id: str,
        *,
        auth: Any,
        display_name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Credential:
        body = CredentialCreateParams(
            auth=auth,
            display_name=display_name,
            metadata=metadata,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_VAULTS}/{vault_id}/credentials",
            json=body,
        )
        return _coerce_credential(resp.data)

    def retrieve(
        self,
        vault_id: str,
        credential_id: str,
    ) -> Credential:
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_VAULTS}/{vault_id}/credentials" f"/{credential_id}",
        )
        return _coerce_credential(resp.data)

    get = retrieve  # type: ignore[assignment]

    def update(
        self,
        vault_id: str,
        credential_id: str,
        *,
        auth: Any = None,
        display_name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Credential:
        body = CredentialUpdateParams(
            auth=auth,
            display_name=display_name,
            metadata=metadata,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_VAULTS}/{vault_id}/credentials" f"/{credential_id}",
            json=body,
        )
        return _coerce_credential(resp.data)

    def list(
        self,
        vault_id: str,
        *,
        include_archived: Optional[bool] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> CursorPage[Credential]:
        params = CredentialListParams(
            include_archived=include_archived,
            limit=limit,
            page=page,
        ).to_dict()
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_VAULTS}/{vault_id}/credentials",
            params=params,
        )
        return build_page(
            payload=resp.data,
            item_factory=_coerce_credential,
            request_id=resp.request_id,
            fetch_next=lambda nxt: self.list(
                vault_id,
                include_archived=include_archived,
                limit=limit,
                page=nxt,
            ),
        )

    def delete(
        self,
        vault_id: str,
        credential_id: str,
    ) -> DeleteResponse:
        resp = self._client.transport.request(
            "DELETE",
            f"{_PATH_VAULTS}/{vault_id}/credentials" f"/{credential_id}",
        )
        return DeleteResponse(**resp.data)

    def archive(
        self,
        vault_id: str,
        credential_id: str,
    ) -> Credential:
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_VAULTS}/{vault_id}/credentials"
            f"/{credential_id}/archive",
        )
        return _coerce_credential(resp.data)


# ---------------------------------------------------------------------------
# Vaults (parent resource)
# ---------------------------------------------------------------------------


class Vaults:
    def __init__(self, client) -> None:
        self._client = client
        self.credentials = Credentials(client)

    def create(
        self,
        *,
        display_name: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Vault:
        body = VaultCreateParams(
            display_name=display_name,
            metadata=metadata,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            _PATH_VAULTS,
            json=body,
        )
        return _coerce_vault(resp.data)

    def retrieve(self, vault_id: str) -> Vault:
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_VAULTS}/{vault_id}",
        )
        return _coerce_vault(resp.data)

    get = retrieve  # type: ignore[assignment]

    def update(
        self,
        vault_id: str,
        *,
        display_name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Vault:
        body = VaultUpdateParams(
            display_name=display_name,
            metadata=metadata,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_VAULTS}/{vault_id}",
            json=body,
        )
        return _coerce_vault(resp.data)

    def list(
        self,
        *,
        keyword: Optional[str] = None,
        include_archived: Optional[bool] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> CursorPage[Vault]:
        params = VaultListParams(
            keyword=keyword,
            include_archived=include_archived,
            limit=limit,
            page=page,
        ).to_dict()
        resp = self._client.transport.request(
            "GET",
            _PATH_VAULTS,
            params=params,
        )
        return build_page(
            payload=resp.data,
            item_factory=_coerce_vault,
            request_id=resp.request_id,
            fetch_next=lambda nxt: self.list(
                keyword=keyword,
                include_archived=include_archived,
                limit=limit,
                page=nxt,
            ),
        )

    def delete(self, vault_id: str) -> DeleteResponse:
        resp = self._client.transport.request(
            "DELETE",
            f"{_PATH_VAULTS}/{vault_id}",
        )
        return DeleteResponse(**resp.data)

    def archive(self, vault_id: str) -> Vault:
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_VAULTS}/{vault_id}/archive",
        )
        return _coerce_vault(resp.data)


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


class AsyncCredentials:
    def __init__(self, client) -> None:
        self._client = client

    async def create(
        self,
        vault_id: str,
        *,
        auth: Any,
        display_name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Credential:
        body = CredentialCreateParams(
            auth=auth,
            display_name=display_name,
            metadata=metadata,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_VAULTS}/{vault_id}/credentials",
            json=body,
        )
        return _coerce_credential(resp.data)

    async def retrieve(
        self,
        vault_id: str,
        credential_id: str,
    ) -> Credential:
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_VAULTS}/{vault_id}/credentials" f"/{credential_id}",
        )
        return _coerce_credential(resp.data)

    get = retrieve  # type: ignore[assignment]

    async def update(
        self,
        vault_id: str,
        credential_id: str,
        *,
        auth: Any = None,
        display_name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Credential:
        body = CredentialUpdateParams(
            auth=auth,
            display_name=display_name,
            metadata=metadata,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_VAULTS}/{vault_id}/credentials" f"/{credential_id}",
            json=body,
        )
        return _coerce_credential(resp.data)

    async def list(
        self,
        vault_id: str,
        *,
        include_archived: Optional[bool] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[Credential]:
        params = CredentialListParams(
            include_archived=include_archived,
            limit=limit,
            page=page,
        ).to_dict()
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_VAULTS}/{vault_id}/credentials",
            params=params,
        )

        async def fetch_next(
            nxt: str,
        ) -> AsyncCursorPage[Credential]:
            return await self.list(
                vault_id,
                include_archived=include_archived,
                limit=limit,
                page=nxt,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_credential,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def delete(
        self,
        vault_id: str,
        credential_id: str,
    ) -> DeleteResponse:
        resp = await self._client.transport.request(
            "DELETE",
            f"{_PATH_VAULTS}/{vault_id}/credentials" f"/{credential_id}",
        )
        return DeleteResponse(**resp.data)

    async def archive(
        self,
        vault_id: str,
        credential_id: str,
    ) -> Credential:
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_VAULTS}/{vault_id}/credentials"
            f"/{credential_id}/archive",
        )
        return _coerce_credential(resp.data)


class AsyncVaults:
    def __init__(self, client) -> None:
        self._client = client
        self.credentials = AsyncCredentials(client)

    async def create(
        self,
        *,
        display_name: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Vault:
        body = VaultCreateParams(
            display_name=display_name,
            metadata=metadata,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            _PATH_VAULTS,
            json=body,
        )
        return _coerce_vault(resp.data)

    async def retrieve(self, vault_id: str) -> Vault:
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_VAULTS}/{vault_id}",
        )
        return _coerce_vault(resp.data)

    get = retrieve  # type: ignore[assignment]

    async def update(
        self,
        vault_id: str,
        *,
        display_name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Vault:
        body = VaultUpdateParams(
            display_name=display_name,
            metadata=metadata,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_VAULTS}/{vault_id}",
            json=body,
        )
        return _coerce_vault(resp.data)

    async def list(
        self,
        *,
        keyword: Optional[str] = None,
        include_archived: Optional[bool] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[Vault]:
        params = VaultListParams(
            keyword=keyword,
            include_archived=include_archived,
            limit=limit,
            page=page,
        ).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _PATH_VAULTS,
            params=params,
        )

        async def fetch_next(
            nxt: str,
        ) -> AsyncCursorPage[Vault]:
            return await self.list(
                keyword=keyword,
                include_archived=include_archived,
                limit=limit,
                page=nxt,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_vault,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def delete(self, vault_id: str) -> DeleteResponse:
        resp = await self._client.transport.request(
            "DELETE",
            f"{_PATH_VAULTS}/{vault_id}",
        )
        return DeleteResponse(**resp.data)

    async def archive(self, vault_id: str) -> Vault:
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_VAULTS}/{vault_id}/archive",
        )
        return _coerce_vault(resp.data)
