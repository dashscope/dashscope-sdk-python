# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``client.api_keys`` -- list and inspect workspace API keys.

Read-only (G2). Even these read ops return the plaintext key value, so
``ApiKey.api_key_value`` is a ``SecretStr``: it is masked in ``repr``,
``model_dump`` and logs, and only ``.get_secret_value()`` reveals it.
Write ops (CreateApiKey / UpdateApiKey / EnableApiKey / DisableApiKey) and
the destructive ResetApiKey / DeleteApiKey are high-risk and deferred to
G3 / G4 -- not implemented here.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    Optional,
    Union,
    overload,
)

from dashscope.management._pagination import TokenPage
from dashscope.management._parse import parse_one, parse_token_page
from dashscope.management._resource import (
    AsyncResource,
    SyncResource,
    options_kwargs,
)
from dashscope.management.types import ApiKey
from dashscope.transport._options import request_options

if TYPE_CHECKING:
    from dashscope.transport._response import APIResponse

__all__ = ["ApiKeys", "AsyncApiKeys"]


class ApiKeys(SyncResource):
    @overload
    def list(
        self,
        *,
        workspace_id: Optional[str] = None,
        api_key_id: Optional[int] = None,
        description: Optional[str] = None,
        order_by: Optional[str] = None,
        order: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenPage[ApiKey]:
        ...

    @overload
    def list(
        self,
        *,
        workspace_id: Optional[str] = None,
        api_key_id: Optional[int] = None,
        description: Optional[str] = None,
        order_by: Optional[str] = None,
        order: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def list(
        self,
        *,
        workspace_id: Optional[str] = None,
        api_key_id: Optional[int] = None,
        description: Optional[str] = None,
        order_by: Optional[str] = None,
        order: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenPage[ApiKey], "APIResponse"]:
        """List API keys; ``api_key_value`` is masked on each entry."""
        filters = {
            "workspace_id": workspace_id,
            "api_key_id": api_key_id,
            "description": description,
            "order_by": order_by,
            "order": order,
            "next_token": next_token,
            "max_results": max_results,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = self._request("ListApiKeys", filters, request_options(**opts))
        if raw:
            return resp

        def fetch(token: str) -> TokenPage[ApiKey]:
            return self.list(  # type: ignore[call-overload]
                **{**filters, "next_token": token},
                **opts,
            )

        return parse_token_page(
            ApiKey,
            resp,
            data_field="apiKeys",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
        )

    @overload
    def get(
        self,
        api_key_id: int,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> ApiKey:
        ...

    @overload
    def get(
        self,
        api_key_id: int,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def get(
        self,
        api_key_id: int,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ApiKey, "APIResponse"]:
        """Fetch one API key by id; ``api_key_value`` stays masked."""
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = self._request(
            "GetApiKey",
            {"api_key_id": api_key_id},
            request_options(**opts),
        )
        if raw:
            return resp
        return parse_one(
            ApiKey,
            resp,
            data_field="apiKey",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
        )


class AsyncApiKeys(AsyncResource):
    @overload
    async def list(
        self,
        *,
        workspace_id: Optional[str] = None,
        api_key_id: Optional[int] = None,
        description: Optional[str] = None,
        order_by: Optional[str] = None,
        order: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenPage[ApiKey]:
        ...

    @overload
    async def list(
        self,
        *,
        workspace_id: Optional[str] = None,
        api_key_id: Optional[int] = None,
        description: Optional[str] = None,
        order_by: Optional[str] = None,
        order: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def list(
        self,
        *,
        workspace_id: Optional[str] = None,
        api_key_id: Optional[int] = None,
        description: Optional[str] = None,
        order_by: Optional[str] = None,
        order: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenPage[ApiKey], "APIResponse"]:
        filters = {
            "workspace_id": workspace_id,
            "api_key_id": api_key_id,
            "description": description,
            "order_by": order_by,
            "order": order,
            "next_token": next_token,
            "max_results": max_results,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = await self._request(
            "ListApiKeys",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp

        async def fetch(token: str) -> TokenPage[ApiKey]:
            return await self.list(  # type: ignore[call-overload]
                **{**filters, "next_token": token},
                **opts,
            )

        return parse_token_page(
            ApiKey,
            resp,
            data_field="apiKeys",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
        )

    @overload
    async def get(
        self,
        api_key_id: int,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> ApiKey:
        ...

    @overload
    async def get(
        self,
        api_key_id: int,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def get(
        self,
        api_key_id: int,
        *,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[ApiKey, "APIResponse"]:
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = await self._request(
            "GetApiKey",
            {"api_key_id": api_key_id},
            request_options(**opts),
        )
        if raw:
            return resp
        return parse_one(
            ApiKey,
            resp,
            data_field="apiKey",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
        )
