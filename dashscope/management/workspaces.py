# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``client.workspaces`` -- business spaces and their model catalog.

Read-only (G2). ``workspaces.list`` pages by ``nextToken``; the nested
``models`` / ``model_limits`` / ``model_permissions`` resources list the
model catalog, per-model rate limits and per-model authorization scopes.
Write ops (CreateWorkspace / DeleteWorkspace / UpdateModelLimits /
UpdateModelPermissions) are high-risk and deferred to G3 -- not implemented
here.
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
from dashscope.management._parse import parse_token_page
from dashscope.management._resource import (
    AsyncResource,
    SyncResource,
    options_kwargs,
)
from dashscope.management.types import (
    ManagementModel,
    ModelLimit,
    ModelPermission,
    Workspace,
)
from dashscope.transport._options import request_options

if TYPE_CHECKING:
    from dashscope.transport._response import APIResponse

__all__ = [
    "AsyncModelLimits",
    "AsyncModelPermissions",
    "AsyncModels",
    "AsyncWorkspaces",
    "ModelLimits",
    "ModelPermissions",
    "Models",
    "Workspaces",
]


class Workspaces(SyncResource):
    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.models = Models(client)
        self.model_limits = ModelLimits(client)
        self.model_permissions = ModelPermissions(client)

    @overload
    def list(
        self,
        *,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        workspace_name: Optional[str] = None,
        workspace_id: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenPage[Workspace]:
        ...

    @overload
    def list(
        self,
        *,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        workspace_name: Optional[str] = None,
        workspace_id: Optional[str] = None,
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
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        workspace_name: Optional[str] = None,
        workspace_id: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenPage[Workspace], "APIResponse"]:
        """List business spaces, paginated by ``nextToken``."""
        filters = {
            "next_token": next_token,
            "max_results": max_results,
            "workspace_name": workspace_name,
            "workspace_id": workspace_id,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = self._request(
            "ListWorkspaces",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp

        def fetch(token: str) -> TokenPage[Workspace]:
            return self.list(  # type: ignore[call-overload]
                **{**filters, "next_token": token},
                **opts,
            )

        return parse_token_page(
            Workspace,
            resp,
            data_field="workspaces",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
        )


class Models(SyncResource):
    @overload
    def list(
        self,
        *,
        providers: Optional[Any] = None,
        capabilities: Optional[Any] = None,
        features: Optional[Any] = None,
        language: Optional[str] = None,
        context_window: Optional[int] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenPage[ManagementModel]:
        ...

    @overload
    def list(
        self,
        *,
        providers: Optional[Any] = None,
        capabilities: Optional[Any] = None,
        features: Optional[Any] = None,
        language: Optional[str] = None,
        context_window: Optional[int] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
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
        providers: Optional[Any] = None,
        capabilities: Optional[Any] = None,
        features: Optional[Any] = None,
        language: Optional[str] = None,
        context_window: Optional[int] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenPage[ManagementModel], "APIResponse"]:
        """List the model catalog (public; not scoped to a workspace)."""
        filters = {
            "providers": providers,
            "capabilities": capabilities,
            "features": features,
            "language": language,
            "context_window": context_window,
            "name": name,
            "model": model,
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
        resp = self._request("ListModels", filters, request_options(**opts))
        if raw:
            return resp

        def fetch(token: str) -> TokenPage[ManagementModel]:
            return self.list(**{**filters, "next_token": token}, **opts)

        return parse_token_page(
            ManagementModel,
            resp,
            data_field="models",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
        )


class ModelLimits(SyncResource):
    @overload
    def list(
        self,
        *,
        workspace_id: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenPage[ModelLimit]:
        ...

    @overload
    def list(
        self,
        *,
        workspace_id: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
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
        name: Optional[str] = None,
        model: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenPage[ModelLimit], "APIResponse"]:
        """List per-model rate limits."""
        filters = {
            "workspace_id": workspace_id,
            "name": name,
            "model": model,
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
        resp = self._request(
            "ListModelLimits",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp

        def fetch(token: str) -> TokenPage[ModelLimit]:
            return self.list(  # type: ignore[call-overload]
                **{**filters, "next_token": token},
                **opts,
            )

        return parse_token_page(
            ModelLimit,
            resp,
            data_field="limits",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
        )


class ModelPermissions(SyncResource):
    @overload
    def list(
        self,
        *,
        workspace_id: str,
        authorization_scope: Optional[str] = None,
        model_action: Optional[str] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenPage[ModelPermission]:
        ...

    @overload
    def list(
        self,
        *,
        workspace_id: str,
        authorization_scope: Optional[str] = None,
        model_action: Optional[str] = None,
        filter_: Optional[Mapping[str, Any]] = None,
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
        workspace_id: str,
        authorization_scope: Optional[str] = None,
        model_action: Optional[str] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenPage[ModelPermission], "APIResponse"]:
        """List per-model authorization scopes (``workspace_id`` required)."""
        filters = {
            "workspace_id": workspace_id,
            "authorization_scope": authorization_scope,
            "model_action": model_action,
            "filter_": filter_,
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
        resp = self._request(
            "ListModelPermissions",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp

        def fetch(token: str) -> TokenPage[ModelPermission]:
            return self.list(  # type: ignore[call-overload]
                **{**filters, "next_token": token},
                **opts,
            )

        return parse_token_page(
            ModelPermission,
            resp,
            data_field="list",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
        )


class AsyncWorkspaces(AsyncResource):
    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.models = AsyncModels(client)
        self.model_limits = AsyncModelLimits(client)
        self.model_permissions = AsyncModelPermissions(client)

    @overload
    async def list(
        self,
        *,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        workspace_name: Optional[str] = None,
        workspace_id: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenPage[Workspace]:
        ...

    @overload
    async def list(
        self,
        *,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        workspace_name: Optional[str] = None,
        workspace_id: Optional[str] = None,
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
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        workspace_name: Optional[str] = None,
        workspace_id: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenPage[Workspace], "APIResponse"]:
        filters = {
            "next_token": next_token,
            "max_results": max_results,
            "workspace_name": workspace_name,
            "workspace_id": workspace_id,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = await self._request(
            "ListWorkspaces",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp

        async def fetch(token: str) -> TokenPage[Workspace]:
            return await self.list(  # type: ignore[call-overload]
                **{**filters, "next_token": token},
                **opts,
            )

        return parse_token_page(
            Workspace,
            resp,
            data_field="workspaces",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
        )


class AsyncModels(AsyncResource):
    @overload
    async def list(
        self,
        *,
        providers: Optional[Any] = None,
        capabilities: Optional[Any] = None,
        features: Optional[Any] = None,
        language: Optional[str] = None,
        context_window: Optional[int] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenPage[ManagementModel]:
        ...

    @overload
    async def list(
        self,
        *,
        providers: Optional[Any] = None,
        capabilities: Optional[Any] = None,
        features: Optional[Any] = None,
        language: Optional[str] = None,
        context_window: Optional[int] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
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
        providers: Optional[Any] = None,
        capabilities: Optional[Any] = None,
        features: Optional[Any] = None,
        language: Optional[str] = None,
        context_window: Optional[int] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenPage[ManagementModel], "APIResponse"]:
        filters = {
            "providers": providers,
            "capabilities": capabilities,
            "features": features,
            "language": language,
            "context_window": context_window,
            "name": name,
            "model": model,
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
            "ListModels",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp

        async def fetch(token: str) -> TokenPage[ManagementModel]:
            return await self.list(
                **{**filters, "next_token": token},
                **opts,
            )

        return parse_token_page(
            ManagementModel,
            resp,
            data_field="models",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
        )


class AsyncModelLimits(AsyncResource):
    @overload
    async def list(
        self,
        *,
        workspace_id: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenPage[ModelLimit]:
        ...

    @overload
    async def list(
        self,
        *,
        workspace_id: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[str] = None,
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
        name: Optional[str] = None,
        model: Optional[str] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenPage[ModelLimit], "APIResponse"]:
        filters = {
            "workspace_id": workspace_id,
            "name": name,
            "model": model,
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
            "ListModelLimits",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp

        async def fetch(token: str) -> TokenPage[ModelLimit]:
            return await self.list(  # type: ignore[call-overload]
                **{**filters, "next_token": token},
                **opts,
            )

        return parse_token_page(
            ModelLimit,
            resp,
            data_field="limits",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
        )


class AsyncModelPermissions(AsyncResource):
    @overload
    async def list(
        self,
        *,
        workspace_id: str,
        authorization_scope: Optional[str] = None,
        model_action: Optional[str] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> TokenPage[ModelPermission]:
        ...

    @overload
    async def list(
        self,
        *,
        workspace_id: str,
        authorization_scope: Optional[str] = None,
        model_action: Optional[str] = None,
        filter_: Optional[Mapping[str, Any]] = None,
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
        workspace_id: str,
        authorization_scope: Optional[str] = None,
        model_action: Optional[str] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        next_token: Optional[str] = None,
        max_results: Optional[int] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[TokenPage[ModelPermission], "APIResponse"]:
        filters = {
            "workspace_id": workspace_id,
            "authorization_scope": authorization_scope,
            "model_action": model_action,
            "filter_": filter_,
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
            "ListModelPermissions",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp

        async def fetch(token: str) -> TokenPage[ModelPermission]:
            return await self.list(  # type: ignore[call-overload]
                **{**filters, "next_token": token},
                **opts,
            )

        return parse_token_page(
            ModelPermission,
            resp,
            data_field="list",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
            fetch_next=fetch,
        )
