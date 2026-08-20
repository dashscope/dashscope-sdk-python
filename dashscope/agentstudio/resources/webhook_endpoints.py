# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Managed Agent webhook endpoint resource classes."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from dashscope.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
    build_page,
)
from dashscope.agentstudio.resources._helpers import (
    _coerce_webhook_endpoint,
    _coerce_webhook_event,
    _coerce_webhook_secret_reset,
)
from dashscope.agentstudio.types import (
    WebhookEndpoint,
    WebhookEndpointList,
    WebhookEvent,
    WebhookSecretReset,
)
from dashscope.agentstudio.types.params import (
    WebhookEndpointCreateParams,
    WebhookEndpointUpdateParams,
    WebhookEventListParams,
)


# Relative path of the webhook endpoint collection.
_PATH_WEBHOOK_ENDPOINTS = "/webhook_endpoints"


class WebhookEndpoints:
    """Synchronous Managed Agent webhook endpoint APIs."""

    def __init__(self, client) -> None:
        self._client = client

    def create(
        self,
        *,
        url: str,
        events: Sequence[str],
        description: Optional[str] = None,
    ) -> WebhookEndpoint:
        body = WebhookEndpointCreateParams(
            url=url,
            events=list(events),
            description=description,
        ).to_dict()
        response = self._client.transport.request(
            "POST",
            _PATH_WEBHOOK_ENDPOINTS,
            json=body,
        )
        return _coerce_webhook_endpoint(response.data)

    def retrieve(self, webhook_id: str) -> WebhookEndpoint:
        response = self._client.transport.request(
            "GET",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}",
        )
        return _coerce_webhook_endpoint(response.data)

    get = retrieve

    def list(self) -> WebhookEndpointList:
        response = self._client.transport.request(
            "GET",
            _PATH_WEBHOOK_ENDPOINTS,
        )
        return WebhookEndpointList(**response.data)

    def update(
        self,
        webhook_id: str,
        *,
        description: Optional[str] = None,
        url: Optional[str] = None,
        events: Optional[Sequence[str]] = None,
    ) -> WebhookEndpoint:
        body = WebhookEndpointUpdateParams(
            description=description,
            url=url,
            events=list(events) if events is not None else None,
        ).to_dict()
        response = self._client.transport.request(
            "PUT",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}",
            json=body,
        )
        return _coerce_webhook_endpoint(response.data)

    def delete(self, webhook_id: str) -> Dict[str, Any]:
        response = self._client.transport.request(
            "DELETE",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}",
        )
        return response.data

    def enable(self, webhook_id: str) -> WebhookEndpoint:
        response = self._client.transport.request(
            "POST",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}/enable",
        )
        return _coerce_webhook_endpoint(response.data)

    def disable(self, webhook_id: str) -> WebhookEndpoint:
        response = self._client.transport.request(
            "POST",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}/disable",
        )
        return _coerce_webhook_endpoint(response.data)

    def test(self, webhook_id: str) -> WebhookEvent:
        response = self._client.transport.request(
            "POST",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}/test",
        )
        return _coerce_webhook_event(response.data)

    def reset_secret(self, webhook_id: str) -> WebhookSecretReset:
        response = self._client.transport.request(
            "POST",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}/reset_secret",
        )
        return _coerce_webhook_secret_reset(response.data)

    def list_events(
        self,
        webhook_id: str,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> CursorPage[WebhookEvent]:
        params = WebhookEventListParams(limit=limit, page=page).to_dict()
        response = self._client.transport.request(
            "GET",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}/events",
            params=params,
        )
        return build_page(
            payload=response.data,
            item_factory=_coerce_webhook_event,
            request_id=response.request_id,
            fetch_next=lambda cursor: self.list_events(
                webhook_id,
                limit=limit,
                page=cursor,
            ),
        )


class AsyncWebhookEndpoints:
    """Asynchronous Managed Agent webhook endpoint APIs."""

    def __init__(self, client) -> None:
        self._client = client

    async def create(
        self,
        *,
        url: str,
        events: Sequence[str],
        description: Optional[str] = None,
    ) -> WebhookEndpoint:
        body = WebhookEndpointCreateParams(
            url=url,
            events=list(events),
            description=description,
        ).to_dict()
        response = await self._client.transport.request(
            "POST",
            _PATH_WEBHOOK_ENDPOINTS,
            json=body,
        )
        return _coerce_webhook_endpoint(response.data)

    async def retrieve(self, webhook_id: str) -> WebhookEndpoint:
        response = await self._client.transport.request(
            "GET",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}",
        )
        return _coerce_webhook_endpoint(response.data)

    get = retrieve

    async def list(self) -> WebhookEndpointList:
        response = await self._client.transport.request(
            "GET",
            _PATH_WEBHOOK_ENDPOINTS,
        )
        return WebhookEndpointList(**response.data)

    async def update(
        self,
        webhook_id: str,
        *,
        description: Optional[str] = None,
        url: Optional[str] = None,
        events: Optional[Sequence[str]] = None,
    ) -> WebhookEndpoint:
        body = WebhookEndpointUpdateParams(
            description=description,
            url=url,
            events=list(events) if events is not None else None,
        ).to_dict()
        response = await self._client.transport.request(
            "PUT",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}",
            json=body,
        )
        return _coerce_webhook_endpoint(response.data)

    async def delete(self, webhook_id: str) -> Dict[str, Any]:
        response = await self._client.transport.request(
            "DELETE",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}",
        )
        return response.data

    async def enable(self, webhook_id: str) -> WebhookEndpoint:
        response = await self._client.transport.request(
            "POST",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}/enable",
        )
        return _coerce_webhook_endpoint(response.data)

    async def disable(self, webhook_id: str) -> WebhookEndpoint:
        response = await self._client.transport.request(
            "POST",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}/disable",
        )
        return _coerce_webhook_endpoint(response.data)

    async def test(self, webhook_id: str) -> WebhookEvent:
        response = await self._client.transport.request(
            "POST",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}/test",
        )
        return _coerce_webhook_event(response.data)

    async def reset_secret(self, webhook_id: str) -> WebhookSecretReset:
        response = await self._client.transport.request(
            "POST",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}/reset_secret",
        )
        return _coerce_webhook_secret_reset(response.data)

    async def list_events(
        self,
        webhook_id: str,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[WebhookEvent]:
        params = WebhookEventListParams(limit=limit, page=page).to_dict()
        response = await self._client.transport.request(
            "GET",
            f"{_PATH_WEBHOOK_ENDPOINTS}/{webhook_id}/events",
            params=params,
        )

        async def fetch_next(cursor: str) -> AsyncCursorPage[WebhookEvent]:
            return await self.list_events(
                webhook_id,
                limit=limit,
                page=cursor,
            )

        return build_page(
            payload=response.data,
            item_factory=_coerce_webhook_event,
            request_id=response.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )
