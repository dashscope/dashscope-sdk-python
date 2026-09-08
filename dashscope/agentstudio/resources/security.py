# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Security center read interface.

Exposes the two highest-value read endpoints: ``GET /security/overview``
(24h dashboard) and ``GET /security/agent_logs`` (alert list). Other
security endpoints (asset summary, policies, alert detail, export,
authorization, activation) are not surfaced here.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashscope.agentstudio.types import SecurityAlertList, SecurityOverview
from dashscope.agentstudio.types.params import SecurityListAgentLogsParams

_PATH_SECURITY_OVERVIEW = "/security/overview"
_PATH_SECURITY_AGENT_LOGS = "/security/agent_logs"


def _overview_from_payload(payload: Mapping[str, Any]) -> SecurityOverview:
    return SecurityOverview(**dict(payload))


def _alert_list_from_payload(payload: Mapping[str, Any]) -> SecurityAlertList:
    return SecurityAlertList(**dict(payload))


class Security:
    """Security center read interface (overview + agent logs)."""

    def __init__(self, client) -> None:
        self._client = client

    def overview(self) -> SecurityOverview:
        """24h protection dashboard (``GET /security/overview``)."""
        resp = self._client.transport.request(
            "GET",
            _PATH_SECURITY_OVERVIEW,
        )
        return _overview_from_payload(resp.data)

    def list_agent_logs(
        self,
        *,
        current_page: Optional[int] = None,
        page_size: Optional[int] = None,
        risk_level: Optional[str] = None,
        status: Optional[str] = None,
        risk_name: Optional[str] = None,
        app_name: Optional[str] = None,
        asset_type: Optional[str] = None,
        vendor: Optional[str] = None,
        order_by: Optional[str] = None,
        order: Optional[str] = None,
        lang: Optional[str] = None,
        status_list: Optional[Sequence[str]] = None,
    ) -> SecurityAlertList:
        """Alert list (``GET /security/agent_logs``).

        Page-number pagination: pass ``current_page + 1`` for the next
        page, or use the returned ``next_page`` cursor if non-null.
        ``check_time`` / ``handle_time`` on rows are millisecond strings.
        """
        params = SecurityListAgentLogsParams(
            current_page=current_page,
            page_size=page_size,
            risk_level=risk_level,
            status=status,
            risk_name=risk_name,
            app_name=app_name,
            asset_type=asset_type,
            vendor=vendor,
            order_by=order_by,
            order=order,
            lang=lang,
            status_list=status_list,
        ).to_dict()
        resp = self._client.transport.request(
            "GET",
            _PATH_SECURITY_AGENT_LOGS,
            params=params,
        )
        return _alert_list_from_payload(resp.data)


class AsyncSecurity:
    """Async security center read interface."""

    def __init__(self, client) -> None:
        self._client = client

    async def overview(self) -> SecurityOverview:
        resp = await self._client.transport.request(
            "GET",
            _PATH_SECURITY_OVERVIEW,
        )
        return _overview_from_payload(resp.data)

    async def list_agent_logs(
        self,
        *,
        current_page: Optional[int] = None,
        page_size: Optional[int] = None,
        risk_level: Optional[str] = None,
        status: Optional[str] = None,
        risk_name: Optional[str] = None,
        app_name: Optional[str] = None,
        asset_type: Optional[str] = None,
        vendor: Optional[str] = None,
        order_by: Optional[str] = None,
        order: Optional[str] = None,
        lang: Optional[str] = None,
        status_list: Optional[Sequence[str]] = None,
    ) -> SecurityAlertList:
        params = SecurityListAgentLogsParams(
            current_page=current_page,
            page_size=page_size,
            risk_level=risk_level,
            status=status,
            risk_name=risk_name,
            app_name=app_name,
            asset_type=asset_type,
            vendor=vendor,
            order_by=order_by,
            order=order,
            lang=lang,
            status_list=status_list,
        ).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _PATH_SECURITY_AGENT_LOGS,
            params=params,
        )
        return _alert_list_from_payload(resp.data)
