# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Usage summary and trend queries for Managed Agents."""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["AsyncUsage", "Usage"]

_PATH_USAGE_SUMMARY = "/usage/summary"
_PATH_USAGE_TREND = "/usage/trend"


class Usage:
    """Read workspace, agent, and session usage."""

    def __init__(self, client) -> None:
        self._client = client

    def summary(self, **params: Any) -> Dict[str, Any]:
        """Query usage totals and model/tool breakdowns (``POST /usage/summary``).

        Pass the API's snake_case body fields as keyword arguments. The
        service validates dimension-specific parameter combinations.
        """
        resp = self._client.transport.request(
            "POST",
            _PATH_USAGE_SUMMARY,
            json=dict(params),
        )
        return resp.data

    def trend(self, **params: Any) -> Dict[str, Any]:
        """Query paginated usage buckets (``POST /usage/trend``).

        Pass the API's snake_case body fields as keyword arguments. To fetch
        another page, pass the returned ``next_page_token`` as ``page_token``.
        """
        resp = self._client.transport.request(
            "POST",
            _PATH_USAGE_TREND,
            json=dict(params),
        )
        return resp.data


class AsyncUsage:
    """Async usage summary and trend queries."""

    def __init__(self, client) -> None:
        self._client = client

    async def summary(self, **params: Any) -> Dict[str, Any]:
        resp = await self._client.transport.request(
            "POST",
            _PATH_USAGE_SUMMARY,
            json=dict(params),
        )
        return resp.data

    async def trend(self, **params: Any) -> Dict[str, Any]:
        resp = await self._client.transport.request(
            "POST",
            _PATH_USAGE_TREND,
            json=dict(params),
        )
        return resp.data
