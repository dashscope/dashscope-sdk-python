# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``client.billing`` -- cost overview and trend (read-only, G2).

Only the two read ops are implemented; there is no purchase / renewal /
payment surface here. ``group_by`` is a JSON-style array and ``filter`` /
``time_period`` are JSON-style objects on the wire.
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

from dashscope.management._parse import parse_one
from dashscope.management._resource import (
    AsyncResource,
    SyncResource,
    options_kwargs,
)
from dashscope.management.types import BillingOverview, BillingTrend
from dashscope.transport._options import request_options

if TYPE_CHECKING:
    from dashscope.transport._response import APIResponse

__all__ = ["AsyncBilling", "Billing"]


class Billing(SyncResource):
    @overload
    def overview(
        self,
        *,
        bill_month: Optional[str] = None,
        group_by: Optional[Any] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        top_num: Optional[int] = None,
        zero_filter: Optional[bool] = None,
        region_id: Optional[str] = None,
        locale: Optional[str] = None,
        granularity: Optional[str] = None,
        time_period: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> BillingOverview:
        ...

    @overload
    def overview(
        self,
        *,
        bill_month: Optional[str] = None,
        group_by: Optional[Any] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        top_num: Optional[int] = None,
        zero_filter: Optional[bool] = None,
        region_id: Optional[str] = None,
        locale: Optional[str] = None,
        granularity: Optional[str] = None,
        time_period: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def overview(
        self,
        *,
        bill_month: Optional[str] = None,
        group_by: Optional[Any] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        top_num: Optional[int] = None,
        zero_filter: Optional[bool] = None,
        region_id: Optional[str] = None,
        locale: Optional[str] = None,
        granularity: Optional[str] = None,
        time_period: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[BillingOverview, "APIResponse"]:
        """Cost overview for a billing month, optionally grouped TopN."""
        filters = {
            "bill_month": bill_month,
            "group_by": group_by,
            "filter_": filter_,
            "top_num": top_num,
            "zero_filter": zero_filter,
            "region_id": region_id,
            "locale": locale,
            "granularity": granularity,
            "time_period": time_period,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = self._request(
            "GetBillingOverview",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp
        return parse_one(
            BillingOverview,
            resp,
            data_field="data",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
        )

    @overload
    def trend(
        self,
        *,
        granularity: Optional[str] = None,
        time_period: Optional[Mapping[str, Any]] = None,
        group_by: Optional[Any] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        top_num: Optional[int] = None,
        zero_filter: Optional[bool] = None,
        include_history_period: Optional[bool] = None,
        region_id: Optional[str] = None,
        locale: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> BillingTrend:
        ...

    @overload
    def trend(
        self,
        *,
        granularity: Optional[str] = None,
        time_period: Optional[Mapping[str, Any]] = None,
        group_by: Optional[Any] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        top_num: Optional[int] = None,
        zero_filter: Optional[bool] = None,
        include_history_period: Optional[bool] = None,
        region_id: Optional[str] = None,
        locale: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def trend(
        self,
        *,
        granularity: Optional[str] = None,
        time_period: Optional[Mapping[str, Any]] = None,
        group_by: Optional[Any] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        top_num: Optional[int] = None,
        zero_filter: Optional[bool] = None,
        include_history_period: Optional[bool] = None,
        region_id: Optional[str] = None,
        locale: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[BillingTrend, "APIResponse"]:
        """Cost trend over time, optionally grouped and with history."""
        filters = {
            "granularity": granularity,
            "time_period": time_period,
            "group_by": group_by,
            "filter_": filter_,
            "top_num": top_num,
            "zero_filter": zero_filter,
            "include_history_period": include_history_period,
            "region_id": region_id,
            "locale": locale,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = self._request(
            "GetBillingTrend",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp
        return parse_one(
            BillingTrend,
            resp,
            data_field="data",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
        )


class AsyncBilling(AsyncResource):
    @overload
    async def overview(
        self,
        *,
        bill_month: Optional[str] = None,
        group_by: Optional[Any] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        top_num: Optional[int] = None,
        zero_filter: Optional[bool] = None,
        region_id: Optional[str] = None,
        locale: Optional[str] = None,
        granularity: Optional[str] = None,
        time_period: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> BillingOverview:
        ...

    @overload
    async def overview(
        self,
        *,
        bill_month: Optional[str] = None,
        group_by: Optional[Any] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        top_num: Optional[int] = None,
        zero_filter: Optional[bool] = None,
        region_id: Optional[str] = None,
        locale: Optional[str] = None,
        granularity: Optional[str] = None,
        time_period: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def overview(
        self,
        *,
        bill_month: Optional[str] = None,
        group_by: Optional[Any] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        top_num: Optional[int] = None,
        zero_filter: Optional[bool] = None,
        region_id: Optional[str] = None,
        locale: Optional[str] = None,
        granularity: Optional[str] = None,
        time_period: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[BillingOverview, "APIResponse"]:
        filters = {
            "bill_month": bill_month,
            "group_by": group_by,
            "filter_": filter_,
            "top_num": top_num,
            "zero_filter": zero_filter,
            "region_id": region_id,
            "locale": locale,
            "granularity": granularity,
            "time_period": time_period,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = await self._request(
            "GetBillingOverview",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp
        return parse_one(
            BillingOverview,
            resp,
            data_field="data",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
        )

    @overload
    async def trend(
        self,
        *,
        granularity: Optional[str] = None,
        time_period: Optional[Mapping[str, Any]] = None,
        group_by: Optional[Any] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        top_num: Optional[int] = None,
        zero_filter: Optional[bool] = None,
        include_history_period: Optional[bool] = None,
        region_id: Optional[str] = None,
        locale: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> BillingTrend:
        ...

    @overload
    async def trend(
        self,
        *,
        granularity: Optional[str] = None,
        time_period: Optional[Mapping[str, Any]] = None,
        group_by: Optional[Any] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        top_num: Optional[int] = None,
        zero_filter: Optional[bool] = None,
        include_history_period: Optional[bool] = None,
        region_id: Optional[str] = None,
        locale: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def trend(
        self,
        *,
        granularity: Optional[str] = None,
        time_period: Optional[Mapping[str, Any]] = None,
        group_by: Optional[Any] = None,
        filter_: Optional[Mapping[str, Any]] = None,
        top_num: Optional[int] = None,
        zero_filter: Optional[bool] = None,
        include_history_period: Optional[bool] = None,
        region_id: Optional[str] = None,
        locale: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[BillingTrend, "APIResponse"]:
        filters = {
            "granularity": granularity,
            "time_period": time_period,
            "group_by": group_by,
            "filter_": filter_,
            "top_num": top_num,
            "zero_filter": zero_filter,
            "include_history_period": include_history_period,
            "region_id": region_id,
            "locale": locale,
        }
        opts = options_kwargs(
            timeout,
            extra_headers,
            extra_query,
            extra_body,
            max_retries,
        )
        resp = await self._request(
            "GetBillingTrend",
            filters,
            request_options(**opts),
        )
        if raw:
            return resp
        return parse_one(
            BillingTrend,
            resp,
            data_field="data",
            strict=self._strict,
            raise_on_error=self._raise_on_error,
        )
