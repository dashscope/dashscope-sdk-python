# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``client.billing`` read tests (overview + trend, sync + async)."""

from __future__ import annotations

import httpx
import pytest

# pylint: disable=wrong-import-position
pytest.importorskip("alibabacloud_credentials")
pytest.importorskip("alibabacloud_openapi_util")

from dashscope.management.types import (  # noqa: E402
    BillingOverview,
    BillingTrend,
)
from dashscope.transport._response import APIResponse  # noqa: E402

from tests.management_helpers import (  # noqa: E402
    action_handler,
    make_async_client,
    make_client,
    ms_body,
    query_of,
)

BODIES = {
    "GetBillingOverview": ms_body(
        data={
            "currency": "CNY",
            "totalAmount": "12.50",
            "groups": [{"key": "g1", "name": "model", "amount": "12.50"}],
        },
    ),
    "GetBillingTrend": ms_body(
        data={
            "costTotals": {"amount": "30.00", "currency": "CNY"},
            "groupByTotal": [{"key": "g1", "amount": "30.00"}],
            "resultByTime": [{"period": "2026-02", "total": "30.00"}],
        },
    ),
}


def test_overview_typed():
    client = make_client(action_handler(BODIES))
    overview = client.billing.overview(bill_month="2026-02")
    assert isinstance(overview, BillingOverview)
    assert overview.total_amount == "12.50"
    assert overview.currency == "CNY"
    assert overview.groups[0].key == "g1"
    assert overview.is_error is False
    client.close()


def test_trend_typed():
    client = make_client(action_handler(BODIES))
    trend = client.billing.trend(granularity="MONTHLY")
    assert isinstance(trend, BillingTrend)
    assert trend.cost_totals.amount == "30.00"
    assert trend.group_by_total[0].amount == "30.00"
    assert trend.result_by_time[0].period == "2026-02"
    client.close()


def test_overview_bool_and_json_query():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=BODIES["GetBillingOverview"])

    client = make_client(handler)
    client.billing.overview(
        zero_filter=True,
        group_by=["model"],
        filter_={"workspaceId": "ws-1"},
        time_period={"start": "2026-02-01", "end": "2026-02-28"},
    )
    q = query_of(seen[0])
    assert q["zeroFilter"] == "true"
    assert q["groupBy"] == '["model"]'
    assert q["filter"] == '{"workspaceId":"ws-1"}'
    assert q["timePeriod"] == '{"start":"2026-02-01","end":"2026-02-28"}'
    assert seen[0].url.path == "/modelstudio/billing/overview"
    client.close()


def test_trend_include_history_bool_false():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=BODIES["GetBillingTrend"])

    client = make_client(handler)
    client.billing.trend(include_history_period=False, zero_filter=False)
    q = query_of(seen[0])
    assert q["includeHistoryPeriod"] == "false"
    assert q["zeroFilter"] == "false"
    client.close()


def test_overview_raw_overload():
    client = make_client(action_handler(BODIES))
    resp = client.billing.overview(raw=True)
    assert isinstance(resp, APIResponse)
    client.close()


async def test_async_overview_and_trend():
    client = make_async_client(action_handler(BODIES))
    overview = await client.billing.overview()
    assert overview.total_amount == "12.50"
    trend = await client.billing.trend()
    assert trend.cost_totals.amount == "30.00"
    await client.aclose()
