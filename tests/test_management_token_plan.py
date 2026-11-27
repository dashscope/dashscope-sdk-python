# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``client.token_plan`` read tests (G5, sync + async)."""

from __future__ import annotations

import httpx
import pytest

# pylint: disable=wrong-import-position
pytest.importorskip("alibabacloud_credentials")
pytest.importorskip("alibabacloud_openapi_util")

from dashscope.management.types import (  # noqa: E402
    OrgMember,
    OrgMemberPage,
    Seat,
    SeatPage,
    SeatStats,
    SharedPackage,
    SharedPackagePage,
    SubscriptionStats,
    TokenPlanAccount,
)
from dashscope.transport._response import APIResponse  # noqa: E402

from tests.management_helpers import (  # noqa: E402
    action_handler,
    make_async_client,
    make_client,
    query_of,
    tp_body,
)

BODIES = {
    "GetTokenPlanAccountDetail": tp_body(
        Data={
            "AccountId": "acc-1",
            "Name": "demo-org",
            "OrgMemberships": [{"OrgId": "org-1", "RoleCode": "owner"}],
        },
    ),
    "ListOrganizationMembers": tp_body(
        Data=[
            {
                "AccountId": "m1",
                "MaskedApiKey": "sk-***abc",
                "Status": "active",
            },
        ],
        PageNo=1,
        PageSize=20,
        Total=1,
    ),
    "GetSubscriptionSeatDetails": tp_body(
        Data={
            "Items": [{"SeatId": "s1", "SpecType": "basic"}],
            "PageNo": 1,
            "PageSize": 20,
            "Total": 1,
        },
    ),
    "GetOrganizationMemberSeatStats": {
        "OrgId": "org-1",
        "TotalMemberCount": 5,
        "SeatedMemberCount": 3,
    },
    "GetSubscriptionStats": tp_body(
        Data={
            "SubscriptionStartTime": 1700000000,
            "Items": [{"SeatType": "basic", "TotalSeats": 5}],
        },
    ),
    "ListSubscriptionSharedPackages": tp_body(
        Data={
            "Items": [{"InstanceCode": "p1", "Status": "active"}],
            "PageNo": 1,
            "PageSize": 20,
            "Total": 1,
        },
    ),
}


def test_account_get_typed():
    client = make_client(action_handler(BODIES))
    account = client.token_plan.account.get()
    assert isinstance(account, TokenPlanAccount)
    assert account.account_id == "acc-1"
    assert account.org_memberships[0].org_id == "org-1"
    assert account.is_error is False
    client.close()


def test_members_root_array_page():
    client = make_client(action_handler(BODIES))
    page = client.token_plan.members.list()
    assert isinstance(page, OrgMemberPage)
    assert isinstance(page.items[0], OrgMember)
    assert page.items[0].account_id == "m1"
    assert page.items[0].masked_api_key == "sk-***abc"
    assert page.page_no == 1
    assert page.total_count == 1
    client.close()


def test_members_wire_pascal_and_bool():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=BODIES["ListOrganizationMembers"])

    client = make_client(handler)
    client.token_plan.members.list(name="alice", has_seat=True, page_num=2)
    q = query_of(seen[0])
    assert q["Name"] == "alice"
    assert q["HasSeat"] == "true"
    assert q["PageNum"] == "2"
    assert seen[0].url.path == "/tokenplan/organization/members"
    client.close()


def test_seats_nested_page():
    client = make_client(action_handler(BODIES))
    page = client.token_plan.seats.list()
    assert isinstance(page, SeatPage)
    assert isinstance(page.items[0], Seat)
    assert page.items[0].seat_id == "s1"
    assert page.total_count == 1
    client.close()


def test_seats_flat_status_list_query():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=BODIES["GetSubscriptionSeatDetails"])

    client = make_client(handler)
    client.token_plan.seats.list(
        status_list=["active", "expired"],
        query_assigned=True,
    )
    q = query_of(seen[0])
    assert q["StatusList.1"] == "active"
    assert q["StatusList.2"] == "expired"
    assert q["QueryAssigned"] == "true"
    client.close()


def test_seat_stats_bare_root():
    client = make_client(action_handler(BODIES))
    stats = client.token_plan.seats.stats()
    assert isinstance(stats, SeatStats)
    assert stats.org_id == "org-1"
    assert stats.total_member_count == 5
    assert stats.seated_member_count == 3
    assert stats.is_error is False
    client.close()


def test_subscription_stats_typed():
    client = make_client(action_handler(BODIES))
    stats = client.token_plan.subscription.stats()
    assert isinstance(stats, SubscriptionStats)
    assert stats.subscription_start_time == 1700000000
    assert stats.items[0].seat_type == "basic"
    assert stats.items[0].total_seats == 5
    client.close()


def test_shared_packages_nested_page():
    client = make_client(action_handler(BODIES))
    page = client.token_plan.shared_packages.list()
    assert isinstance(page, SharedPackagePage)
    assert isinstance(page.items[0], SharedPackage)
    assert page.items[0].instance_code == "p1"
    client.close()


def test_account_raw_overload():
    client = make_client(action_handler(BODIES))
    resp = client.token_plan.account.get(raw=True)
    assert isinstance(resp, APIResponse)
    client.close()


async def test_async_token_plan_full_surface():
    client = make_async_client(action_handler(BODIES))
    account = await client.token_plan.account.get()
    assert account.account_id == "acc-1"
    members = await client.token_plan.members.list()
    assert members.items[0].account_id == "m1"
    seats = await client.token_plan.seats.list()
    assert seats.items[0].seat_id == "s1"
    stats = await client.token_plan.seats.stats()
    assert stats.total_member_count == 5
    sub = await client.token_plan.subscription.stats()
    assert sub.items[0].total_seats == 5
    packages = await client.token_plan.shared_packages.list()
    assert packages.items[0].instance_code == "p1"
    await client.aclose()
