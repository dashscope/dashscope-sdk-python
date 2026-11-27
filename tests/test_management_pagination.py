# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Dual-pagination traversal tests: TokenPage (nextToken) + BasePage (PageNo)."""

from __future__ import annotations

import httpx
import pytest

# pylint: disable=wrong-import-position
pytest.importorskip("alibabacloud_credentials")
pytest.importorskip("alibabacloud_openapi_util")

from tests.management_helpers import (  # noqa: E402
    make_async_client,
    make_client,
    ms_body,
    query_of,
    tp_body,
)


def _token_handler(seen):
    """Three single-item pages chained by nextToken t2 -> t3 -> None."""
    pages = {
        None: ("ws-1", "t2"),
        "t2": ("ws-2", "t3"),
        "t3": ("ws-3", None),
    }

    def handler(request):
        q = query_of(request)
        seen.append(q)
        ws, nxt = pages[q.get("nextToken")]
        return httpx.Response(
            200,
            json=ms_body(
                workspaces=[{"workspaceId": ws}],
                nextToken=nxt,
                maxResults=1,
                totalCount=3,
            ),
        )

    return handler


def test_token_page_iter_current_only():
    seen = []
    client = make_client(_token_handler(seen))
    page = client.workspaces.list(max_results=1)
    assert [w.workspace_id for w in page] == ["ws-1"]
    assert len(seen) == 1  # iterating does not fetch ahead
    client.close()


def test_token_page_iter_pages_crosses_all():
    seen = []
    client = make_client(_token_handler(seen))
    page = client.workspaces.list(max_results=1)
    all_ids = [w.workspace_id for pg in page.iter_pages() for w in pg]
    assert all_ids == ["ws-1", "ws-2", "ws-3"]
    assert len(seen) == 3
    client.close()


def test_token_page_next_page_and_has_next():
    seen = []
    client = make_client(_token_handler(seen))
    page = client.workspaces.list(max_results=1)
    assert page.has_next_page() is True
    second = page.next_page()
    assert [w.workspace_id for w in second] == ["ws-2"]
    third = second.next_page()
    assert [w.workspace_id for w in third] == ["ws-3"]
    assert third.has_next_page() is False
    client.close()


def test_token_page_forwards_filters_across_pages():
    seen = []
    client = make_client(_token_handler(seen))
    page = client.workspaces.list(max_results=1, workspace_name="demo")
    list(page.iter_pages())
    assert all(q.get("workspaceName") == "demo" for q in seen)
    client.close()


def _page_handler(seen):
    """Three single-item member pages by PageNum, Total=3."""

    def handler(request):
        q = query_of(request)
        seen.append(q)
        no = int(q.get("PageNum", "1"))
        return httpx.Response(
            200,
            json=tp_body(
                Data=[{"AccountId": f"m{no}"}],
                PageNo=no,
                PageSize=1,
                Total=3,
            ),
        )

    return handler


def test_base_page_iter_pages_crosses_all():
    seen = []
    client = make_client(_page_handler(seen))
    page = client.token_plan.members.list(page_size=1)
    all_ids = [m.account_id for pg in page.iter_pages() for m in pg]
    assert all_ids == ["m1", "m2", "m3"]
    assert len(seen) == 3
    client.close()


def test_base_page_next_page_and_has_next():
    seen = []
    client = make_client(_page_handler(seen))
    page = client.token_plan.members.list(page_size=1)
    assert page.has_next_page() is True
    second = page.next_page()
    assert [m.account_id for m in second] == ["m2"]
    third = second.next_page()
    assert third.has_next_page() is False
    client.close()


def test_base_page_forwards_filters_across_pages():
    seen = []
    client = make_client(_page_handler(seen))
    page = client.token_plan.members.list(page_size=1, name="alice")
    list(page.iter_pages())
    assert all(q.get("Name") == "alice" for q in seen)
    client.close()


async def test_async_token_page_iter_pages():
    seen = []
    client = make_async_client(_token_handler(seen))
    page = await client.workspaces.list(max_results=1)
    all_ids = [
        w.workspace_id async for pg in page.iter_pages_async() for w in pg
    ]
    assert all_ids == ["ws-1", "ws-2", "ws-3"]
    await client.aclose()


async def test_async_base_page_iter_pages():
    seen = []
    client = make_async_client(_page_handler(seen))
    page = await client.token_plan.members.list(page_size=1)
    all_ids = [
        m.account_id async for pg in page.iter_pages_async() for m in pg
    ]
    assert all_ids == ["m1", "m2", "m3"]
    await client.aclose()
