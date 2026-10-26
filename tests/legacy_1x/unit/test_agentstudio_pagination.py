# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Tests for cursor-based pagination."""

from __future__ import annotations

from typing import List

import pytest

from dashscope.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
    build_page,
)


def test_build_page():
    page = build_page(
        payload={"data": [{"id": "a"}, {"id": "b"}], "next_page": "token_2"},
        item_factory=lambda x: x,
        request_id="req_1",
    )
    assert len(page.data) == 2
    assert page.next_page == "token_2"
    assert page.request_id == "req_1"


def test_build_page_empty():
    page = build_page(payload={"data": None}, item_factory=lambda x: x)
    assert page.data == []


def test_cursor_page_has_next_and_get_next():
    pages: List[CursorPage[int]] = [
        CursorPage(data=[1, 2], next_page="p2", request_id=None),
        CursorPage(data=[3], next_page=None, request_id=None),
    ]
    call_idx = iter(range(1, 3))

    def _fetch(_token: str) -> CursorPage[int]:
        return pages[next(call_idx)]

    pages[0]._fetch_next = _fetch
    assert pages[0].has_next() is True
    nxt = pages[0].get_next()
    assert nxt is not None
    assert nxt.data == [3]
    assert nxt.has_next() is False


def test_cursor_page_iter_auto_paginates():
    pages: List[CursorPage[int]] = [
        CursorPage(data=[1, 2], next_page="p2", request_id=None),
        CursorPage(data=[3], next_page=None, request_id=None),
    ]
    call_idx = iter(range(1, 3))

    def _fetch(_token: str) -> CursorPage[int]:
        return pages[next(call_idx)]

    pages[0]._fetch_next = _fetch
    assert list(pages[0]) == [1, 2, 3]


def test_cursor_page_no_next():
    page = CursorPage(data=[1], next_page=None, request_id=None)
    assert page.has_next() is False
    assert page.get_next() is None
    assert list(page) == [1]


@pytest.mark.asyncio
async def test_async_cursor_page_iter():
    pages: List[AsyncCursorPage[int]] = [
        AsyncCursorPage(data=[1], next_page="p2", request_id=None),
        AsyncCursorPage(data=[2], next_page=None, request_id=None),
    ]
    call_idx = iter(range(1, 3))

    async def _fetch(_token: str) -> AsyncCursorPage[int]:
        return pages[next(call_idx)]

    pages[0]._fetch_next = _fetch
    result = [item async for item in pages[0]]
    assert result == [1, 2]


@pytest.mark.asyncio
async def test_async_cursor_page_no_next():
    page = AsyncCursorPage(data=[1], next_page=None, request_id=None)
    assert page.has_next() is False
    assert await page.get_next() is None
    result = [item async for item in page]
    assert result == [1]
