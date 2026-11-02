# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the shared offset-pagination abstraction (T4)."""

from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.resources._jobs import JobList
from dashscope.resources.files import FileList
from dashscope.resources.models import ModelList
from dashscope.types.pagination import BasePage

BASE_URL = "https://api.test.local/api/v1"


def _make_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def _make_async_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return AsyncClient(api_key="sk-test", http_client=http_client, **kwargs)


def _slice_page(
    catalog: List[Dict[str, Any]],
    key: str,
    params: Dict[str, str],
    *,
    envelope: str,
) -> Dict[str, Any]:
    """Build one page body for ``catalog`` from the request's query params."""
    page_no = int(params.get("page_no", 1))
    page_size = int(params.get("page_size", 10))
    start = (page_no - 1) * page_size
    window = catalog[start : start + page_size]
    page = {
        key: window,
        "page_no": page_no,
        "page_size": page_size,
        "total": len(catalog),
    }
    if envelope == "output":
        return {"request_id": "req-page", "output": page}
    return {"request_id": "req-page", "data": page}


# --- models.list (sync) ---------------------------------------------------


def _models_handler(catalog, seen):
    def handler(request):
        params = dict(request.url.params)
        seen.append(int(params.get("page_no", 1)))
        return httpx.Response(
            200,
            json=_slice_page(catalog, "models", params, envelope="output"),
        )

    return handler


def test_models_iter_yields_current_page_only():
    catalog = [{"model": f"m{i}", "provider": "dashscope"} for i in range(5)]
    seen: List[int] = []
    client = _make_client(_models_handler(catalog, seen))

    page = client.models.list(page_no=1, page_size=2)
    assert [m.model for m in page] == ["m0", "m1"]
    # Iterating the page does not itself fetch more pages.
    assert seen == [1]


def test_models_iter_pages_crosses_all_pages():
    catalog = [{"model": f"m{i}"} for i in range(5)]
    seen: List[int] = []
    client = _make_client(_models_handler(catalog, seen))

    page = client.models.list(page_no=1, page_size=2)
    collected = [m.model for p in page.iter_pages() for m in p]

    assert collected == ["m0", "m1", "m2", "m3", "m4"]
    assert seen == [1, 2, 3]


def test_models_next_page_advances_then_ends():
    catalog = [{"model": f"m{i}"} for i in range(3)]
    client = _make_client(_models_handler(catalog, []))

    first = client.models.list(page_no=1, page_size=2)
    assert first.has_next_page() is True
    second = first.next_page()
    assert second is not None
    assert second.page_no == 2
    assert [m.model for m in second] == ["m2"]
    assert second.has_next_page() is False
    assert second.next_page() is None


def test_models_boundary_exact_multiple_of_page_size():
    catalog = [{"model": f"m{i}"} for i in range(4)]
    seen: List[int] = []
    client = _make_client(_models_handler(catalog, seen))

    page = client.models.list(page_no=1, page_size=2)
    pages = list(page.iter_pages())
    assert [p.page_no for p in pages] == [1, 2]
    assert seen == [1, 2]


def test_models_boundary_partial_last_page():
    catalog = [{"model": f"m{i}"} for i in range(5)]
    client = _make_client(_models_handler(catalog, []))

    page = client.models.list(page_no=1, page_size=2)
    pages = list(page.iter_pages())
    assert [len(p.models) for p in pages] == [2, 2, 1]


def test_models_boundary_empty_catalog():
    seen: List[int] = []
    client = _make_client(_models_handler([], seen))

    page = client.models.list(page_no=1, page_size=10)
    assert page.total_count == 0
    assert page.has_next_page() is False
    assert list(page.iter_pages()) == [page]
    assert seen == [1]


def test_models_provider_filter_applies_per_page_only():
    # total_count stays the pre-filter catalog total; the filter narrows only
    # the items on each fetched page.
    catalog = [
        {"model": "a", "provider": "dashscope"},
        {"model": "b", "provider": "other"},
        {"model": "c", "provider": "dashscope"},
        {"model": "d", "provider": "other"},
    ]
    client = _make_client(_models_handler(catalog, []))

    page = client.models.list(page_no=1, page_size=2, provider="dashscope")
    assert page.total_count == 4  # unfiltered
    assert [m.model for m in page] == ["a"]

    all_filtered = [m.model for p in page.iter_pages() for m in p]
    assert all_filtered == ["a", "c"]


def test_model_dump_excludes_callback():
    catalog = [{"model": "m0"}]
    client = _make_client(_models_handler(catalog, []))
    page = client.models.list(page_no=1, page_size=10)

    dumped = page.model_dump()
    assert "_fetch_next" not in dumped
    assert not any(k.startswith("_") for k in dumped)
    assert dumped["models"] == [
        {
            "request_id": None,
            "status_code": None,
            "code": None,
            "message": None,
            "is_error": False,
            "headers": {},
            "model": "m0",
            "source": None,
            "provider": None,
            "open_source": None,
        },
    ]


def test_base_page_without_callback_cannot_advance():
    # has_next_page() is pure arithmetic and stays true, but next_page() needs
    # an injected callback, so a standalone page cannot fetch.
    page = ModelList(page_no=1, page_size=2, total_count=5)
    assert page.has_next_page() is True
    assert page.next_page() is None


def test_list_models_share_base_page():
    for cls in (ModelList, FileList, JobList):
        assert issubclass(cls, BasePage)
    # The default items view is empty until a subclass maps its own list.
    assert BasePage().items == []


def test_base_page_accepts_the_wire_total_alias():
    # The alias lives on BasePage so every list model inherits it, and a page
    # parsed straight from a body traverses instead of reading a zero total.
    for cls in (ModelList, FileList, JobList):
        assert cls.model_validate({"total": 7}).total_count == 7


# --- models.list (async) --------------------------------------------------


@pytest.mark.asyncio
async def test_models_async_iter_pages():
    catalog = [{"model": f"m{i}"} for i in range(5)]
    seen: List[int] = []
    client = _make_async_client(_models_handler(catalog, seen))

    page = await client.models.list(page_no=1, page_size=2)
    collected = [m.model async for p in page.iter_pages_async() for m in p]

    assert collected == ["m0", "m1", "m2", "m3", "m4"]
    assert seen == [1, 2, 3]


@pytest.mark.asyncio
async def test_models_async_next_page_ends():
    catalog = [{"model": f"m{i}"} for i in range(3)]
    client = _make_async_client(_models_handler(catalog, []))

    first = await client.models.list(page_no=1, page_size=2)
    second = await first.next_page_async()
    assert second is not None and second.page_no == 2
    assert await second.next_page_async() is None


@pytest.mark.asyncio
async def test_models_async_iter_current_page():
    catalog = [{"model": f"m{i}"} for i in range(5)]
    client = _make_async_client(_models_handler(catalog, []))

    page = await client.models.list(page_no=1, page_size=2)
    assert [m.model async for m in page] == ["m0", "m1"]


# --- files.list (sync) ----------------------------------------------------


def test_files_iter_pages():
    catalog = [{"file_id": f"f{i}", "name": f"f{i}.jsonl"} for i in range(3)]

    def handler(request):
        params = dict(request.url.params)
        return httpx.Response(
            200,
            json=_slice_page(catalog, "files", params, envelope="data"),
        )

    client = _make_client(handler)
    page = client.files.list(page_no=1, page_size=2)
    assert isinstance(page, FileList)
    assert [f.file_id for p in page.iter_pages() for f in p] == [
        "f0",
        "f1",
        "f2",
    ]


# --- jobs.list (sync) -----------------------------------------------------


def test_jobs_list_total_alias_and_traversal():
    catalog = [
        {"task_id": f"t{i}", "task_status": "SUCCEEDED"} for i in range(5)
    ]
    seen: List[int] = []

    def handler(request):
        params = dict(request.url.params)
        seen.append(int(params.get("page_no", 1)))
        page_no = int(params.get("page_no", 1))
        page_size = int(params.get("page_size", 10))
        start = (page_no - 1) * page_size
        return httpx.Response(
            200,
            json={
                "request_id": "req-jobs",
                "tasks": catalog[start : start + page_size],
                "page_no": page_no,
                "page_size": page_size,
                "total": len(catalog),
            },
        )

    client = _make_client(handler)
    page = client.images.jobs.list(page_no=1, page_size=2)

    assert isinstance(page, JobList)
    # Wire ``total`` is surfaced as ``total_count`` and via the ``total`` prop.
    assert page.total_count == 5
    assert page.total == 5
    assert page.has_next_page() is True

    collected = [t.task_id for p in page.iter_pages() for t in p]
    assert collected == ["t0", "t1", "t2", "t3", "t4"]
    assert seen == [1, 2, 3]


def test_jobs_list_model_dump_uses_total_count():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-jobs",
                "tasks": [],
                "page_no": 1,
                "page_size": 10,
                "total": 0,
            },
        )

    client = _make_client(handler)
    page = client.images.jobs.list()
    dumped = page.model_dump()
    assert "total_count" in dumped
    assert "_fetch_next" not in dumped
