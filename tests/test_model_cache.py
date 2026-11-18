# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Region-aware model catalog cache (A4 Part 2).

Covers the key helpers and ``_CatalogCache`` directly, then the wiring
through ``Models`` / ``AsyncModels``: hits, ``refresh``, key isolation,
``raw``/error bypass, the shared get/is_available read, pagination refresh,
non-aliasing and a thread-safety smoke.
"""

from __future__ import annotations

import threading

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.errors import DashScopeError
from dashscope.resources import _model_cache as mc
from dashscope.resources._model_cache import (
    _CatalogCache,
    catalog_key,
    credential_fingerprint,
    normalize_params,
)

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


def _page(models, total=None, page_no=1, page_size=10):
    return httpx.Response(
        200,
        json={
            "request_id": "r",
            "output": {
                "models": models,
                "total": len(models) if total is None else total,
                "page_no": page_no,
                "page_size": page_size,
            },
        },
    )


class _Cfg:
    """Stand-in for the frozen client Config in key-helper unit tests."""

    def __init__(self, region="cn-beijing", workspace="ws-1", api_key="sk-1"):
        self.region = region
        self.workspace = workspace
        self.api_key = api_key


class TestKeyHelpers:
    def test_normalize_params_is_order_independent_and_hashable(self):
        a = normalize_params({"b": 2, "a": ["x", "y"]})
        b = normalize_params({"a": ("x", "y"), "b": 2})
        assert a == b == (("a", ("x", "y")), ("b", 2))
        assert hash(a) == hash(b)

    def test_credential_fingerprint_hides_the_key(self):
        fp = credential_fingerprint("sk-secret")
        assert "sk-secret" not in fp
        assert len(fp) == 16
        assert fp == credential_fingerprint("sk-secret")
        assert fp != credential_fingerprint("sk-other")
        assert credential_fingerprint(None) == credential_fingerprint("")

    def test_catalog_key_isolates_every_dimension(self):
        cfg = _Cfg()
        params = {"model": "qwen-plus", "page_no": 1, "page_size": 1}
        base = catalog_key(cfg, params, api_key=None, workspace=None)
        assert (
            catalog_key(
                _Cfg(region="ap-southeast-1"),
                params,
                api_key=None,
                workspace=None,
            )
            != base
        )
        assert (
            catalog_key(cfg, params, api_key=None, workspace="ws-2") != base
        )
        assert (
            catalog_key(cfg, params, api_key="sk-2", workspace=None) != base
        )
        assert (
            catalog_key(
                cfg,
                {**params, "page_no": 2},
                api_key=None,
                workspace=None,
            )
            != base
        )

    def test_catalog_key_resolves_overrides_against_config(self):
        cfg = _Cfg()
        params = {"page_no": 1, "page_size": 10}
        # Per-call values equal to the config resolve to the same key.
        assert (
            catalog_key(cfg, params, api_key="sk-1", workspace="ws-1")
            == catalog_key(cfg, params, api_key=None, workspace=None)
        )


class TestCatalogCache:
    def test_hit_then_ttl_expiry(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
        cache = _CatalogCache(ttl=60.0)
        sentinel = object()
        cache.set("k", sentinel)
        assert cache.get("k") is sentinel
        now[0] += 59.0
        assert cache.get("k") is sentinel
        now[0] += 2.0
        assert cache.get("k") is None
        assert "k" not in cache._entries

    def test_miss_on_unknown_key(self):
        assert _CatalogCache().get("absent") is None

    def test_eviction_drops_the_oldest_at_the_cap(self, monkeypatch):
        now = [0.0]
        monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
        cache = _CatalogCache(ttl=1000.0)
        for i in range(mc._MAX_ENTRIES + 1):
            now[0] = float(i)
            cache.set(("k", i), object())
        assert len(cache._entries) == mc._MAX_ENTRIES
        assert cache.get(("k", 0)) is None
        assert cache.get(("k", mc._MAX_ENTRIES)) is not None

    def test_resetting_a_key_does_not_evict(self, monkeypatch):
        now = [0.0]
        monkeypatch.setattr(mc.time, "monotonic", lambda: now[0])
        cache = _CatalogCache(ttl=1000.0)
        for i in range(mc._MAX_ENTRIES):
            now[0] = float(i)
            cache.set(("k", i), object())
        now[0] = 9999.0
        cache.set(("k", 0), object())  # refresh existing, not a new entry
        assert len(cache._entries) == mc._MAX_ENTRIES
        assert cache.get(("k", 0)) is not None


class TestSyncCache:
    def test_same_params_hits_cache(self):
        calls = []

        def handler(request):
            calls.append(dict(request.url.params))
            return _page([{"model": "qwen-plus"}])

        client = _make_client(handler)
        a = client.models.list()
        b = client.models.list()
        assert len(calls) == 1
        assert [m.model for m in a.models] == ["qwen-plus"]
        assert [m.model for m in b.models] == ["qwen-plus"]
        assert a is not b

    def test_refresh_bypasses_and_updates_cache(self):
        calls = []

        def handler(request):
            calls.append(1)
            return _page([{"model": f"m{len(calls)}"}])

        client = _make_client(handler)
        assert client.models.list().models[0].model == "m1"
        assert client.models.list().models[0].model == "m1"  # hit
        assert len(calls) == 1
        assert client.models.list(refresh=True).models[0].model == "m2"
        assert len(calls) == 2
        assert client.models.list().models[0].model == "m2"  # updated hit
        assert len(calls) == 2

    def test_key_isolation_per_call_overrides(self):
        calls = []

        def handler(request):
            calls.append(dict(request.url.params))
            return _page([{"model": "qwen-plus"}])

        client = _make_client(handler)
        client.models.list()  # 1: base
        client.models.list()  # hit
        client.models.list(workspace="ws-2")  # 2
        client.models.list(api_key="sk-other")  # 3
        client.models.list(provider="qwen")  # 4
        client.models.list(page_no=2)  # 5
        client.models.list(service_site="intl")  # 6
        assert len(calls) == 6

    def test_raw_bypasses_cache(self):
        calls = []

        def handler(request):
            calls.append(1)
            return _page([{"model": "qwen-plus"}])

        client = _make_client(handler)
        r1 = client.models.list(raw=True)
        r2 = client.models.list(raw=True)
        assert r1.status_code == 200
        assert len(calls) == 2  # raw never caches
        client.models.list()  # non-raw still has to fetch
        assert len(calls) == 3
        client.models.list()  # now cached
        assert len(calls) == 3

    def test_error_is_not_cached(self):
        calls = []

        def handler(request):
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(500, json={"message": "boom"})
            return _page([{"model": "qwen-plus"}])

        client = _make_client(handler)
        with pytest.raises(DashScopeError):
            client.models.list(max_retries=0)
        assert len(calls) == 1
        # The failure was not cached: the retry reaches the network again.
        assert client.models.list(max_retries=0).models[0].model == "qwen-plus"
        assert len(calls) == 2
        client.models.list(max_retries=0)  # success now cached
        assert len(calls) == 2

    def test_business_error_in_non_raising_mode_is_not_cached(self):
        calls = []

        def handler(request):
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(
                    200,
                    json={"code": "InternalError", "message": "boom"},
                )
            return _page([{"model": "qwen-plus"}])

        client = _make_client(handler, raise_on_error=False)
        first = client.models.list(max_retries=0)
        assert first.code == "InternalError"
        second = client.models.list(max_retries=0)
        assert [m.model for m in second.models] == ["qwen-plus"]
        assert len(calls) == 2

    def test_get_and_is_available_share_the_read(self):
        calls = []

        def handler(request):
            calls.append(dict(request.url.params))
            return _page([{"model": "qwen-plus"}], total=1)

        client = _make_client(handler)
        model = client.models.get("qwen-plus")
        assert model.model == "qwen-plus"
        assert client.models.is_available("qwen-plus") is True
        assert len(calls) == 1  # same query, one fetch
        client.models.get("qwen-plus")
        assert len(calls) == 1
        # A different model is a different key.
        assert client.models.is_available("qwen-max") is True
        assert len(calls) == 2

    def test_cache_hit_returns_fresh_objects(self):
        def handler(request):
            return _page([{"model": "qwen-plus"}])

        client = _make_client(handler)
        a = client.models.list()
        b = client.models.list()
        assert a is not b
        assert a.models is not b.models
        assert a.models[0] is not b.models[0]
        assert b.models[0].model == "qwen-plus"

    def test_iter_refresh_refetches_every_page(self):
        catalog = [{"model": "m1"}, {"model": "m2"}]
        calls = []

        def handler(request):
            page_no = int(request.url.params.get("page_no", 1))
            page_size = int(request.url.params.get("page_size", 10))
            calls.append(page_no)
            start = (page_no - 1) * page_size
            return _page(
                catalog[start : start + page_size],
                total=len(catalog),
                page_no=page_no,
                page_size=page_size,
            )

        client = _make_client(handler)
        assert [m.model for m in client.models.list_all(page_size=1)] == [
            "m1",
            "m2",
        ]
        assert calls == [1, 2]
        client.models.list_all(page_size=1)  # both pages cached
        assert calls == [1, 2]
        client.models.list_all(page_size=1, refresh=True)
        assert calls == [1, 2, 1, 2]

    def test_disabled_ttl_never_hits(self):
        calls = []

        def handler(request):
            calls.append(1)
            return _page([{"model": "qwen-plus"}])

        client = _make_client(handler)
        client.models._catalog_cache = _CatalogCache(ttl=0.0)
        client.models.list()
        client.models.list()
        assert len(calls) == 2

    def test_concurrent_reads_of_a_warm_key_do_not_refetch(self):
        calls = []

        def handler(request):
            calls.append(1)
            return _page([{"model": "qwen-plus"}])

        client = _make_client(handler)
        client.models.list()  # warm the cache
        assert len(calls) == 1
        errors = []

        def worker():
            try:
                for _ in range(20):
                    result = client.models.list()
                    assert result.models[0].model == "qwen-plus"
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(calls) == 1  # every concurrent read was a hit


class TestAsyncCache:
    @pytest.mark.asyncio
    async def test_same_params_hits_cache(self):
        calls = []

        async def handler(request):
            calls.append(1)
            return _page([{"model": "qwen-plus"}])

        client = _make_async_client(handler)
        a = await client.models.list()
        b = await client.models.list()
        assert len(calls) == 1
        assert a is not b
        assert [m.model for m in b.models] == ["qwen-plus"]
        await client.aclose()

    @pytest.mark.asyncio
    async def test_refresh_bypasses_and_updates_cache(self):
        calls = []

        async def handler(request):
            calls.append(1)
            return _page([{"model": f"m{len(calls)}"}])

        client = _make_async_client(handler)
        assert (await client.models.list()).models[0].model == "m1"
        assert (await client.models.list()).models[0].model == "m1"
        assert len(calls) == 1
        fresh = await client.models.list(refresh=True)
        assert fresh.models[0].model == "m2"
        assert len(calls) == 2
        assert (await client.models.list()).models[0].model == "m2"
        assert len(calls) == 2
        await client.aclose()

    @pytest.mark.asyncio
    async def test_raw_bypasses_cache(self):
        calls = []

        async def handler(request):
            calls.append(1)
            return _page([{"model": "qwen-plus"}])

        client = _make_async_client(handler)
        await client.models.list(raw=True)
        await client.models.list(raw=True)
        assert len(calls) == 2
        await client.models.list()
        assert len(calls) == 3
        await client.aclose()

    @pytest.mark.asyncio
    async def test_error_is_not_cached(self):
        calls = []

        async def handler(request):
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(500, json={"message": "boom"})
            return _page([{"model": "qwen-plus"}])

        client = _make_async_client(handler)
        with pytest.raises(DashScopeError):
            await client.models.list(max_retries=0)
        result = await client.models.list(max_retries=0)
        assert result.models[0].model == "qwen-plus"
        assert len(calls) == 2
        await client.aclose()

    @pytest.mark.asyncio
    async def test_get_and_is_available_share_the_read(self):
        calls = []

        async def handler(request):
            calls.append(1)
            return _page([{"model": "qwen-plus"}], total=1)

        client = _make_async_client(handler)
        assert (await client.models.get("qwen-plus")).model == "qwen-plus"
        assert await client.models.is_available("qwen-plus") is True
        assert len(calls) == 1
        await client.aclose()

    @pytest.mark.asyncio
    async def test_iter_refresh_refetches_every_page(self):
        catalog = [{"model": "m1"}, {"model": "m2"}]
        calls = []

        async def handler(request):
            page_no = int(request.url.params.get("page_no", 1))
            page_size = int(request.url.params.get("page_size", 10))
            calls.append(page_no)
            start = (page_no - 1) * page_size
            return _page(
                catalog[start : start + page_size],
                total=len(catalog),
                page_no=page_no,
                page_size=page_size,
            )

        client = _make_async_client(handler)
        first = await client.models.list_all(page_size=1)
        assert [m.model for m in first] == ["m1", "m2"]
        assert calls == [1, 2]
        await client.models.list_all(page_size=1)
        assert calls == [1, 2]
        await client.models.list_all(page_size=1, refresh=True)
        assert calls == [1, 2, 1, 2]
        await client.aclose()