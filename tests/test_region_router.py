# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Explicit cross-Region model discovery router (A5).

The router is disabled by default and requires an explicit
``allow_cross_region=True`` opt-in that warns cross-border data compliance
is the user's sole responsibility.
"""

from __future__ import annotations

import httpx
import pytest

from dashscope import (
    AsyncClient,
    AsyncRegionRouter,
    Client,
    ModelAvailability,
    RegionModelInfo,
    RegionRouter,
)

BASE_URL = "https://api.test.local/api/v1"


def _router(clients):
    with pytest.warns(UserWarning, match="sole responsibility"):
        return RegionRouter(clients=clients, allow_cross_region=True)


def _async_router(clients):
    with pytest.warns(UserWarning, match="sole responsibility"):
        return AsyncRegionRouter(clients=clients, allow_cross_region=True)


def _catalog_handler(catalog, calls):
    def handler(request):
        model = request.url.params.get("model")
        calls.append(model)
        models = (
            [m for m in catalog if m["model"] == model]
            if model
            else catalog
        )
        return httpx.Response(
            200,
            json={
                "request_id": "r",
                "output": {
                    "models": models,
                    "total": len(models),
                    "page_no": 1,
                    "page_size": 1,
                },
            },
        )

    return handler


def _region_client(region, catalog, calls, **kwargs):
    handler = _catalog_handler(catalog, calls)
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    return Client(
        api_key="sk-test",
        region=region,
        workspace=f"ws-{region}",
        base_url=BASE_URL,
        http_client=http_client,
        **kwargs,
    )


def _async_region_client(region, catalog, calls, **kwargs):
    handler = _catalog_handler(catalog, calls)

    async def async_handler(request):
        return handler(request)

    mock = httpx.MockTransport(async_handler)
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    return AsyncClient(
        api_key="sk-test",
        region=region,
        workspace=f"ws-{region}",
        base_url=BASE_URL,
        http_client=http_client,
        **kwargs,
    )


BJ = [{"model": "qwen-plus"}, {"model": "qwen-max"}]
SG = [{"model": "qwen-max"}]


class TestOptIn:
    def test_disabled_by_default(self):
        client = _region_client("cn-beijing", BJ, [])
        with pytest.raises(ValueError, match="disabled by default"):
            RegionRouter(clients={"cn-beijing": client})

    def test_async_disabled_by_default(self):
        client = _async_region_client("cn-beijing", BJ, [])
        with pytest.raises(ValueError, match="disabled by default"):
            AsyncRegionRouter(clients={"cn-beijing": client})

    def test_default_off_message_states_compliance(self):
        with pytest.raises(ValueError, match="sole responsibility"):
            RegionRouter(clients={})

    def test_enabling_warns_compliance(self):
        client = _region_client("cn-beijing", BJ, [])
        with pytest.warns(UserWarning, match="sole responsibility"):
            RegionRouter(
                clients={"cn-beijing": client},
                allow_cross_region=True,
            )

    def test_async_enabling_warns_compliance(self):
        client = _async_region_client("cn-beijing", BJ, [])
        with pytest.warns(UserWarning, match="sole responsibility"):
            AsyncRegionRouter(
                clients={"cn-beijing": client},
                allow_cross_region=True,
            )


class TestValidation:
    def test_empty_clients_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            _router({})

    def test_unsafe_region_label_rejected(self):
        client = _region_client("cn-beijing", BJ, [])
        with pytest.raises(ValueError, match="invalid region label"):
            _router({"cn-beijing.evil": client})

    def test_wrong_client_type_rejected(self):
        with pytest.raises(TypeError, match="must be a Client"):
            _router({"cn-beijing": object()})

    def test_key_must_match_client_region(self):
        client = _region_client("cn-beijing", BJ, [])
        with pytest.raises(ValueError, match="must match the client's"):
            _router({"ap-southeast-1": client})

    def test_async_router_requires_async_clients(self):
        sync = _region_client("cn-beijing", BJ, [])
        with pytest.raises(TypeError, match="must be a AsyncClient"):
            _async_router({"cn-beijing": sync})

    def test_valid_construction(self):
        router = _router(
            {
                "cn-beijing": _region_client("cn-beijing", BJ, []),
                "ap-southeast-1": _region_client("ap-southeast-1", SG, []),
            },
        )
        assert callable(router.models.available_regions)


class TestSyncDiscovery:
    def test_available_in_only_matching_regions(self):
        bj_calls, sg_calls = [], []
        router = _router(
            {
                "cn-beijing": _region_client("cn-beijing", BJ, bj_calls),
                "ap-southeast-1": _region_client(
                    "ap-southeast-1",
                    SG,
                    sg_calls,
                ),
            },
        )
        result = router.models.available_regions("qwen-plus")
        assert isinstance(result, ModelAvailability)
        assert result.model == "qwen-plus"
        assert result.available_in == ["cn-beijing"]
        bj = result.regions["cn-beijing"]
        assert isinstance(bj, RegionModelInfo)
        assert bj.available is True
        assert bj.model.model == "qwen-plus"
        assert bj.error is None
        sg = result.regions["ap-southeast-1"]
        assert sg.available is False
        assert sg.model is None
        assert sg.error is None
        # is_available + get share one cached read per available region.
        assert bj_calls == ["qwen-plus"]
        assert sg_calls == ["qwen-plus"]

    def test_model_in_every_region(self):
        router = _router(
            {
                "cn-beijing": _region_client("cn-beijing", BJ, []),
                "ap-southeast-1": _region_client("ap-southeast-1", SG, []),
            },
        )
        result = router.models.available_regions("qwen-max")
        assert result.available_in == ["cn-beijing", "ap-southeast-1"]

    def test_model_in_no_region(self):
        router = _router(
            {
                "cn-beijing": _region_client("cn-beijing", BJ, []),
                "ap-southeast-1": _region_client("ap-southeast-1", SG, []),
            },
        )
        result = router.models.available_regions("absent-model")
        assert result.available_in == []
        assert all(
            info.error is None for info in result.regions.values()
        )

    def test_blank_model_rejected(self):
        router = _router(
            {"cn-beijing": _region_client("cn-beijing", BJ, [])},
        )
        with pytest.raises(ValueError, match="non-empty model id"):
            router.models.available_regions("  ")

    def test_region_error_is_reported_not_fallback(self):
        def failing(request):
            return httpx.Response(500, json={"message": "boom"})

        mock = httpx.MockTransport(failing)
        bad = Client(
            api_key="sk-test",
            region="cn-beijing",
            workspace="ws-bj",
            base_url=BASE_URL,
            http_client=httpx.Client(transport=mock, base_url=BASE_URL),
            max_retries=0,
        )
        sg_calls = []
        router = _router(
            {
                "cn-beijing": bad,
                "ap-southeast-1": _region_client(
                    "ap-southeast-1",
                    SG,
                    sg_calls,
                ),
            },
        )
        result = router.models.available_regions("qwen-max")
        # The failing region is an error entry, the other is still probed.
        assert result.regions["cn-beijing"].available is False
        assert result.regions["cn-beijing"].error is not None
        assert result.regions["ap-southeast-1"].available is True
        assert result.available_in == ["ap-southeast-1"]
        assert sg_calls == ["qwen-max"]

    def test_refresh_refetches(self):
        calls = []
        router = _router(
            {"cn-beijing": _region_client("cn-beijing", BJ, calls)},
        )
        router.models.available_regions("qwen-plus")
        assert calls == ["qwen-plus"]
        router.models.available_regions("qwen-plus")  # cached
        assert calls == ["qwen-plus"]
        router.models.available_regions("qwen-plus", refresh=True)
        assert calls == ["qwen-plus", "qwen-plus"]


class TestAsyncDiscovery:
    @pytest.mark.asyncio
    async def test_available_in_only_matching_regions(self):
        bj_calls, sg_calls = [], []
        bj = _async_region_client("cn-beijing", BJ, bj_calls)
        sg = _async_region_client("ap-southeast-1", SG, sg_calls)
        router = _async_router({"cn-beijing": bj, "ap-southeast-1": sg})
        result = await router.models.available_regions("qwen-plus")
        assert result.available_in == ["cn-beijing"]
        assert result.regions["cn-beijing"].model.model == "qwen-plus"
        assert result.regions["ap-southeast-1"].available is False
        assert bj_calls == ["qwen-plus"]
        await bj.aclose()
        await sg.aclose()

    @pytest.mark.asyncio
    async def test_blank_model_rejected(self):
        bj = _async_region_client("cn-beijing", BJ, [])
        router = _async_router({"cn-beijing": bj})
        with pytest.raises(ValueError, match="non-empty model id"):
            await router.models.available_regions("")
        await bj.aclose()

    @pytest.mark.asyncio
    async def test_region_error_is_reported_not_fallback(self):
        async def failing(request):
            return httpx.Response(500, json={"message": "boom"})

        mock = httpx.MockTransport(failing)
        bad = AsyncClient(
            api_key="sk-test",
            region="cn-beijing",
            workspace="ws-bj",
            base_url=BASE_URL,
            http_client=httpx.AsyncClient(
                transport=mock,
                base_url=BASE_URL,
            ),
            max_retries=0,
        )
        sg = _async_region_client("ap-southeast-1", SG, [])
        router = _async_router({"cn-beijing": bad, "ap-southeast-1": sg})
        result = await router.models.available_regions("qwen-max")
        assert result.regions["cn-beijing"].error is not None
        assert result.regions["ap-southeast-1"].available is True
        assert result.available_in == ["ap-southeast-1"]
        await bad.aclose()
        await sg.aclose()