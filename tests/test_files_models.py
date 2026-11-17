# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for files/models resources (M1 task 1.8)."""

from __future__ import annotations

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.errors import BadRequestError, NotFoundError, SdkError

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


def test_files_upload(tmp_path):
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content.decode("utf-8", "replace")
        return httpx.Response(
            200,
            json={
                "request_id": "req-u",
                "data": {
                    "uploaded_files": [{"name": "t.jsonl", "file_id": "f-1"}],
                },
            },
        )

    file_path = tmp_path / "t.jsonl"
    file_path.write_text('{"a": 1}\n', encoding="utf-8")
    client = _make_client(handler)
    resp = client.files.upload(
        file_path=str(file_path),
        purpose="fine-tune",
        description="train",
    )
    assert captured["path"] == "/api/v1/files"
    assert captured["content_type"].startswith("multipart/form-data")
    assert 'name="purpose"' not in captured["body"]
    assert 'name="descriptions"' in captured["body"]
    assert '["train"]' not in captured["body"]
    assert "t.jsonl" in captured["body"]
    assert resp.file_id == "f-1"
    assert resp.name == "t.jsonl"


def test_files_upload_validation(tmp_path):
    client = _make_client(lambda r: httpx.Response(200, json={"data": {}}))
    with pytest.raises(SdkError) as missing:
        client.files.upload(file_path=str(tmp_path / "nope.jsonl"))
    assert missing.value.code == "sdk.IoError"
    assert "nope.jsonl" in missing.value.message
    assert missing.value.allow_retry is False

    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl"):
        client.files.upload(file_path=str(bad))

    resp = client.files.upload(file_path=str(bad), purpose="inference")
    assert resp is not None


def test_files_upload_unreadable_file_is_catalog_io_error(
    tmp_path,
    monkeypatch,
):
    client = _make_client(lambda r: httpx.Response(200, json={"data": {}}))
    path = tmp_path / "x.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    def _deny(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("builtins.open", _deny)
    with pytest.raises(SdkError) as ei:
        client.files.upload(file_path=str(path))
    assert ei.value.code == "sdk.IoError"
    assert ei.value.message == "Local I/O failed: denied."


def test_files_upload_retry_resends_full_body(tmp_path, monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(503, json={"message": "busy"})
        return httpx.Response(
            200,
            json={
                "request_id": "req-r",
                "data": {"uploaded_files": [{"file_id": "f-r"}]},
            },
        )

    from dashscope.transport import http as transport_http

    monkeypatch.setattr(transport_http, "_sleep", lambda seconds: None)
    file_path = tmp_path / "t.jsonl"
    file_path.write_text('{"a": 1}\n', encoding="utf-8")
    client = _make_client(handler)
    resp = client.files.upload(file_path=str(file_path))
    assert resp.file_id == "f-r"
    assert len(bodies) == 2
    # Multipart boundaries are random per request; what matters is the
    # retry re-sends the full payload rather than a drained handle.
    for body in bodies:
        assert b'{"a": 1}' in body
        assert b"t.jsonl" in body


def test_files_list_get_delete():
    def handler(request):
        if request.method == "GET" and request.url.path == "/api/v1/files":
            assert request.url.params["page_no"] == "2"
            return httpx.Response(
                200,
                json={
                    "request_id": "r",
                    "data": {
                        "page_no": 2,
                        "page_size": 10,
                        "total": 1,
                        "files": [{"file_id": "f-1", "name": "t.jsonl"}],
                    },
                },
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": {"file_id": "f-1", "name": "t.jsonl", "size": 10},
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json={"request_id": "r"})
        raise AssertionError("unexpected request")

    client = _make_client(handler)
    listed = client.files.list(page_no=2)
    assert listed.total_count == 1
    assert listed.files[0].file_id == "f-1"
    got = client.files.get("f-1")
    assert got.size == 10
    deleted = client.files.delete("f-1")
    assert deleted.deleted is True


def test_files_get_missing_file_raises_not_found():
    # DashScope reports a missing file as a 400 InvalidParameter ("File not
    # found.") rather than a 404, so the catalog mapper yields BadRequestError.
    # get/delete reclassify it so ``except NotFoundError`` works, while keeping
    # the server's own status/code/request_id (never an invented 404).
    client = _make_client(
        lambda r: httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "File not found.",
                "request_id": "req-nf",
            },
        ),
    )
    with pytest.raises(NotFoundError) as exc:
        client.files.get("missing-id")
    err = exc.value
    assert err.code == "InvalidParameter"
    assert err.status_code == 400
    assert err.request_id == "req-nf"
    assert err.message == "File not found."


def test_files_delete_missing_file_raises_not_found():
    client = _make_client(
        lambda r: httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "File not found.",
            },
        ),
    )
    with pytest.raises(NotFoundError):
        client.files.delete("missing-id")


def test_files_missing_file_not_found_regardless_of_status():
    # InvalidParameter selects BadRequestError at any status (the wire code
    # outranks the status), so the reclassification cannot hinge on 400.
    for status in (400, 404):
        client = _make_client(
            lambda r, s=status: httpx.Response(
                s,
                json={
                    "code": "InvalidParameter",
                    "message": "File not found.",
                },
            ),
        )
        with pytest.raises(NotFoundError) as exc:
            client.files.get("missing-id")
        assert exc.value.status_code == status


def test_files_get_delete_empty_file_id_rejected_locally():
    # "" (or blank) would collapse the path to "/files/" and hit the list
    # endpoint; the call must fail locally without touching the transport.
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"data": {}})

    client = _make_client(handler)
    for file_id in ("", "   "):
        with pytest.raises(ValueError, match="file_id"):
            client.files.get(file_id)
        with pytest.raises(ValueError, match="file_id"):
            client.files.delete(file_id)
    assert calls == []


@pytest.mark.asyncio
async def test_async_files_get_delete_empty_file_id_rejected_locally():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"data": {}})

    client = _make_async_client(handler)
    for file_id in ("", "   "):
        with pytest.raises(ValueError, match="file_id"):
            await client.files.get(file_id)
        with pytest.raises(ValueError, match="file_id"):
            await client.files.delete(file_id)
    assert calls == []
    await client.aclose()


def test_files_get_other_bad_request_stays_bad_request():
    # Only a "not found"/"not exist" message is reclassified; any other
    # InvalidParameter (a genuinely malformed request) stays BadRequestError.
    client = _make_client(
        lambda r: httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "file_id contains illegal characters",
            },
        ),
    )
    with pytest.raises(BadRequestError) as exc:
        client.files.get("bad id")
    assert type(exc.value) is BadRequestError


def test_files_get_missing_file_no_raise_reports_envelope():
    # With raise_on_error=False nothing is raised, so there is no exception to
    # reclassify; the envelope keeps the gateway's own code.
    client = _make_client(
        lambda r: httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "File not found.",
                "request_id": "req-nf",
            },
        ),
        raise_on_error=False,
    )
    info = client.files.get("missing-id")
    assert info.is_error
    assert info.code == "InvalidParameter"
    assert info.request_id == "req-nf"


@pytest.mark.asyncio
async def test_async_files_get_missing_file_raises_not_found():
    client = _make_async_client(
        lambda r: httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "File not found.",
            },
        ),
    )
    with pytest.raises(NotFoundError):
        await client.files.get("missing-id")
    await client.aclose()


@pytest.mark.asyncio
async def test_async_files_delete_missing_file_raises_not_found():
    client = _make_async_client(
        lambda r: httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "File not found.",
            },
        ),
    )
    with pytest.raises(NotFoundError):
        await client.files.delete("missing-id")
    await client.aclose()


def test_models_list_with_provider_filter():
    # ``data``-wrapped pages stay supported for gateways that re-wrap the
    # native ``output`` envelope OpenAI-style. ``provider`` / ``providers``
    # are sent server-side as a repeated ``providers`` param, and the
    # gateway's filtered total drives pagination.
    catalog = [
        {"model": "qwen-plus", "provider": "qwen"},
        {"model": "qwen3-max", "provider": "qwen"},
        {"model": "llama-3", "provider": "community"},
    ]
    captured = {}

    def handler(request):
        captured["params"] = request.url.params
        providers = request.url.params.get_list("providers")
        models = (
            [m for m in catalog if m["provider"] in providers]
            if providers
            else catalog
        )
        return httpx.Response(
            200,
            json={
                "request_id": "r",
                "data": {
                    "page_no": 1,
                    "page_size": 10,
                    "total": len(models),
                    "models": models,
                },
            },
        )

    client = _make_client(handler)
    all_models = client.models.list()
    assert dict(captured["params"]) == {"page_no": "1", "page_size": "10"}
    assert len(all_models.models) == 3
    assert all_models.total_count == 3

    qwen = client.models.list(provider="qwen")
    assert captured["params"].get_list("providers") == ["qwen"]
    assert [m.model for m in qwen.models] == ["qwen-plus", "qwen3-max"]
    assert qwen.total_count == 2

    community = client.models.list(providers=["community"])
    assert captured["params"].get_list("providers") == ["community"]
    assert [m.model for m in community.models] == ["llama-3"]


def test_models_list_array_filters_use_repeated_keys():
    # Array filters serialize as repeated parameter names, not a joined
    # string or a bracketed key: capabilities=TG&capabilities=Reasoning.
    captured = {}

    def handler(request):
        captured["params"] = request.url.params
        return httpx.Response(
            200,
            json={"request_id": "r", "output": {"models": [], "total": 0}},
        )

    client = _make_client(handler)
    client.models.list(
        providers=["qwen", "deepseek"],
        inference_providers=["aliyun-bailian"],
        capabilities=["TG", "Reasoning"],
        features=["function-calling"],
        supports=["fine-tune"],
        deployment_methods=["ptu"],
        deployment_ptu_service_tiers=["standard"],
    )
    params = captured["params"]
    assert params.get_list("providers") == ["qwen", "deepseek"]
    assert params.get_list("inference_providers") == ["aliyun-bailian"]
    assert params.get_list("capabilities") == ["TG", "Reasoning"]
    assert params.get_list("features") == ["function-calling"]
    assert params.get_list("supports") == ["fine-tune"]
    assert params.get_list("deployment_methods") == ["ptu"]
    assert params.get_list("deployment_ptu_service_tiers") == ["standard"]


def test_models_list_scalar_filters_pass_through():
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={"request_id": "r", "output": {"models": [], "total": 0}},
        )

    client = _make_client(handler)
    client.models.list(
        name="qwen",
        model="qwen-plus",
        language="zh",
        context_window=131072,
        service_site="international",
    )
    assert captured["params"] == {
        "page_no": "1",
        "page_size": "10",
        "name": "qwen",
        "model": "qwen-plus",
        "language": "zh",
        "context_window": "131072",
        "service_site": "international",
    }


def test_models_list_provider_merges_into_providers_without_dupes():
    captured = {}

    def handler(request):
        captured["params"] = request.url.params
        return httpx.Response(
            200,
            json={"request_id": "r", "output": {"models": [], "total": 0}},
        )

    client = _make_client(handler)
    client.models.list(provider="qwen", providers=["qwen", "deepseek"])
    assert captured["params"].get_list("providers") == ["qwen", "deepseek"]


def test_models_list_parses_rich_metadata():
    body = {
        "request_id": "r",
        "output": {
            "page_no": 1,
            "page_size": 10,
            "total": 1,
            "models": [
                {
                    "model": "qwen3-max",
                    "name": "Qwen3-Max",
                    "description": "flagship",
                    "capabilities": ["TG", "Reasoning"],
                    "inference_provider": "aliyun-bailian",
                    "features": ["function-calling"],
                    "published_time": "2025-09-01",
                    "equivalent_snapshot": "qwen3-max-2025-09-01",
                    "prices": [
                        {
                            "range_name": "Default",
                            "prices": [
                                {
                                    "type": "input",
                                    "price": "0.0008",
                                    "price_unit": "per 1k tokens",
                                    "price_name": "Input",
                                },
                            ],
                        },
                    ],
                    "inference_metadata": {
                        "request_modality": ["Text"],
                        "response_modality": ["Text"],
                    },
                    "model_info": {
                        "context_window": 262144,
                        "max_input_tokens": 258048,
                        "max_output_tokens": 32768,
                    },
                },
            ],
        },
    }
    client = _make_client(lambda r: httpx.Response(200, json=body))
    model = client.models.list().models[0]
    assert model.name == "Qwen3-Max"
    assert model.capabilities == ["TG", "Reasoning"]
    assert model.inference_provider == "aliyun-bailian"
    assert model.equivalent_snapshot == "qwen3-max-2025-09-01"
    assert model.prices[0].range_name == "Default"
    assert model.prices[0].prices[0].price == "0.0008"  # verbatim string
    assert model.inference_metadata.request_modality == ["Text"]
    assert model.model_info.context_window == 262144
    assert model.model_info.max_output_tokens == 32768


def test_models_list_pagination_preserves_filters():
    catalog = [{"model": f"m{i}", "provider": "qwen"} for i in range(5)]
    seen = []

    def handler(request):
        params = request.url.params
        seen.append(
            (int(params.get("page_no", 1)), params.get_list("capabilities")),
        )
        page_no = int(params.get("page_no", 1))
        page_size = int(params.get("page_size", 10))
        start = (page_no - 1) * page_size
        return httpx.Response(
            200,
            json={
                "request_id": "r",
                "output": {
                    "models": catalog[start : start + page_size],
                    "page_no": page_no,
                    "page_size": page_size,
                    "total": len(catalog),
                },
            },
        )

    client = _make_client(handler)
    page = client.models.list(
        page_no=1,
        page_size=2,
        capabilities=["TG"],
        provider="qwen",
    )
    collected = [m.model for p in page.iter_pages() for m in p]
    assert collected == ["m0", "m1", "m2", "m3", "m4"]
    # Every advanced page re-sends the same filters.
    assert seen == [(1, ["TG"]), (2, ["TG"]), (3, ["TG"])]


@pytest.mark.asyncio
async def test_models_async_list_filters_and_pagination():
    catalog = [{"model": f"m{i}"} for i in range(5)]
    seen = []

    def handler(request):
        params = request.url.params
        seen.append(
            (int(params.get("page_no", 1)), params.get_list("providers")),
        )
        page_no = int(params.get("page_no", 1))
        page_size = int(params.get("page_size", 10))
        start = (page_no - 1) * page_size
        return httpx.Response(
            200,
            json={
                "request_id": "r",
                "output": {
                    "models": catalog[start : start + page_size],
                    "page_no": page_no,
                    "page_size": page_size,
                    "total": len(catalog),
                },
            },
        )

    client = _make_async_client(handler)
    page = await client.models.list(page_no=1, page_size=2, provider="qwen")
    assert seen[0] == (1, ["qwen"])
    collected = [
        m.model async for p in page.iter_pages_async() for m in p
    ]
    assert collected == ["m0", "m1", "m2", "m3", "m4"]
    assert seen == [(1, ["qwen"]), (2, ["qwen"]), (3, ["qwen"])]
    await client.aclose()


def test_models_list_parses_native_output_envelope():
    # The native /api/v1/models envelope nests the page under
    # ``output``; parsing only ``data`` yielded an empty list with
    # default pagination against the real endpoint.
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-page",
                "output": {
                    "page_no": 1,
                    "page_size": 3,
                    "total": 8,
                    "models": [
                        {"model": "qwen-plus", "source": "dashscope"},
                        {"model": "qwen3-max", "source": "bailian"},
                        {"model": "qwen-turbo", "source": "dashscope"},
                    ],
                },
            },
        )

    client = _make_client(handler)
    page = client.models.list(page_no=1, page_size=3)
    assert page.page_no == 1
    assert page.page_size == 3
    assert page.total_count == 8
    assert [m.model for m in page.models] == [
        "qwen-plus",
        "qwen3-max",
        "qwen-turbo",
    ]
    assert page.request_id == "req-page"


def test_models_list_falls_back_to_item_count_when_total_missing():
    # Gateways that omit ``total`` still have to report a meaningful
    # ``total_count``: reading the field default would show an empty
    # catalog for a page that clearly has entries.
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-no-total",
                "output": {
                    "page_no": 1,
                    "page_size": 10,
                    "models": [
                        {"model": "qwen-plus", "source": "dashscope"},
                        {"model": "qwen3-max", "source": "bailian"},
                    ],
                },
            },
        )

    page = _make_client(handler).models.list()
    assert [m.model for m in page.models] == ["qwen-plus", "qwen3-max"]
    assert page.total_count == 2


def test_models_list_parses_a_bare_list_page():
    # A page delivered as a bare list carries no ``page_no`` / ``total`` to
    # read, so it is taken as one whole page. The generic envelope lifter
    # only unwraps a dict, which parsed this shape as an empty catalog.
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-bare-list",
                "output": [
                    {"model": "qwen-plus", "source": "dashscope"},
                    {"model": "qwen3-max", "source": "bailian"},
                ],
            },
        )

    page = _make_client(handler).models.list()
    assert [m.model for m in page.models] == ["qwen-plus", "qwen3-max"]
    assert page.page_no == 1
    assert page.total_count == 2
    assert page.request_id == "req-bare-list"


def test_models_list_retyped_models_does_not_poison_the_total():
    # ``total`` falls back to ``len(models)``, and the length of a drifted
    # string is not a page total: without a guard this reported a 10-item
    # catalog for a page that repaired down to zero entries.
    def handler(request):
        return httpx.Response(
            200,
            json={"request_id": "req-str", "output": {"models": "nope"}},
        )

    page = _make_client(handler).models.list()
    assert page.models == []
    assert page.total_count == 0


def test_models_list_scalar_page_degrades_instead_of_raising():
    # ``len()`` on a scalar page would escape as a bare TypeError from the
    # resource, outside any ``except DashScopeError`` the caller has.
    def handler(request):
        return httpx.Response(200, json={"request_id": "req-n", "output": 5})

    page = _make_client(handler).models.list()
    assert page.models == []
    assert page.total_count == 0


def test_files_list_retyped_files_does_not_poison_the_total():
    def handler(request):
        return httpx.Response(
            200,
            json={"request_id": "req-fs", "data": {"files": "nope"}},
        )

    page = _make_client(handler).files.list()
    assert page.files == []
    assert page.total_count == 0


def test_files_list_scalar_page_degrades_instead_of_raising():
    def handler(request):
        return httpx.Response(200, json={"request_id": "req-fn", "data": 5})

    page = _make_client(handler).files.list()
    assert page.files == []
    assert page.total_count == 0


def test_models_list_carries_raw_response_headers():
    # Uniform success-path contract: every HTTP resource mirrors the
    # gateway's response headers on ``resp.headers`` (1.x parity).
    def handler(request):
        return httpx.Response(
            200,
            json={"request_id": "req-h", "output": {"models": []}},
            headers={"x-request-id": "req-h", "x-gw-region": "cn-hz"},
        )

    client = _make_client(handler)
    page = client.models.list()
    assert page.headers["x-request-id"] == "req-h"
    assert page.headers["x-gw-region"] == "cn-hz"


def test_models_get():
    def handler(request):
        assert request.url.path == "/api/v1/models"
        assert request.url.params["model"] == "qwen-plus"
        assert request.url.params["page_no"] == "1"
        assert request.url.params["page_size"] == "1"
        return httpx.Response(
            200,
            json={
                "output": {
                    "models": [
                        {"model": "qwen-plus", "source": "dashscope"},
                    ],
                },
            },
        )

    client = _make_client(handler)
    model = client.models.get("qwen-plus")
    assert model.model == "qwen-plus"
    assert model.source == "dashscope"


def test_models_get_not_found():
    client = _make_client(
        lambda r: httpx.Response(
            200,
            json={"output": {"models": []}},
            headers={"x-request-id": "req-none"},
        ),
    )
    with pytest.raises(NotFoundError, match="not found") as exc:
        client.models.get("nope")
    error = exc.value
    assert error.code == "Model.NotExist"
    # The list call itself succeeded; the model was simply absent. So the
    # status is the one the server sent, not an invented 404.
    assert error.status_code == 200
    assert error.request_id == "req-none"


@pytest.mark.asyncio
async def test_async_files_models():
    def handler(request):
        if request.url.path == "/api/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "models": [
                            {"model": "qwen-plus", "source": "dashscope"},
                        ],
                    },
                },
            )
        if request.url.path == "/api/v1/files":
            return httpx.Response(
                200,
                json={
                    "data": {"files": [{"file_id": "f-9"}], "total_count": 1},
                },
            )
        return httpx.Response(200, json={"data": {"file_id": "f-9"}})

    client = _make_async_client(handler)
    models = await client.models.list()
    assert models.models[0].model == "qwen-plus"
    files = await client.files.list()
    assert files.files[0].file_id == "f-9"
    one = await client.files.get("f-9")
    assert one.file_id == "f-9"
    await client.aclose()


def test_business_code_reaches_single_item_responses():
    # DashScope can fail with HTTP 200 plus a business code. These two
    # parsers build their result from an item nested under data/output, so
    # the envelope has to be threaded on separately for is_error to mean the
    # same thing here as it does on chat and apps.
    def handler(request):
        if request.url.path == "/api/v1/models":
            return httpx.Response(
                200,
                json={
                    "request_id": "req-m",
                    "code": "Throttling",
                    "message": "slow down",
                    "output": {"models": []},
                },
            )
        return httpx.Response(
            200,
            json={
                "request_id": "req-f",
                "code": "Throttling",
                "message": "slow down",
                "data": {"file_id": "f-1"},
            },
        )

    client = _make_client(handler, raise_on_error=False)
    model = client.models.get("qwen-plus")
    assert model.is_error
    assert model.code == "Throttling"
    assert model.request_id == "req-m"

    info = client.files.get("f-1")
    assert info.is_error
    assert info.code == "Throttling"
    assert info.request_id == "req-f"


def test_nested_list_items_carry_no_envelope():
    def handler(request):
        assert request.url.path == "/api/v1/files"
        return httpx.Response(
            200,
            json={
                "request_id": "req-l",
                "data": {"files": [{"file_id": "f-1"}], "total": 1},
            },
        )

    page = _make_client(handler).files.list()
    assert page.request_id == "req-l"
    assert not page.is_error
    # The envelope belongs to the response, not to each item inside it.
    assert page.files[0].request_id is None
    assert not page.files[0].is_error
