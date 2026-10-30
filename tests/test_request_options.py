# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the per-request ``RequestOptions`` channel (T1)."""

# pytest fixture injection looks like unused/redefined names to pylint.
# pylint: disable=redefined-outer-name,unused-argument

from __future__ import annotations

import json
import logging

import httpx
import pytest

from dashscope.errors import APIConnectionError
from dashscope.transport import (
    AsyncTransport,
    RequestOptions,
    SyncTransport,
)
from dashscope.transport import http as transport_http
from dashscope.transport._options import (
    deep_merge,
    merge_query,
    request_options,
    resolve_max_retries,
    resolve_timeout,
)

BASE_URL = "https://api.test.local"


class _Recorder:
    def __init__(self, responses=None):
        self.calls = 0
        self.requests = []
        self._responses = list(responses or [])

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.calls += 1
        if self.calls <= len(self._responses):
            item = self._responses[self.calls - 1]
            if isinstance(item, Exception):
                raise item
            status, headers = item
            return httpx.Response(status, headers=headers)
        return httpx.Response(200)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def sent_json(self):
        return json.loads(self.last.content)

    def sent_query(self):
        return dict(self.last.url.params)

    def sent_timeout(self):
        return self.last.extensions["timeout"]


def _make_sync(recorder, **kwargs):
    http_client = httpx.Client(
        transport=httpx.MockTransport(recorder.handler),
        base_url=BASE_URL,
    )
    return SyncTransport(base_url=BASE_URL, http_client=http_client, **kwargs)


def _make_async(recorder, **kwargs):
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(recorder.handler),
        base_url=BASE_URL,
    )
    return AsyncTransport(base_url=BASE_URL, http_client=http_client, **kwargs)


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(transport_http, "_sleep", sleeps.append)

    async def _asleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(transport_http, "_asleep", _asleep)
    return sleeps


# --------------------------------------------------------------------------
# Pure helpers: request_options / deep_merge / merge_query / resolvers
# --------------------------------------------------------------------------


def test_request_options_is_none_when_nothing_is_set():
    # The transport keeps its no-options fast path, so an all-defaults
    # resource call must not materialize an empty RequestOptions.
    assert request_options() is None
    assert request_options({}, {}, {}) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"extra_headers": {"X-A": "1"}},
        {"extra_query": {"a": "1"}},
        {"extra_body": {"a": 1}},
    ],
)
def test_request_options_built_when_any_extra_is_set(kwargs):
    options = request_options(**kwargs)
    assert isinstance(options, RequestOptions)
    for name, value in kwargs.items():
        assert getattr(options, name) == value


def test_deep_merge_nested_dicts_merge_scalars_replace():
    base = {"model": "qwen-max", "params": {"top_p": 0.5, "seed": 1}}
    merged = deep_merge(base, {"params": {"top_p": 0.9}, "stream": True})
    assert merged == {
        "model": "qwen-max",
        "params": {"top_p": 0.9, "seed": 1},
        "stream": True,
    }


def test_deep_merge_lists_replace_rather_than_concatenate():
    merged = deep_merge({"messages": [{"role": "user"}]}, {"messages": []})
    assert merged == {"messages": []}


def test_deep_merge_leaves_both_inputs_untouched():
    base = {"params": {"seed": 1}}
    extra = {"params": {"top_p": 0.5}}
    deep_merge(base, extra)
    assert base == {"params": {"seed": 1}}
    assert extra == {"params": {"top_p": 0.5}}


def test_merge_query_passes_params_through_without_options():
    params = {"page_no": 1}
    assert merge_query(params, None) is params
    assert merge_query(params, RequestOptions()) is params


def test_merge_query_extra_wins_on_conflict():
    options = RequestOptions(extra_query={"page_no": 9, "flag": "x"})
    assert merge_query({"page_no": 1}, options) == {
        "page_no": 9,
        "flag": "x",
    }


def test_merge_query_adds_query_where_call_sent_none():
    options = RequestOptions(extra_query={"flag": "x"})
    assert merge_query(None, options) == {"flag": "x"}


def test_resolve_timeout_explicit_argument_wins():
    options = RequestOptions(timeout=1.5)
    assert resolve_timeout(9.0, options) == 9.0


def test_resolve_timeout_falls_back_to_options():
    options = RequestOptions(timeout=1.5)
    assert resolve_timeout(None, options) == 1.5
    assert resolve_timeout(None, None) is None


def test_resolve_max_retries_override_default_and_clamp():
    assert resolve_max_retries(2, None) == 2
    assert resolve_max_retries(2, RequestOptions()) == 2
    assert resolve_max_retries(2, RequestOptions(max_retries=0)) == 0
    assert resolve_max_retries(2, RequestOptions(max_retries=5)) == 5
    assert resolve_max_retries(2, RequestOptions(max_retries=-1)) == 0


# --------------------------------------------------------------------------
# Sync transport: what actually reaches the wire
# --------------------------------------------------------------------------


def test_extra_headers_reach_the_wire(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec)
    options = RequestOptions(extra_headers={"X-Experiment": "on"})
    t.request("GET", "/x", options=options)
    assert rec.last.headers["x-experiment"] == "on"


def test_extra_headers_override_sdk_identity_case_insensitively(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec)
    options = RequestOptions(
        extra_headers={"X-DashScope-SDK-Client": "custom/1.0"},
    )
    t.request("GET", "/x", options=options)
    sent = rec.last.headers
    assert sent["x-dashscope-sdk-client"] == "custom/1.0"
    # A case-insensitive override replaces rather than duplicates.
    assert len(sent.get_list("x-dashscope-sdk-client")) == 1


def test_extra_headers_beat_caller_headers_on_the_same_key(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec)
    options = RequestOptions(extra_headers={"X-Trace": "extra"})
    t.request("GET", "/x", headers={"X-Trace": "caller"}, options=options)
    assert rec.last.headers["x-trace"] == "extra"


def test_extra_headers_beat_per_call_credentials(no_sleep):
    # Documented precedence puts extra_headers above everything, so a
    # caller can point one request at a different key than the client.
    rec = _Recorder()
    t = _make_sync(rec, api_key="client-key")
    options = RequestOptions(extra_headers={"Authorization": "Bearer extra"})
    t.request("GET", "/x", api_key="per-call", options=options)
    assert rec.last.headers["authorization"] == "Bearer extra"


def test_extra_query_joins_the_query_string(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec)
    options = RequestOptions(extra_query={"beta": "true"})
    t.request("GET", "/x", params={"page_no": 1}, options=options)
    assert rec.sent_query() == {"page_no": "1", "beta": "true"}


def test_extra_body_deep_merges_into_json(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec)
    options = RequestOptions(extra_body={"params": {"seed": 7}})
    t.request(
        "POST",
        "/chat",
        json={"model": "qwen-max", "params": {"top_p": 0.5}},
        options=options,
    )
    assert rec.sent_json() == {
        "model": "qwen-max",
        "params": {"top_p": 0.5, "seed": 7},
    }


def test_extra_body_on_a_request_without_a_json_body(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec)
    options = RequestOptions(extra_body={"flag": True})
    t.request("POST", "/x", options=options)
    assert rec.sent_json() == {"flag": True}


def test_extra_body_is_ignored_with_a_warning_for_raw_content(
    no_sleep,
    caplog,
):
    rec = _Recorder()
    t = _make_sync(rec)
    options = RequestOptions(extra_body={"flag": True})
    with caplog.at_level(logging.WARNING, logger="dashscope._logging"):
        t.request("POST", "/upload", content=b"raw", options=options)
    assert rec.last.content == b"raw"
    messages = [r.getMessage() for r in caplog.records]
    assert any("ignoring extra_body" in m for m in messages)


def test_options_timeout_applies_when_the_call_omits_its_own(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec)
    t.request("GET", "/x", options=RequestOptions(timeout=3.5))
    assert rec.sent_timeout()["read"] == 3.5


def test_call_timeout_wins_over_options_timeout(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec)
    t.request(
        "GET",
        "/x",
        timeout=8.0,
        options=RequestOptions(timeout=3.5),
    )
    assert rec.sent_timeout()["read"] == 8.0


def test_extra_body_is_ignored_with_a_warning_for_a_non_object_json(
    no_sleep,
    caplog,
):
    rec = _Recorder()
    t = _make_sync(rec)
    options = RequestOptions(extra_body={"flag": True})
    with caplog.at_level(logging.WARNING, logger="dashscope._logging"):
        t.request("POST", "/batch", json=[{"a": 1}], options=options)
    assert rec.sent_json() == [{"a": 1}]
    messages = [r.getMessage() for r in caplog.records]
    assert any("body is list" in m for m in messages)


def test_max_retries_zero_disables_retrying_for_one_call(no_sleep):
    rec = _Recorder([(503, {})])
    t = _make_sync(rec, max_retries=3)
    resp = t.request("GET", "/x", options=RequestOptions(max_retries=0))
    assert resp.status_code == 503
    assert rec.calls == 1


def test_max_retries_one_retries_once(no_sleep):
    rec = _Recorder([(503, {}), (503, {}), (200, {})])
    t = _make_sync(rec, max_retries=5)
    resp = t.request("GET", "/x", options=RequestOptions(max_retries=1))
    assert resp.status_code == 503
    assert rec.calls == 2
    assert len(no_sleep) == 1


def test_max_retries_override_also_caps_transport_error_retries(no_sleep):
    rec = _Recorder([httpx.ConnectError("boom") for _ in range(4)])
    t = _make_sync(rec, max_retries=5)
    with pytest.raises(APIConnectionError):
        t.request("GET", "/x", options=RequestOptions(max_retries=1))
    assert rec.calls == 2


def test_without_options_the_client_default_retry_budget_applies(no_sleep):
    rec = _Recorder([(503, {}), (503, {}), (200, {})])
    t = _make_sync(rec, max_retries=2)
    resp = t.request("GET", "/x")
    assert resp.status_code == 200
    assert rec.calls == 3


def test_no_options_sends_no_extra_headers_or_query(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec, api_key="client-key")
    t.request("GET", "/x", params={"page_no": 1}, json={"a": 1})
    assert rec.sent_query() == {"page_no": "1"}
    assert rec.sent_json() == {"a": 1}
    assert rec.last.headers["authorization"] == "Bearer client-key"
    assert "x-experiment" not in rec.last.headers


def test_open_stream_applies_options(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec)
    options = RequestOptions(
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
        extra_body={"params": {"seed": 7}},
        max_retries=0,
    )
    t.open_stream(
        "POST",
        "/chat",
        json={"params": {"top_p": 0.5}},
        options=options,
    ).close()
    assert rec.last.headers["x-experiment"] == "on"
    assert rec.sent_query() == {"beta": "true"}
    assert rec.sent_json() == {"params": {"top_p": 0.5, "seed": 7}}


# --------------------------------------------------------------------------
# Async transport: same contract
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_extra_headers_and_query_and_body(no_sleep):
    rec = _Recorder()
    t = _make_async(rec)
    options = RequestOptions(
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
        extra_body={"params": {"seed": 7}},
    )
    await t.request(
        "POST",
        "/chat",
        params={"page_no": 1},
        headers={"X-Trace": "caller"},
        json={"params": {"top_p": 0.5}},
        options=options,
    )
    assert rec.last.headers["x-experiment"] == "on"
    assert rec.last.headers["x-trace"] == "caller"
    assert rec.sent_query() == {"page_no": "1", "beta": "true"}
    assert rec.sent_json() == {"params": {"top_p": 0.5, "seed": 7}}
    await t.aclose()


@pytest.mark.asyncio
async def test_async_extra_headers_beat_caller_headers(no_sleep):
    rec = _Recorder()
    t = _make_async(rec)
    options = RequestOptions(extra_headers={"X-Trace": "extra"})
    await t.request(
        "GET",
        "/x",
        headers={"X-Trace": "caller"},
        options=options,
    )
    assert rec.last.headers["x-trace"] == "extra"
    await t.aclose()


@pytest.mark.asyncio
async def test_async_max_retries_zero_disables_retrying(no_sleep):
    rec = _Recorder([(503, {})])
    t = _make_async(rec, max_retries=3)
    resp = await t.request("GET", "/x", options=RequestOptions(max_retries=0))
    assert resp.status_code == 503
    assert rec.calls == 1
    await t.aclose()


@pytest.mark.asyncio
async def test_async_max_retries_one_retries_once(no_sleep):
    rec = _Recorder([(503, {}), (503, {}), (200, {})])
    t = _make_async(rec, max_retries=5)
    resp = await t.request("GET", "/x", options=RequestOptions(max_retries=1))
    assert resp.status_code == 503
    assert rec.calls == 2
    assert len(no_sleep) == 1
    await t.aclose()


@pytest.mark.asyncio
async def test_async_options_timeout_applies(no_sleep):
    rec = _Recorder()
    t = _make_async(rec)
    await t.request("GET", "/x", options=RequestOptions(timeout=3.5))
    assert rec.sent_timeout()["read"] == 3.5
    await t.aclose()


@pytest.mark.asyncio
async def test_async_open_stream_applies_options(no_sleep):
    rec = _Recorder()
    t = _make_async(rec)
    options = RequestOptions(
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
        extra_body={"params": {"seed": 7}},
    )
    response = await t.open_stream(
        "POST",
        "/chat",
        json={"params": {"top_p": 0.5}},
        options=options,
    )
    await response.aclose()
    assert rec.last.headers["x-experiment"] == "on"
    assert rec.sent_query() == {"beta": "true"}
    assert rec.sent_json() == {"params": {"top_p": 0.5, "seed": 7}}
    await t.aclose()


@pytest.mark.asyncio
async def test_async_extra_body_ignored_with_warning_for_raw_content(
    no_sleep,
    caplog,
):
    rec = _Recorder()
    t = _make_async(rec)
    options = RequestOptions(extra_body={"flag": True})
    with caplog.at_level(logging.WARNING, logger="dashscope._logging"):
        await t.request("POST", "/upload", content=b"raw", options=options)
    assert rec.last.content == b"raw"
    messages = [r.getMessage() for r in caplog.records]
    assert any("ignoring extra_body" in m for m in messages)
    await t.aclose()
