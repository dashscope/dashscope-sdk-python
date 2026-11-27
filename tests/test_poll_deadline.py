# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for the shared batch/compression wait budget."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.errors import DashScopeError
from dashscope.resources.batches import Batch
from dashscope.resources.fine_tunes_compress import CompressJob
from dashscope.transport import _deadline
from dashscope.transport import http as transport_http


@pytest.fixture(params=["batch", "compress"])
def job(request):
    if request.param == "batch":
        return (
            lambda client: client.batches,
            Batch(id="batch-1", status="in_progress"),
            {"id": "batch-1", "status": "completed"},
        )
    return (
        lambda client: client.fine_tunes.compress.jobs,
        CompressJob(job_id="cj-1", status="RUNNING"),
        {"output": {"job_id": "cj-1", "status": "SUCCEEDED"}},
    )


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        _deadline,
        "time",
        SimpleNamespace(monotonic=lambda: now[0]),
    )

    def advance(seconds):
        now[0] += seconds

    return advance


def test_no_poll_after_waiter_deadline(job, clock):
    resource, seed, _ = job

    def unexpected(request):
        pytest.fail(f"request sent past the deadline: {request.url}")

    with httpx.Client(
        transport=httpx.MockTransport(unexpected),
    ) as http_client:
        client = Client(api_key="test", http_client=http_client)
        with pytest.raises(DashScopeError, match="timed out"):
            resource(client).wait(
                seed,
                timeout=1,
                poll_interval=2,
                _sleep=clock,
            )


@pytest.mark.asyncio
async def test_async_no_poll_after_waiter_deadline(job, clock):
    resource, seed, _ = job

    def unexpected(request):
        pytest.fail(f"request sent past the deadline: {request.url}")

    async def sleep(delay):
        clock(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(unexpected),
    ) as http_client:
        client = AsyncClient(api_key="test", http_client=http_client)
        with pytest.raises(DashScopeError, match="timed out"):
            await resource(client).wait(
                seed,
                timeout=1,
                poll_interval=2,
                _sleep=sleep,
            )


def test_initial_read_and_retries_share_remaining_http_timeout(
    job,
    clock,
    monkeypatch,
):
    resource, seed, complete = job
    requests = []

    def handler(request):
        requests.append(request.extensions["timeout"])
        if len(requests) == 1:
            clock(0.25)
            return httpx.Response(503, headers={"Retry-After": "1"})
        return httpx.Response(200, json=complete)

    monkeypatch.setattr(transport_http, "_sleep", clock)
    with httpx.Client(
        transport=httpx.MockTransport(handler),
        timeout=5,
    ) as http_client:
        client = Client(api_key="test", http_client=http_client)
        job_id = seed.id if isinstance(seed, Batch) else seed.job_id
        assert resource(client).wait(job_id, timeout=2, max_retries=1).finished
    assert requests[0] == dict.fromkeys(
        ("connect", "read", "write", "pool"),
        2.0,
    )
    assert requests[1] == dict.fromkeys(
        ("connect", "read", "write", "pool"),
        0.75,
    )


def test_late_terminal_response_does_not_turn_timeout_into_success(job, clock):
    resource, seed, complete = job

    def handler(request):
        clock(1)
        return httpx.Response(200, json=complete)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = Client(api_key="test", http_client=http_client)
        with pytest.raises(DashScopeError, match="timed out"):
            resource(client).wait(
                seed,
                timeout=1,
                poll_interval=0.25,
                _sleep=clock,
            )


@pytest.mark.asyncio
async def test_async_deadline_cancels_in_flight_poll(job):
    resource, seed, _ = job
    cancelled = asyncio.Event()

    async def handler(request):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = AsyncClient(api_key="test", http_client=http_client)
        # Outer timeout keeps the regression test bounded if the waiter
        # accidentally leaves the HTTP request unbounded again.
        with pytest.raises(DashScopeError, match="timed out"):
            await asyncio.wait_for(
                resource(client).wait(seed, timeout=0.2, poll_interval=0),
                timeout=2,
            )
        assert cancelled.is_set()


def test_retry_after_cannot_outlive_wait_budget(job, clock, monkeypatch):
    resource, seed, _ = job
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(503, headers={"Retry-After": "10"})

    def unexpected_backoff(delay):
        pytest.fail(f"retry delay did not fit the deadline: {delay}")

    monkeypatch.setattr(transport_http, "_sleep", unexpected_backoff)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = Client(api_key="test", http_client=http_client)
        job_id = seed.id if isinstance(seed, Batch) else seed.job_id
        with pytest.raises(DashScopeError, match="timed out"):
            resource(client).wait(job_id, timeout=2, max_retries=1)
    assert len(sent) == 1
