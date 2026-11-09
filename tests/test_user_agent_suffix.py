# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``Client(user_agent_suffix=...)`` reaches HTTP and websocket alike.

The suffix is how the service attributes traffic to a surface (the
bundled CLI tags itself ``dashscope-cli``), so a channel that silently
drops it would make that surface's volume unmeasurable. These tests pin
both channels plus the ``with_options`` inheritance, and in particular
the pool-sharing case: a derived client reusing its parent's httpx pool
must still send its *own* agent.
"""

# pylint: disable=protected-access,redefined-outer-name

from __future__ import annotations

import httpx
import pytest

from dashscope.client import AsyncClient, Client
from dashscope.resources.audio.asr import Recognition
from dashscope.resources.audio.tts import SynthesisSession
from dashscope.resources.tingwu import TingWuRealtime
from dashscope.transport.http import default_user_agent
from dashscope.transport.ws import with_client_user_agent
from dashscope.version import __version__

BASE_URL = "https://api.test.local"
BASE_AGENT = f"dashscope-sdk-python/{__version__}"


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


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"data": []}, request=request)


class TestDefaultUserAgent:
    def test_without_suffix_is_unchanged(self):
        """The no-argument form must stay byte-identical for 1.x parity."""
        assert default_user_agent() == BASE_AGENT

    def test_suffix_appends_as_space_separated_token(self):
        assert default_user_agent("dashscope-cli") == (
            f"{BASE_AGENT} dashscope-cli"
        )

    def test_empty_suffix_is_ignored(self):
        assert default_user_agent("") == BASE_AGENT


class TestHttpChannel:
    def test_request_carries_suffixed_agent(self):
        seen = []

        def handler(request):
            seen.append(request.headers.get("user-agent"))
            return _ok(request)

        client = _make_client(handler, user_agent_suffix="dashscope-cli")
        try:
            client.transport.request("GET", "/models")
        finally:
            client.close()
        assert seen == [f"{BASE_AGENT} dashscope-cli"]

    def test_request_without_suffix_carries_base_agent(self):
        seen = []

        def handler(request):
            seen.append(request.headers.get("user-agent"))
            return _ok(request)

        client = _make_client(handler)
        try:
            client.transport.request("GET", "/models")
        finally:
            client.close()
        assert seen == [BASE_AGENT]

    def test_transport_exposes_its_agent(self):
        client = _make_client(_ok, user_agent_suffix="dashscope-cli")
        try:
            assert client.transport.user_agent == (
                f"{BASE_AGENT} dashscope-cli"
            )
            assert client.user_agent_suffix == "dashscope-cli"
        finally:
            client.close()

    async def test_async_client_parity(self):
        seen = []

        def handler(request):
            seen.append(request.headers.get("user-agent"))
            return _ok(request)

        client = _make_async_client(handler, user_agent_suffix="dashscope-cli")
        try:
            await client.transport.request("GET", "/models")
            assert client.user_agent_suffix == "dashscope-cli"
            assert client.transport.user_agent == (
                f"{BASE_AGENT} dashscope-cli"
            )
        finally:
            await client.aclose()
        assert seen == [f"{BASE_AGENT} dashscope-cli"]


class TestWithOptions:
    def test_suffix_is_inherited(self):
        client = _make_client(_ok, user_agent_suffix="dashscope-cli")
        try:
            derived = client.with_options()
            assert derived.user_agent_suffix == "dashscope-cli"
            assert derived.transport.user_agent == (
                f"{BASE_AGENT} dashscope-cli"
            )
        finally:
            client.close()

    def test_suffix_can_be_overridden(self):
        client = _make_client(_ok, user_agent_suffix="dashscope-cli")
        try:
            derived = client.with_options(user_agent_suffix="notebook")
            assert derived.user_agent_suffix == "notebook"
            assert derived.transport.user_agent == f"{BASE_AGENT} notebook"
        finally:
            client.close()

    def test_override_can_clear_the_suffix(self):
        """An explicit empty suffix drops the tag rather than inheriting."""
        client = _make_client(_ok, user_agent_suffix="dashscope-cli")
        try:
            derived = client.with_options(user_agent_suffix="")
            assert derived.transport.user_agent == BASE_AGENT
        finally:
            client.close()

    def test_shared_pool_still_sends_the_derived_agent(self):
        """Regression pin: the agent is applied per request, not per pool.

        ``with_options`` shares the parent's httpx client when no
        connection-level option changed, and the httpx client carries its
        own default headers. A naive reader may conclude the derived
        client therefore sends the parent's agent and "fix" it by
        splitting pools. It does not: ``_request_headers`` re-applies the
        transport's defaults on every request, overriding the pool's
        case-insensitively. Splitting pools here would cost connection
        reuse for nothing.
        """
        seen = []

        def handler(request):
            seen.append(request.headers.get("user-agent"))
            return _ok(request)

        client = _make_client(handler, user_agent_suffix="dashscope-cli")
        try:
            derived = client.with_options(user_agent_suffix="notebook")
            assert (
                derived.transport.http_client is client.transport.http_client
            )
            client.transport.request("GET", "/models")
            derived.transport.request("GET", "/models")
        finally:
            client.close()
        assert seen == [
            f"{BASE_AGENT} dashscope-cli",
            f"{BASE_AGENT} notebook",
        ]

    async def test_async_suffix_is_inherited(self):
        client = _make_async_client(_ok, user_agent_suffix="dashscope-cli")
        try:
            derived = client.with_options()
            assert derived.user_agent_suffix == "dashscope-cli"
            derived = client.with_options(user_agent_suffix="notebook")
            assert derived.transport.user_agent == f"{BASE_AGENT} notebook"
        finally:
            await client.aclose()


class TestWithClientUserAgent:
    class _Stub:
        def __init__(self, agent):
            self.transport = type("T", (), {"user_agent": agent})()

    def test_seeds_agent_when_absent(self):
        client = self._Stub(f"{BASE_AGENT} dashscope-cli")
        assert with_client_user_agent(client, None) == {
            "user-agent": f"{BASE_AGENT} dashscope-cli",
        }

    def test_caller_header_wins(self):
        client = self._Stub(f"{BASE_AGENT} dashscope-cli")
        assert with_client_user_agent(client, {"user-agent": "mine"}) == {
            "user-agent": "mine",
        }

    def test_caller_header_wins_under_any_casing(self):
        """Both spellings must not survive: that would send the header twice."""
        client = self._Stub(f"{BASE_AGENT} dashscope-cli")
        merged = with_client_user_agent(client, {"User-Agent": "mine"})
        assert merged == {"User-Agent": "mine"}
        assert [k.lower() for k in merged].count("user-agent") == 1

    def test_other_headers_are_preserved(self):
        client = self._Stub(BASE_AGENT)
        merged = with_client_user_agent(client, {"X-Trace": "1"})
        assert merged == {"X-Trace": "1", "user-agent": BASE_AGENT}

    def test_client_without_transport_is_passed_through(self):
        assert with_client_user_agent(object(), {"X-Trace": "1"}) == {
            "X-Trace": "1",
        }

    def test_input_mapping_is_not_mutated(self):
        client = self._Stub(BASE_AGENT)
        original = {"X-Trace": "1"}
        with_client_user_agent(client, original)
        assert original == {"X-Trace": "1"}


class TestWebsocketChannel:
    def test_ws_connection_seeds_the_client_agent(self, monkeypatch):
        captured = {}

        def fake_connection(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("dashscope.client.WsConnection", fake_connection)
        client = _make_client(_ok, user_agent_suffix="dashscope-cli")
        try:
            client.ws_connection(url="wss://ws.test.local/inference")
        finally:
            client.close()
        assert captured["headers"]["user-agent"] == (
            f"{BASE_AGENT} dashscope-cli"
        )

    def test_ws_connection_caller_header_wins(self, monkeypatch):
        captured = {}

        def fake_connection(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("dashscope.client.WsConnection", fake_connection)
        client = _make_client(_ok, user_agent_suffix="dashscope-cli")
        try:
            client.ws_connection(
                url="wss://ws.test.local/inference",
                headers={"User-Agent": "mine"},
            )
        finally:
            client.close()
        headers = captured["headers"]
        assert headers == {"User-Agent": "mine"}

    @pytest.mark.parametrize(
        "build",
        [
            lambda client: SynthesisSession(client, model="cosyvoice-v1"),
            lambda client: Recognition(client, model="paraformer-realtime"),
            lambda client: TingWuRealtime(
                client,
                model="tingwu-realtime",
                app_id="app-1",
            ),
        ],
        ids=["tts", "asr", "tingwu"],
    )
    def test_audio_sessions_seed_the_client_agent(self, build):
        """Each realtime session opens its own socket off the transport."""
        client = _make_client(_ok, user_agent_suffix="dashscope-cli")
        try:
            session = build(client)
            assert session._session_kwargs["headers"]["user-agent"] == (
                f"{BASE_AGENT} dashscope-cli"
            )
        finally:
            client.close()

    @pytest.mark.parametrize(
        "build",
        [
            lambda client: SynthesisSession(
                client,
                model="cosyvoice-v1",
                headers={"user-agent": "mine"},
            ),
            lambda client: Recognition(
                client,
                model="paraformer-realtime",
                headers={"user-agent": "mine"},
            ),
        ],
        ids=["tts", "asr"],
    )
    def test_audio_sessions_let_caller_headers_win(self, build):
        client = _make_client(_ok, user_agent_suffix="dashscope-cli")
        try:
            session = build(client)
            assert session._session_kwargs["headers"] == {
                "user-agent": "mine",
            }
        finally:
            client.close()

    def test_realtime_session_inherits_the_client_agent(self):
        from dashscope.resources.audio.realtime import RealtimeSession

        client = _make_client(_ok, user_agent_suffix="dashscope-cli")
        try:
            session = RealtimeSession(
                client,
                model="qwen-omni-turbo-realtime",
                ws_url="wss://ws.test.local/realtime",
            )
            assert session._headers["user-agent"] == (
                f"{BASE_AGENT} dashscope-cli"
            )
        finally:
            client.close()

    def test_realtime_session_caller_header_wins(self):
        from dashscope.resources.audio.realtime import RealtimeSession

        client = _make_client(_ok, user_agent_suffix="dashscope-cli")
        try:
            session = RealtimeSession(
                client,
                model="qwen-omni-turbo-realtime",
                ws_url="wss://ws.test.local/realtime",
                headers={"User-Agent": "mine"},
            )
            agents = [
                value
                for key, value in session._headers.items()
                if key.lower() == "user-agent"
            ]
            assert agents == ["mine"]
        finally:
            client.close()
