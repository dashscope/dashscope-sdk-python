# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the dev mock (dashscope.devkit.mock).

Every test asserts one behavior of the static/replay/record modes at the
transport choke point. The mock is process-global, so an autouse fixture
resets code + env activation around each case.
"""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import dashscope
from dashscope import AsyncClient, Client, devkit
from dashscope.devkit import DevMockMiss
from dashscope.devkit import mock as m
from dashscope.errors import DashScopeError

BASE = "https://api.test.local/api/v1"
DEAD = "https://dead.invalid/api/v1"

NATIVE_BODY = {
    "output": {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "REAL"},
            },
        ],
    },
    "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    "request_id": "real-1",
}
NATIVE_SSE = (
    "id: 1\ndata: "
    + json.dumps(
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "REAL-STREAM",
                        },
                    },
                ],
            },
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
            },
            "request_id": "real-1",
        },
    )
    + "\n\n"
)


@pytest.fixture(autouse=True)
def _reset_mock(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_DEV_MOCK", raising=False)
    monkeypatch.delenv("DASHSCOPE_DEV_MOCK_FIXTURE", raising=False)
    devkit.disable_dev_mock()
    yield
    devkit.disable_dev_mock()


def _is_stream(body: dict) -> bool:
    return bool(body.get("parameters", {}).get("incremental_output"))


def _real_handler(request):
    body = json.loads(request.content)
    if _is_stream(body):
        return httpx.Response(
            200,
            content=NATIVE_SSE,
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(200, json=NATIVE_BODY)


def _boom(request):
    raise AssertionError("network hit while the mock should serve it")


def _client(handler=None, **kwargs):
    if handler is None:
        kwargs.setdefault("base_url", DEAD)
        return Client(api_key="sk-test", **kwargs)
    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=BASE,
    )
    kwargs.setdefault("base_url", BASE)
    return Client(api_key="sk-secret", http_client=http_client, **kwargs)


def _async_client(handler):
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=BASE,
    )
    return AsyncClient(
        api_key="sk-secret",
        http_client=http_client,
        base_url=BASE,
    )


_MSG = [{"role": "user", "content": "hi"}]


# --- static ---------------------------------------------------------------


def test_static_nonstream_needs_no_network():
    dashscope.enable_dev_mock("static")
    resp = _client().chat.completions.create(model="qwen-plus", messages=_MSG)
    assert resp.output_text.startswith(devkit.MOCK_MARKER)


def test_static_nonstream_request_id_is_marked():
    dashscope.enable_dev_mock("static")
    resp = _client().chat.completions.create(model="qwen-plus", messages=_MSG)
    assert resp.request_id.startswith("mock-")


def test_static_stream_reassembles_and_reports_usage():
    dashscope.enable_dev_mock("static")
    stream = _client().chat.completions.create(
        model="qwen-plus",
        messages=_MSG,
        stream=True,
    )
    assert "".join(stream.text_stream).startswith(devkit.MOCK_MARKER)
    assert stream.get_final_message().usage is not None


def test_static_openai_protocol():
    dashscope.enable_dev_mock("static")
    client = _client(protocol="openai", base_url="https://dead.invalid/v1")
    resp = client.chat.completions.create(model="qwen-plus", messages=_MSG)
    assert resp.output_text.startswith(devkit.MOCK_MARKER)


def test_static_anthropic_protocol_stream():
    dashscope.enable_dev_mock("static")
    client = _client(protocol="anthropic", base_url="https://dead.invalid")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=_MSG,
        stream=True,
    )
    assert "".join(stream.text_stream).startswith(devkit.MOCK_MARKER)


def test_static_multimodal_path_is_supported():
    dashscope.enable_dev_mock("static")
    resp = _client().multimodal.chat.create(
        model="qwen-vl",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )
    assert resp.output_text.startswith(devkit.MOCK_MARKER)


def test_static_refuses_non_chat_endpoint_loudly():
    dashscope.enable_dev_mock("static")
    with pytest.raises(DevMockMiss):
        _client().embeddings.create(model="text-embedding-v4", input=["hi"])


# --- env activation -------------------------------------------------------


def test_env_var_activates_static(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_DEV_MOCK", "static")
    assert dashscope.dev_mock_active() is True
    resp = _client().chat.completions.create(model="qwen-plus", messages=_MSG)
    assert resp.output_text.startswith(devkit.MOCK_MARKER)


def test_invalid_env_mode_raises(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_DEV_MOCK", "nope")
    with pytest.raises(ValueError):
        dashscope.dev_mock_active()


def test_disable_restores_passthrough():
    dashscope.enable_dev_mock("static")
    dashscope.disable_dev_mock()
    assert dashscope.dev_mock_active() is False
    with pytest.raises(AssertionError):
        _client(_boom).chat.completions.create(
            model="qwen-plus",
            messages=_MSG,
        )


# --- record / replay ------------------------------------------------------


def _read_fixture(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_record_writes_fixture_without_credentials(tmp_path):
    fix = str(tmp_path / "mock.jsonl")
    dashscope.enable_dev_mock("record", fix)
    resp = _client(_real_handler).chat.completions.create(
        model="qwen-plus",
        messages=_MSG,
    )
    dashscope.disable_dev_mock()
    assert resp.output_text == "REAL"
    with open(fix, encoding="utf-8") as f:
        raw = f.read()
    assert "sk-secret" not in raw
    assert "uthorization" not in raw
    assert len(_read_fixture(fix)) == 1


def test_record_then_replay_serves_from_fixture(tmp_path):
    fix = str(tmp_path / "mock.jsonl")
    dashscope.enable_dev_mock("record", fix)
    _client(_real_handler).chat.completions.create(
        model="qwen-plus",
        messages=_MSG,
    )
    dashscope.disable_dev_mock()

    dashscope.enable_dev_mock("replay", fix)
    resp = _client(_boom).chat.completions.create(
        model="qwen-plus",
        messages=_MSG,
    )
    dashscope.disable_dev_mock()
    assert resp.output_text == "REAL"


def test_replay_stream_roundtrip(tmp_path):
    fix = str(tmp_path / "mock.jsonl")
    dashscope.enable_dev_mock("record", fix)
    recorded = "".join(
        _client(_real_handler)
        .chat.completions.create(model="qwen-plus", messages=_MSG, stream=True)
        .text_stream,
    )
    dashscope.disable_dev_mock()

    dashscope.enable_dev_mock("replay", fix)
    replayed = "".join(
        _client(_boom)
        .chat.completions.create(model="qwen-plus", messages=_MSG, stream=True)
        .text_stream,
    )
    dashscope.disable_dev_mock()
    assert replayed == recorded == "REAL-STREAM"


def test_replay_miss_raises_dev_mock_miss(tmp_path):
    fix = str(tmp_path / "mock.jsonl")
    dashscope.enable_dev_mock("record", fix)
    _client(_real_handler).chat.completions.create(
        model="qwen-plus",
        messages=_MSG,
    )
    dashscope.disable_dev_mock()

    dashscope.enable_dev_mock("replay", fix)
    with pytest.raises(DevMockMiss):
        _client(_boom).chat.completions.create(
            model="different-model",
            messages=_MSG,
        )
    dashscope.disable_dev_mock()


def test_dev_mock_miss_is_a_dashscope_error():
    assert issubclass(DevMockMiss, DashScopeError)


# --- record hardening -----------------------------------------------------


def test_record_fixture_is_owner_only(tmp_path):
    fix = tmp_path / "mock.jsonl"
    dashscope.enable_dev_mock("record", str(fix))
    _client(_real_handler).chat.completions.create(
        model="qwen-plus",
        messages=_MSG,
    )
    dashscope.disable_dev_mock()
    # Recorded bodies can be sensitive; the file must not be group/other
    # readable.
    assert fix.stat().st_mode & 0o777 == 0o600


def test_record_mode_warns_once(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_warned_record", False)
    fix = str(tmp_path / "mock.jsonl")
    dashscope.enable_dev_mock("record", fix)
    with pytest.warns(UserWarning, match="record"):
        _client(_real_handler).chat.completions.create(
            model="qwen-plus",
            messages=_MSG,
        )
    dashscope.disable_dev_mock()


def test_re_record_newest_wins(tmp_path):
    fix = str(tmp_path / "mock.jsonl")
    dashscope.enable_dev_mock("record", fix)
    _client(_real_handler).chat.completions.create(
        model="qwen-plus",
        messages=_MSG,
    )
    dashscope.disable_dev_mock()

    def handler2(request):
        body = dict(NATIVE_BODY)
        body["output"] = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "REAL2"},
                },
            ],
        }
        return httpx.Response(200, json=body)

    dashscope.enable_dev_mock("record", fix)
    _client(handler2).chat.completions.create(
        model="qwen-plus",
        messages=_MSG,
    )
    dashscope.disable_dev_mock()

    # Same request re-recorded: replay must serve the newest capture,
    # not the stale first one.
    dashscope.enable_dev_mock("replay", fix)
    resp = _client(_boom).chat.completions.create(
        model="qwen-plus",
        messages=_MSG,
    )
    dashscope.disable_dev_mock()
    assert resp.output_text == "REAL2"


def test_replay_misses_when_endpoint_differs(tmp_path):
    fix = str(tmp_path / "mock.jsonl")
    dashscope.enable_dev_mock("record", fix)
    _client(_real_handler).chat.completions.create(
        model="qwen-plus",
        messages=_MSG,
    )
    dashscope.disable_dev_mock()

    # Same path + body but a different endpoint (region): the capture
    # must not replay, or a Hangzhou recording would answer a Singapore
    # request.
    other = "https://api.other-region.test.local/api/v1"
    dashscope.enable_dev_mock("replay", fix)
    http_client = httpx.Client(
        transport=httpx.MockTransport(_boom),
        base_url=other,
    )
    client = Client(
        api_key="sk-secret",
        http_client=http_client,
        base_url=other,
    )
    with pytest.raises(DevMockMiss):
        client.chat.completions.create(model="qwen-plus", messages=_MSG)
    dashscope.disable_dev_mock()


def test_redactor_scrubs_recorded_body(tmp_path):
    fix = str(tmp_path / "mock.jsonl")
    devkit.set_redactor(lambda body: {**body, "output": "<redacted>"})
    try:
        dashscope.enable_dev_mock("record", fix)
        _client(_real_handler).chat.completions.create(
            model="qwen-plus",
            messages=_MSG,
        )
        dashscope.disable_dev_mock()
    finally:
        devkit.set_redactor(None)
    with open(fix, encoding="utf-8") as f:
        raw = f.read()
    assert "REAL" not in raw
    assert "<redacted>" in raw


# --- fixture encoding / stream completeness (unit level) ------------------


def test_binary_body_roundtrips_losslessly():
    raw = bytes(range(256))  # not valid UTF-8
    encoded = m._encode_body(raw)
    assert encoded["body_encoding"] == "base64"
    assert m._decode_body(encoded) == raw


def test_text_body_stays_readable():
    encoded = m._encode_body(b'{"ok": true}')
    assert encoded["body_encoding"] == "utf-8"
    assert encoded["body"] == '{"ok": true}'
    assert m._decode_body(encoded) == b'{"ok": true}'


def test_stream_drained_to_eof_is_complete():
    class Inner(httpx.SyncByteStream):
        def __iter__(self):
            yield b"a"
            yield b"b"

    seen = {}
    tee = m._TeeSyncStream(
        Inner(),
        lambda raw, complete: seen.update(raw=raw, complete=complete),
    )
    assert list(tee) == [b"a", b"b"]
    assert seen["complete"] is True
    assert seen["raw"] == b"ab"


def test_stream_closed_early_is_incomplete():
    class Inner(httpx.SyncByteStream):
        def __iter__(self):
            yield b"a"
            yield b"b"

    seen = {}
    tee = m._TeeSyncStream(
        Inner(),
        lambda raw, complete: seen.update(raw=raw, complete=complete),
    )
    iterator = iter(tee)
    next(iterator)  # consume one chunk, then abandon
    tee.close()
    assert seen["complete"] is False
    assert seen["raw"] == b"a"


def test_replay_refuses_incomplete_stream(tmp_path, monkeypatch):
    fix = str(tmp_path / "mock.jsonl")
    monkeypatch.setenv("DASHSCOPE_DEV_MOCK_FIXTURE", fix)
    key = m._fingerprint(
        "POST",
        "/x",
        {"model": "qwen-plus"},
        None,
        base_url="https://b/api/v1",
    )
    m._append_fixture(
        {
            "key": key,
            "stream": True,
            "complete": False,
            "status_code": 200,
            "content_type": "text/event-stream",
            "body": "data: {}\n\n",
            "body_encoding": "utf-8",
        },
    )
    with pytest.raises(DevMockMiss):
        m._replay(key, "POST", "https://b/api/v1/x", True)
    monkeypatch.setenv("DASHSCOPE_DEV_MOCK_ALLOW_PARTIAL", "1")
    assert (
        m._replay(key, "POST", "https://b/api/v1/x", True).status_code == 200
    )


def test_fingerprint_separates_body_channels():
    base = m._fingerprint("POST", "/x", {"a": 1}, None)
    assert base != m._fingerprint("POST", "/x", {"a": 1}, None, content=b"z")
    assert base != m._fingerprint(
        "POST",
        "/x",
        {"a": 1},
        None,
        base_url="https://other/api/v1",
    )
    # Volatile identity headers never change the key.
    assert base == m._fingerprint(
        "POST",
        "/x",
        {"a": 1},
        None,
        headers={"Authorization": "Bearer x", "User-Agent": "y"},
    )


# --- async ----------------------------------------------------------------


def test_async_record_then_replay(tmp_path):
    fix = str(tmp_path / "mock.jsonl")

    async def run():
        dashscope.enable_dev_mock("record", fix)
        client = _async_client(_real_handler)
        recorded = (
            await client.chat.completions.create(
                model="qwen-plus",
                messages=_MSG,
            )
        ).output_text
        await client.aclose()
        dashscope.disable_dev_mock()

        dashscope.enable_dev_mock("replay", fix)
        client = _async_client(_boom)
        replayed = (
            await client.chat.completions.create(
                model="qwen-plus",
                messages=_MSG,
            )
        ).output_text
        await client.aclose()
        dashscope.disable_dev_mock()
        return recorded, replayed

    recorded, replayed = asyncio.run(run())
    assert recorded == replayed == "REAL"


def test_async_static_needs_no_network():
    async def run():
        dashscope.enable_dev_mock("static")
        client = AsyncClient(api_key="sk-test", base_url=DEAD)
        resp = await client.chat.completions.create(
            model="qwen-plus",
            messages=_MSG,
        )
        await client.aclose()
        return resp.output_text

    assert asyncio.run(run()).startswith(devkit.MOCK_MARKER)
