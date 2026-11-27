# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.responses (C1: create/get/delete/input_items + stream).

Same compatible-mode wire as batches (/compatible-mode/v1/responses,
protocol-independent, flat OpenAI response object), so these mirror
test_batches.py conventions. C1-specific coverage: the ten-type output
array and output_text convenience, two-generation usage + tool_usage,
Bailian extension normalization (reasoning/prompt_cache_options/ocr_options/
text/tools), session_cache header, the previous_response_id ⊕ conversation
and function_call_output adjacency pre-validations, enable_thinking
deprecation, the input_items cursor (created_at milliseconds, last_id),
SSE streaming (text_stream / get_final_response) and raw/stream overloads.
"""

# Mock handlers intentionally ignore the request argument; pytest fixture
# injection looks like unused/redefined names to pylint.
# pylint: disable=unused-argument,redefined-outer-name

from __future__ import annotations

import json

import httpx
import pytest

import dashscope.resources.responses as responses_mod
from dashscope import AsyncClient, Client
from dashscope.errors import BadRequestError, DashScopeError
from dashscope.resources.responses import InputItemList
from dashscope.transport._ident import sdk_client_value
from dashscope.types.responses import (
    FunctionTool,
    OcrOptions,
    Reasoning,
    Response,
    ResponseDeleted,
    WebSearchTool,
)

BASE_URL = "https://api.test.local/api/v1"
_WS_HOST = "https://ws-123.cn-beijing.maas.aliyuncs.com"
_COMPAT_RESPONSES = "/compatible-mode/v1/responses"

_RESPONSE_BODY = {
    "id": "resp_abc123",
    "object": "response",
    "created_at": 1771165900,
    "status": "completed",
    "model": "qwen-plus",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "Hello", "annotations": []},
            ],
        },
    ],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
    "tool_usage": {},
    "error": None,
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
    "store": True,
}

_INPUT_ITEMS_BODY = {
    "created_at": 1778674066000,
    "data": [
        {
            "id": "msg_1",
            "type": "message",
            "role": "user",
            "status": "completed",
            "content": [{"type": "input_text", "text": "hi"}],
        },
    ],
    "first_id": "msg_1",
    "last_id": "msg_1",
    "has_more": False,
    "id": "resp_abc123",
    "model": "qwen-plus",
}


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


def _sse(frames):
    return "".join(
        f"id: {i}\ndata: {json.dumps(f)}\n\n" for i, f in enumerate(frames, 1)
    )


@pytest.fixture
def clean_env(monkeypatch):
    from dashscope._config import settings

    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_HTTP_BASE_URL",
        "DASHSCOPE_WORKSPACE",
        "DASHSCOPE_REGION",
        "DASHSCOPE_PROTOCOL",
        "DASHSCOPE_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)
    settings.clear()
    yield monkeypatch
    settings.clear()


# --------------------------------------------------------------------------
# create
# --------------------------------------------------------------------------


def test_create_wire_body_and_parse():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_RESPONSE_BODY)

    client = _make_client(handler)
    resp = client.responses.create(model="qwen-plus", input="hi")
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v1/responses"
    # Unset optionals stay off the wire; stream defaults to False.
    assert captured["body"] == {
        "model": "qwen-plus",
        "input": "hi",
        "stream": False,
    }
    assert isinstance(resp, Response)
    assert resp.id == "resp_abc123"
    assert resp.object == "response"
    assert resp.status == "completed"
    assert resp.output_text == "Hello"
    assert resp.output[0].type == "message"
    assert resp.usage.total_tokens == 15
    assert resp.usage.input_tokens_details.cached_tokens == 0
    assert resp.tool_usage is not None
    assert resp.is_error is False
    client.close()


def test_create_normalizes_extensions_and_tools():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_RESPONSE_BODY)

    client = _make_client(handler)
    client.responses.create(
        model="qwen-plus",
        input=[{"role": "user", "content": "hi"}],
        instructions="be brief",
        store=False,
        tools=[
            WebSearchTool(),
            FunctionTool(name="get_weather", parameters={"type": "object"}),
            {"type": "code_interpreter"},
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=1024,
        reasoning=Reasoning(effort="high", summary="auto"),
        prompt_cache_options={"mode": "explicit", "ttl": "1h"},
        ocr_options=OcrOptions(task="document_parsing"),
        text={"format": {"type": "json_schema", "name": "out"}},
    )
    body = captured["body"]
    assert body["input"] == [{"role": "user", "content": "hi"}]
    assert body["instructions"] == "be brief"
    assert body["store"] is False
    assert body["tools"] == [
        {"type": "web_search"},
        {
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object"},
        },
        {"type": "code_interpreter"},
    ]
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "1h"}
    assert body["ocr_options"] == {"task": "document_parsing"}
    assert body["text"] == {"format": {"type": "json_schema", "name": "out"}}
    assert body["temperature"] == 0.7
    assert body["max_output_tokens"] == 1024
    client.close()


def test_create_session_cache_header():
    seen = []

    def handler(request):
        seen.append(request.headers.get("x-dashscope-session-cache"))
        return httpx.Response(200, json=_RESPONSE_BODY)

    client = _make_client(handler)
    client.responses.create(model="qwen-plus", input="hi", session_cache=True)
    client.responses.create(model="qwen-plus", input="hi", session_cache=False)
    client.responses.create(model="qwen-plus", input="hi")
    assert seen == ["enable", "disable", None]
    client.close()


def test_create_previous_response_id_and_conversation_exclusive():
    def handler(request):
        return httpx.Response(200, json=_RESPONSE_BODY)

    client = _make_client(handler)
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.responses.create(
            model="qwen-plus",
            input="hi",
            previous_response_id="resp_1",
            conversation="conv_1",
        )
    # Either alone is fine.
    client.responses.create(
        model="qwen-plus",
        input="hi",
        previous_response_id="resp_1",
    )
    client.responses.create(
        model="qwen-plus",
        input="hi",
        conversation="conv_1",
    )
    client.close()


def test_create_function_call_output_adjacency():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_RESPONSE_BODY)

    client = _make_client(handler)
    good = [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "sun"},
    ]
    client.responses.create(model="qwen-plus", input=good)
    assert captured["body"]["input"] == good

    orphan = [
        {"type": "function_call_output", "call_id": "call_1", "output": "sun"},
    ]
    with pytest.raises(ValueError, match="function_call_output"):
        client.responses.create(model="qwen-plus", input=orphan)

    mismatch = [
        {"type": "function_call", "call_id": "call_1", "name": "f"},
        {"type": "function_call_output", "call_id": "call_2", "output": "x"},
    ]
    with pytest.raises(ValueError, match="function_call_output"):
        client.responses.create(model="qwen-plus", input=mismatch)
    client.close()


def test_create_enable_thinking_deprecation():
    def handler(request):
        return httpx.Response(200, json=_RESPONSE_BODY)

    responses_mod._ENABLE_THINKING_WARNED = False
    client = _make_client(handler)
    with pytest.warns(DeprecationWarning, match="enable_thinking"):
        client.responses.create(
            model="qwen-plus",
            input="hi",
            enable_thinking=True,
        )
    client.close()


def test_create_requires_model_and_input():
    def handler(request):
        return httpx.Response(200, json=_RESPONSE_BODY)

    client = _make_client(handler)
    with pytest.raises(ValueError, match="model"):
        client.responses.create(model="", input="hi")
    with pytest.raises(ValueError, match="input"):
        client.responses.create(model="qwen-plus", input=None)
    client.close()


# --------------------------------------------------------------------------
# get / delete
# --------------------------------------------------------------------------


def test_get_parses_response():
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/responses/resp_abc123"
        return httpx.Response(
            200,
            json=_RESPONSE_BODY,
            headers={"x-request-id": "rid-1"},
        )

    client = _make_client(handler)
    resp = client.responses.get("resp_abc123")
    assert resp.id == "resp_abc123"
    assert resp.request_id == "rid-1"
    assert resp.headers["x-request-id"] == "rid-1"
    # A Response object is accepted as the id too.
    again = client.responses.get(resp)
    assert again.id == "resp_abc123"
    client.close()


def test_get_requires_non_empty_id():
    def handler(request):
        return httpx.Response(200, json=_RESPONSE_BODY)

    client = _make_client(handler)
    with pytest.raises(ValueError, match="response_id"):
        client.responses.get("  ")
    client.close()


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_response_id_rejected_before_wire(bad):
    def handler(request):
        raise AssertionError("request must not reach the wire")

    client = _make_client(handler)
    for call in (
        client.responses.get,
        client.responses.delete,
        client.responses.input_items.list,
    ):
        with pytest.raises(ValueError, match="response_id"):
            call(bad)
    client.close()


def test_response_id_percent_encoded_in_path():
    seen = []

    def handler(request):
        seen.append(request.url.raw_path)
        if request.method == "DELETE":
            return httpx.Response(
                200,
                json={"id": "resp/a?b#c", "deleted": True},
            )
        if request.url.path.endswith("/input_items"):
            return httpx.Response(200, json=_INPUT_ITEMS_BODY)
        return httpx.Response(200, json=_RESPONSE_BODY)

    client = _make_client(handler)
    tricky = "resp/a?b#c"
    client.responses.get(tricky)
    client.responses.delete(tricky)
    client.responses.input_items.list(tricky)
    # '/', '?' and '#' stay inside one path segment instead of retargeting
    # the request to another path or query.
    encoded = b"/api/v1/responses/resp%2Fa%3Fb%23c"
    assert seen == [encoded, encoded, encoded + b"/input_items"]
    client.close()


def test_delete_returns_id_and_deleted_no_object():
    def handler(request):
        assert request.method == "DELETE"
        assert request.url.path == "/api/v1/responses/resp_abc123"
        return httpx.Response(
            200,
            json={"id": "resp_abc123", "deleted": True},
        )

    client = _make_client(handler)
    resp = client.responses.delete("resp_abc123")
    assert isinstance(resp, ResponseDeleted)
    assert resp.id == "resp_abc123"
    assert resp.deleted is True
    client.close()


def test_http_error_maps_openai_nested_body():
    def handler(request):
        return httpx.Response(
            400,
            json={"error": {"code": "InvalidParameter", "message": "bad"}},
            headers={"x-request-id": "rid-err"},
        )

    client = _make_client(handler)
    with pytest.raises(BadRequestError) as ei:
        client.responses.get("resp_x")
    assert ei.value.code == "InvalidParameter"
    assert ei.value.status_code == 400
    assert ei.value.request_id == "rid-err"
    client.close()


def test_error_envelope_when_not_raising():
    def handler(request):
        return httpx.Response(
            404,
            json={"error": {"code": "NotFound", "message": "gone"}},
            headers={"x-request-id": "rid-404"},
        )

    client = _make_client(handler, raise_on_error=False)
    resp = client.responses.get("resp_typo")
    assert resp.is_error is True
    assert resp.status_code == 404
    assert resp.code == "NotFound"
    assert resp.message == "gone"
    assert resp.request_id == "rid-404"
    assert resp.id is None
    client.close()


def test_business_error_layer_raise_and_report():
    body = {
        "code": "InvalidParameter",
        "message": "unsupported",
        "request_id": "rid-biz",
    }

    def handler(request):
        return httpx.Response(200, json=body)

    client = _make_client(handler)
    with pytest.raises(BadRequestError) as ei:
        client.responses.get("resp_1")
    assert ei.value.status_code == 200
    client.close()

    client = _make_client(handler, raise_on_error=False)
    resp = client.responses.get("resp_1")
    assert resp.is_error is True
    assert resp.code == "InvalidParameter"
    client.close()


# --------------------------------------------------------------------------
# input_items.list
# --------------------------------------------------------------------------


def test_input_items_parse_and_created_at_milliseconds():
    def handler(request):
        assert request.url.path == "/api/v1/responses/resp_abc123/input_items"
        return httpx.Response(200, json=_INPUT_ITEMS_BODY)

    client = _make_client(handler)
    page = client.responses.input_items.list("resp_abc123")
    assert isinstance(page, InputItemList)
    assert page.id == "resp_abc123"
    assert page.model == "qwen-plus"
    # Milliseconds, not seconds (13 digits).
    assert page.created_at == 1778674066000
    assert page.first_id == "msg_1"
    assert page.last_id == "msg_1"
    assert [i.id for i in page.items] == ["msg_1"]
    assert page.items[0].content[0].type == "input_text"
    assert page.has_next_page() is False
    client.close()


def test_input_items_cursor_pagination_uses_last_id():
    seen = []

    def handler(request):
        after = request.url.params.get("after")
        seen.append(after)
        if after is None:
            return httpx.Response(
                200,
                json={
                    "data": [{"id": "msg_1"}, {"id": "msg_2"}],
                    "first_id": "msg_1",
                    "last_id": "msg_2",
                    "has_more": True,
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [{"id": "msg_3"}],
                "first_id": "msg_3",
                "last_id": "msg_3",
                "has_more": False,
            },
        )

    client = _make_client(handler)
    page = client.responses.input_items.list("resp_1", limit=2, order="asc")
    pages = list(page.iter_pages())
    assert [[i.id for i in p] for p in pages] == [
        ["msg_1", "msg_2"],
        ["msg_3"],
    ]
    assert seen == [None, "msg_2"]
    client.close()


def test_input_items_params_validation():
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"data": [], "has_more": False})

    client = _make_client(handler)
    for bad_limit in (0, 101):
        with pytest.raises(ValueError, match="limit"):
            client.responses.input_items.list("resp_1", limit=bad_limit)
    with pytest.raises(ValueError, match="order"):
        client.responses.input_items.list("resp_1", order="sideways")
    client.responses.input_items.list("resp_1")
    assert captured["params"] == {}
    client.responses.input_items.list(
        "resp_1",
        after="msg_9",
        limit=100,
        order="desc",
    )
    assert captured["params"] == {
        "after": "msg_9",
        "limit": "100",
        "order": "desc",
    }
    client.close()


def test_input_items_error_page_does_not_bind_pagination():
    def handler(request):
        return httpx.Response(
            400,
            json={"error": {"code": "InvalidParameter", "message": "bad"}},
        )

    client = _make_client(handler, raise_on_error=False)
    page = client.responses.input_items.list("resp_1")
    assert page.is_error is True
    assert page.has_next_page() is False
    assert page.next_page() is None
    client.close()


# --------------------------------------------------------------------------
# header / routing / raw
# --------------------------------------------------------------------------


def test_sdk_module_header():
    seen = {}

    def handler(request):
        seen["client"] = request.headers.get("x-dashscope-sdk-client")
        return httpx.Response(200, json=_RESPONSE_BODY)

    client = _make_client(handler)
    client.responses.get("resp_1")
    assert seen["client"] == sdk_client_value("responses")
    assert seen["client"].endswith("/responses2")
    client.close()


@pytest.mark.parametrize("protocol", ["native", "openai", "anthropic"])
def test_routing_default_gateway_ignores_chat_protocol(clean_env, protocol):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_RESPONSE_BODY)

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=_WS_HOST + "/api/v1",
    )
    client = Client(
        api_key="sk-test",
        workspace="ws-123",
        protocol=protocol,
        http_client=http_client,
    )
    client.responses.get("resp_1")
    assert urls == [_WS_HOST + _COMPAT_RESPONSES + "/resp_1"], protocol
    client.close()


def test_routing_explicit_base_url():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_RESPONSE_BODY)

    client = _make_client(handler)
    client.responses.create(model="qwen-plus", input="hi")
    assert urls == [BASE_URL + "/responses"]
    client.close()


def test_raw_returns_apiresponse_before_error_check():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad"}})

    client = _make_client(handler)
    raw = client.responses.get("resp_1", raw=True)
    assert raw.status_code == 400
    assert raw.json()["error"]["message"] == "bad"
    client.close()


def test_raw_with_stream_raises():
    def handler(request):
        return httpx.Response(200, json=_RESPONSE_BODY)

    client = _make_client(handler)
    with pytest.raises(ValueError, match="raw=True"):
        client.responses.create(
            model="qwen-plus",
            input="hi",
            stream=True,
            raw=True,
        )
    client.close()


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------

_STREAM_FRAMES = [
    {
        "type": "response.created",
        "sequence_number": 0,
        "response": {"id": "resp_s", "object": "response", "status": "queued"},
    },
    {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "delta": "He",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
    },
    {
        "type": "response.output_text.delta",
        "sequence_number": 2,
        "delta": "llo",
    },
    {
        "type": "response.completed",
        "sequence_number": 3,
        "response": {**_RESPONSE_BODY, "id": "resp_s"},
    },
]


def _stream_handler(request):
    return httpx.Response(
        200,
        content=_sse(_STREAM_FRAMES),
        headers={
            "content-type": "text/event-stream",
            "x-request-id": "rid-stream",
        },
    )


def test_stream_iterates_events_and_text_stream():
    client = _make_client(_stream_handler)
    stream = client.responses.create(
        model="qwen-plus",
        input="hi",
        stream=True,
    )
    events = list(stream)
    assert [e.type for e in events] == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ]
    assert events[0].response.status == "queued"
    assert stream.headers["x-request-id"] == "rid-stream"
    client.close()

    client = _make_client(_stream_handler)
    stream = client.responses.create(
        model="qwen-plus",
        input="hi",
        stream=True,
    )
    assert "".join(stream.text_stream) == "Hello"
    client.close()


def test_stream_get_final_response():
    client = _make_client(_stream_handler)
    stream = client.responses.create(
        model="qwen-plus",
        input="hi",
        stream=True,
    )
    final = stream.get_final_response()
    assert isinstance(final, Response)
    assert final.id == "resp_s"
    assert final.status == "completed"
    assert final.output_text == "Hello"
    assert final.usage.total_tokens == 15
    assert final.headers["x-request-id"] == "rid-stream"
    client.close()


def test_stream_in_stream_error_raises():
    def handler(request):
        frames = [
            _STREAM_FRAMES[0],
            {"error": {"code": "InvalidParameter", "message": "boom"}},
        ]
        return httpx.Response(
            200,
            content=_sse(frames),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    stream = client.responses.create(
        model="qwen-plus",
        input="hi",
        stream=True,
    )
    with pytest.raises(BadRequestError) as ei:
        list(stream)
    assert ei.value.code == "InvalidParameter"
    client.close()


def test_stream_without_terminal_raises():
    def handler(request):
        return httpx.Response(
            200,
            content=_sse(_STREAM_FRAMES[:3]),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    stream = client.responses.create(
        model="qwen-plus",
        input="hi",
        stream=True,
    )
    with pytest.raises(DashScopeError, match="terminal"):
        stream.get_final_response()
    client.close()


# --------------------------------------------------------------------------
# async symmetry
# --------------------------------------------------------------------------


async def test_async_create_get_delete():
    captured = []

    def handler(request):
        captured.append((request.method, request.url.path))
        if request.method == "DELETE":
            return httpx.Response(
                200,
                json={"id": "resp_abc123", "deleted": True},
            )
        return httpx.Response(200, json=_RESPONSE_BODY)

    client = _make_async_client(handler)
    created = await client.responses.create(model="qwen-plus", input="hi")
    assert created.output_text == "Hello"
    got = await client.responses.get("resp_abc123")
    assert got.status == "completed"
    deleted = await client.responses.delete("resp_abc123")
    assert deleted.deleted is True
    assert captured == [
        ("POST", "/api/v1/responses"),
        ("GET", "/api/v1/responses/resp_abc123"),
        ("DELETE", "/api/v1/responses/resp_abc123"),
    ]
    await client.aclose()


async def test_async_input_items_pagination():
    seen = []

    def handler(request):
        after = request.url.params.get("after")
        seen.append(after)
        if after is None:
            return httpx.Response(
                200,
                json={
                    "data": [{"id": "msg_1"}],
                    "last_id": "msg_1",
                    "has_more": True,
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [{"id": "msg_2"}],
                "last_id": "msg_2",
                "has_more": False,
            },
        )

    client = _make_async_client(handler)
    page = await client.responses.input_items.list("resp_1")
    pages = [p async for p in page.iter_pages_async()]
    assert [[i.id for i in p] for p in pages] == [["msg_1"], ["msg_2"]]
    assert seen == [None, "msg_1"]
    await client.aclose()


async def test_async_stream_get_final_response():
    client = _make_async_client(_stream_handler)
    stream = await client.responses.create(
        model="qwen-plus",
        input="hi",
        stream=True,
    )
    final = await stream.get_final_response()
    assert final.id == "resp_s"
    assert final.output_text == "Hello"
    await client.aclose()


async def test_async_error_envelope_when_not_raising():
    def handler(request):
        return httpx.Response(
            404,
            json={"error": {"code": "NotFound", "message": "gone"}},
        )

    client = _make_async_client(handler, raise_on_error=False)
    resp = await client.responses.get("resp_typo")
    assert resp.is_error is True
    assert resp.code == "NotFound"
    await client.aclose()


async def test_async_routing_default_gateway(clean_env):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_RESPONSE_BODY)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=_WS_HOST + "/api/v1",
    )
    client = AsyncClient(
        api_key="sk-test",
        workspace="ws-123",
        http_client=http_client,
    )
    await client.responses.get("resp_1")
    assert urls == [_WS_HOST + _COMPAT_RESPONSES + "/resp_1"]
    await client.aclose()


async def test_async_response_id_percent_encoded_in_path():
    seen = []

    def handler(request):
        seen.append(request.url.raw_path)
        if request.url.path.endswith("/input_items"):
            return httpx.Response(200, json=_INPUT_ITEMS_BODY)
        return httpx.Response(200, json=_RESPONSE_BODY)

    client = _make_async_client(handler)
    await client.responses.get("resp/a?b#c")
    await client.responses.input_items.list("resp/a?b#c")
    encoded = b"/api/v1/responses/resp%2Fa%3Fb%23c"
    assert seen == [encoded, encoded + b"/input_items"]
    await client.aclose()


@pytest.mark.parametrize("bad", ["", "   "])
async def test_async_empty_response_id_rejected_before_wire(bad):
    def handler(request):
        raise AssertionError("request must not reach the wire")

    client = _make_async_client(handler)
    with pytest.raises(ValueError, match="response_id"):
        await client.responses.get(bad)
    with pytest.raises(ValueError, match="response_id"):
        await client.responses.delete(bad)
    with pytest.raises(ValueError, match="response_id"):
        await client.responses.input_items.list(bad)
    await client.aclose()


def _two_page_input_items_handler(seen):
    def handler(request):
        seen.append(request.url.raw_path)
        after = request.url.params.get("after")
        item_id = "msg_2" if after else "msg_1"
        return httpx.Response(
            200,
            json={
                "data": [{"id": item_id}],
                "first_id": item_id,
                "last_id": item_id,
                "has_more": after is None,
            },
        )

    return handler


_TRICKY_RESP = "resp/a?b#c"
_ENCODED_RESP_ITEMS = b"/api/v1/responses/resp%2Fa%3Fb%23c/input_items"


def test_input_items_pagination_does_not_double_encode_response_id():
    seen = []
    client = _make_client(_two_page_input_items_handler(seen))
    page = client.responses.input_items.list(_TRICKY_RESP)
    ids = [item.id for p in page.iter_pages() for item in p]
    assert ids == ["msg_1", "msg_2"]
    # Page 2 must reuse the caller's raw id, not the encoded page-1 value.
    assert seen == [
        _ENCODED_RESP_ITEMS,
        _ENCODED_RESP_ITEMS + b"?after=msg_1",
    ]
    client.close()


@pytest.mark.asyncio
async def test_async_input_items_pagination_does_not_double_encode_response_id():
    seen = []
    client = _make_async_client(_two_page_input_items_handler(seen))
    page = await client.responses.input_items.list(_TRICKY_RESP)
    ids = [item.id async for p in page.iter_pages_async() for item in p]
    assert ids == ["msg_1", "msg_2"]
    assert seen == [
        _ENCODED_RESP_ITEMS,
        _ENCODED_RESP_ITEMS + b"?after=msg_1",
    ]
    await client.aclose()
