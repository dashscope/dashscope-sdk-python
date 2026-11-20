# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.conversations (C2).

The OpenAI-compatible Conversations API: conversation create/get/update/
delete plus items.create/list/get/delete, on the same compatible-mode wire
as Responses (/compatible-mode/v1/conversations). Covers routing (chat
protocol cannot move it), the conversation/item wire shapes, cursor
pagination, metadata/items/limit pre-validation, error envelopes, the
conversations2 SDK header, Responses linkage and async symmetry.

Mock conventions match test_responses.py / test_batches.py.
"""

# Mock handlers intentionally ignore the request argument; pytest fixture
# injection looks like unused/redefined names to pylint.
# pylint: disable=unused-argument,redefined-outer-name

from __future__ import annotations

import json

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.errors import BadRequestError
from dashscope.resources.conversations import ConversationItemList
from dashscope.transport._ident import sdk_client_value
from dashscope.transport._response import APIResponse
from dashscope.types import (
    Conversation,
    ConversationDeleted,
    ConversationItem,
    ConversationItemDeleted,
)

BASE_URL = "https://api.test.local/api/v1"
_WS_HOST = "https://ws-123.cn-beijing.maas.aliyuncs.com"
_COMPAT_CONVERSATIONS = "/compatible-mode/v1/conversations"
_CONV_ID = "conv_abc123"
_ITEM_ID = "msg_abc123"

_CONVERSATION_BODY = {
    "id": _CONV_ID,
    "object": "conversation",
    "created_at": 1771316949128,
    "metadata": {"topic": "demo"},
}
_CONV_DELETED_BODY = {
    "id": _CONV_ID,
    "deleted": True,
    "object": "conversation.deleted",
}
_ITEM_BODY = {
    "id": _ITEM_ID,
    "type": "message",
    "role": "user",
    "status": "completed",
    "content": [{"type": "input_text", "text": "hi"}],
}
_ITEM_DELETED_BODY = {
    "id": _ITEM_ID,
    "deleted": True,
    "object": "conversation.item.deleted",
}
_ITEMS_LIST_BODY = {
    "data": [_ITEM_BODY],
    "first_id": _ITEM_ID,
    "last_id": _ITEM_ID,
    "has_more": False,
    "object": "list",
}
# items.create answers with the same list shape but no ``object`` field.
_ITEMS_CREATE_BODY = {
    "data": [_ITEM_BODY],
    "first_id": _ITEM_ID,
    "last_id": _ITEM_ID,
    "has_more": False,
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


def _capture_handler(captured, body):
    def handler(request):
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        captured["sdk"] = request.headers.get("x-dashscope-sdk-client")
        if request.content:
            captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=body)

    return handler


# --------------------------------------------------------------------------
# conversation CRUD
# --------------------------------------------------------------------------


def test_create_conversation_wire_and_parse():
    captured = {}
    client = _make_client(_capture_handler(captured, _CONVERSATION_BODY))
    conv = client.conversations.create(
        metadata={"topic": "demo"},
        items=[{"type": "message", "role": "system", "content": "hello"}],
    )
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v1/conversations"
    assert captured["body"]["metadata"] == {"topic": "demo"}
    assert captured["body"]["items"] == [
        {"type": "message", "role": "system", "content": "hello"},
    ]
    assert captured["sdk"] == sdk_client_value("conversations")
    assert captured["sdk"].endswith("/conversations2")
    assert isinstance(conv, Conversation)
    assert conv.id == _CONV_ID
    assert conv.object == "conversation"
    # created_at is Unix milliseconds.
    assert conv.created_at == 1771316949128
    assert conv.metadata == {"topic": "demo"}
    assert conv.is_error is False
    client.close()


def test_create_conversation_empty_body():
    captured = {}
    client = _make_client(_capture_handler(captured, _CONVERSATION_BODY))
    client.conversations.create()
    assert captured["body"] == {}
    client.close()


def test_get_conversation_accepts_object_or_id():
    captured = {}
    client = _make_client(_capture_handler(captured, _CONVERSATION_BODY))
    conv = client.conversations.get(_CONV_ID)
    assert captured["method"] == "GET"
    assert captured["path"] == f"/api/v1/conversations/{_CONV_ID}"
    assert conv.id == _CONV_ID
    # A Conversation object is accepted in place of its id.
    client.conversations.get(conv)
    assert captured["path"] == f"/api/v1/conversations/{_CONV_ID}"
    client.close()


def test_update_conversation_replaces_metadata():
    captured = {}
    client = _make_client(_capture_handler(captured, _CONVERSATION_BODY))
    client.conversations.update(_CONV_ID, metadata={"topic": "update"})
    assert captured["method"] == "POST"
    assert captured["path"] == f"/api/v1/conversations/{_CONV_ID}"
    assert captured["body"] == {"metadata": {"topic": "update"}}
    client.close()


def test_update_conversation_metadata_required():
    client = _make_client(_capture_handler({}, _CONVERSATION_BODY))
    with pytest.raises(ValueError, match="metadata is required"):
        client.conversations.update(_CONV_ID, metadata=None)
    client.close()


def test_delete_conversation():
    captured = {}
    client = _make_client(_capture_handler(captured, _CONV_DELETED_BODY))
    result = client.conversations.delete(_CONV_ID)
    assert captured["method"] == "DELETE"
    assert captured["path"] == f"/api/v1/conversations/{_CONV_ID}"
    assert isinstance(result, ConversationDeleted)
    assert result.deleted is True
    assert result.id == _CONV_ID
    assert result.object == "conversation.deleted"
    client.close()


# --------------------------------------------------------------------------
# items
# --------------------------------------------------------------------------


def test_items_create():
    captured = {}
    client = _make_client(_capture_handler(captured, _ITEMS_CREATE_BODY))
    page = client.conversations.items.create(
        _CONV_ID,
        items=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            },
        ],
    )
    assert captured["method"] == "POST"
    assert captured["path"] == f"/api/v1/conversations/{_CONV_ID}/items"
    assert captured["body"]["items"][0]["role"] == "user"
    assert isinstance(page, ConversationItemList)
    assert [i.id for i in page.items] == [_ITEM_ID]
    assert page.first_id == _ITEM_ID
    assert page.last_id == _ITEM_ID
    assert page.has_more is False
    client.close()


def test_items_create_requires_non_empty_items():
    client = _make_client(_capture_handler({}, _ITEMS_CREATE_BODY))
    with pytest.raises(ValueError, match="items must not be empty"):
        client.conversations.items.create(_CONV_ID, items=[])
    with pytest.raises(ValueError, match="items is required"):
        client.conversations.items.create(_CONV_ID, items=None)
    client.close()


def test_items_list_params_and_parse():
    captured = {}
    client = _make_client(_capture_handler(captured, _ITEMS_LIST_BODY))
    page = client.conversations.items.list(
        _CONV_ID,
        after="msg_prev",
        limit=10,
        order="asc",
    )
    assert captured["method"] == "GET"
    assert captured["path"] == f"/api/v1/conversations/{_CONV_ID}/items"
    assert captured["params"] == {
        "after": "msg_prev",
        "limit": "10",
        "order": "asc",
    }
    assert isinstance(page, ConversationItemList)
    assert page.object == "list"
    assert page.items[0].text == "hi"
    client.close()


def test_items_list_cursor_traversal():
    pages = {
        None: {
            "data": [{**_ITEM_BODY, "id": "msg_1"}],
            "first_id": "msg_1",
            "last_id": "msg_1",
            "has_more": True,
            "object": "list",
        },
        "msg_1": {
            "data": [{**_ITEM_BODY, "id": "msg_2"}],
            "first_id": "msg_2",
            "last_id": "msg_2",
            "has_more": False,
            "object": "list",
        },
    }

    def handler(request):
        after = request.url.params.get("after")
        return httpx.Response(200, json=pages[after])

    client = _make_client(handler)
    first = client.conversations.items.list(_CONV_ID)
    assert first.has_next_page() is True
    ids = [item.id for page in first.iter_pages() for item in page]
    assert ids == ["msg_1", "msg_2"]
    client.close()


def test_items_get_and_text():
    captured = {}
    client = _make_client(_capture_handler(captured, _ITEM_BODY))
    item = client.conversations.items.get(_CONV_ID, _ITEM_ID)
    assert captured["method"] == "GET"
    assert captured["path"] == (
        f"/api/v1/conversations/{_CONV_ID}/items/{_ITEM_ID}"
    )
    assert isinstance(item, ConversationItem)
    assert item.role == "user"
    assert item.status == "completed"
    assert item.text == "hi"
    client.close()


def test_item_text_from_string_content():
    body = {**_ITEM_BODY, "content": "plain string"}
    client = _make_client(_capture_handler({}, body))
    item = client.conversations.items.get(_CONV_ID, _ITEM_ID)
    assert item.content == "plain string"
    assert item.text == "plain string"
    client.close()


def test_items_delete():
    captured = {}
    client = _make_client(_capture_handler(captured, _ITEM_DELETED_BODY))
    result = client.conversations.items.delete(_CONV_ID, _ITEM_ID)
    assert captured["method"] == "DELETE"
    assert captured["path"] == (
        f"/api/v1/conversations/{_CONV_ID}/items/{_ITEM_ID}"
    )
    assert isinstance(result, ConversationItemDeleted)
    assert result.deleted is True
    assert result.object == "conversation.item.deleted"
    client.close()


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", ["native", "openai", "anthropic"])
def test_routing_default_gateway_ignores_chat_protocol(clean_env, protocol):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_CONVERSATION_BODY)

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
    client.conversations.get(_CONV_ID)
    assert urls == [
        _WS_HOST + _COMPAT_CONVERSATIONS + f"/{_CONV_ID}",
    ], protocol
    client.close()


def test_routing_explicit_base_url():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_CONVERSATION_BODY)

    client = _make_client(handler)
    client.conversations.create()
    assert urls == [BASE_URL + "/conversations"]
    client.close()


# --------------------------------------------------------------------------
# pre-validation
# --------------------------------------------------------------------------


def test_conversation_id_empty_raises():
    client = _make_client(_capture_handler({}, _CONVERSATION_BODY))
    with pytest.raises(ValueError, match="conversation_id must be"):
        client.conversations.get("  ")
    client.close()


def test_item_id_empty_raises():
    client = _make_client(_capture_handler({}, _ITEM_BODY))
    with pytest.raises(ValueError, match="item_id must be"):
        client.conversations.items.get(_CONV_ID, "")
    client.close()


@pytest.mark.parametrize("limit", [0, 101, -1])
def test_items_limit_validation(limit):
    client = _make_client(_capture_handler({}, _ITEMS_LIST_BODY))
    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        client.conversations.items.list(_CONV_ID, limit=limit)
    client.close()


def test_items_order_validation():
    client = _make_client(_capture_handler({}, _ITEMS_LIST_BODY))
    with pytest.raises(ValueError, match="order must be"):
        client.conversations.items.list(_CONV_ID, order="sideways")
    client.close()


def test_metadata_pair_limit():
    client = _make_client(_capture_handler({}, _CONVERSATION_BODY))
    too_many = {f"k{i}": "v" for i in range(17)}
    with pytest.raises(ValueError, match="at most 16"):
        client.conversations.create(metadata=too_many)
    client.close()


def test_metadata_key_value_length_limits():
    client = _make_client(_capture_handler({}, _CONVERSATION_BODY))
    with pytest.raises(ValueError, match="keys must be strings"):
        client.conversations.create(metadata={"k" * 65: "v"})
    with pytest.raises(ValueError, match="values must be strings"):
        client.conversations.create(metadata={"k": "v" * 513})
    client.close()


def test_items_max_20():
    client = _make_client(_capture_handler({}, _ITEMS_CREATE_BODY))
    items = [
        {"type": "message", "role": "user", "content": "x"} for _ in range(21)
    ]
    with pytest.raises(ValueError, match="at most 20"):
        client.conversations.items.create(_CONV_ID, items=items)
    client.close()


# --------------------------------------------------------------------------
# errors / raw
# --------------------------------------------------------------------------


def test_http_error_raises():
    def handler(request):
        return httpx.Response(
            400,
            json={"error": {"code": "InvalidParameter", "message": "bad"}},
            headers={"x-request-id": "rid-err"},
        )

    client = _make_client(handler)
    with pytest.raises(BadRequestError) as ei:
        client.conversations.get(_CONV_ID)
    assert ei.value.code == "InvalidParameter"
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
    conv = client.conversations.get(_CONV_ID)
    assert conv.is_error is True
    assert conv.status_code == 404
    assert conv.code == "NotFound"
    assert conv.message == "gone"
    assert conv.request_id == "rid-404"
    client.close()


def test_business_error_layer():
    body = {
        "code": "InvalidParameter",
        "message": "unsupported",
        "request_id": "rid-biz",
    }

    def handler(request):
        return httpx.Response(200, json=body)

    client = _make_client(handler)
    with pytest.raises(BadRequestError):
        client.conversations.create()
    client.close()

    client = _make_client(handler, raise_on_error=False)
    conv = client.conversations.create()
    assert conv.is_error is True
    assert conv.code == "InvalidParameter"
    client.close()


def test_raw_returns_apiresponse_before_error_check():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad"}})

    client = _make_client(handler)
    raw = client.conversations.get(_CONV_ID, raw=True)
    assert isinstance(raw, APIResponse)
    assert raw.http_response.status_code == 400
    assert raw.json()["error"]["message"] == "bad"
    client.close()


def test_items_list_error_not_paginated():
    def handler(request):
        return httpx.Response(
            404,
            json={"error": {"code": "NotFound", "message": "gone"}},
        )

    client = _make_client(handler, raise_on_error=False)
    page = client.conversations.items.list(_CONV_ID)
    assert page.is_error is True
    assert page.has_next_page() is False
    client.close()


# --------------------------------------------------------------------------
# Responses linkage (C1 + C2)
# --------------------------------------------------------------------------


def test_responses_create_with_conversation():
    bodies = {}

    def handler(request):
        if request.url.path == "/api/v1/conversations":
            return httpx.Response(200, json=_CONVERSATION_BODY)
        if request.url.path == "/api/v1/responses":
            bodies["responses"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"id": "resp_1", "object": "response", "output": []},
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    client = _make_client(handler)
    conv = client.conversations.create()
    client.responses.create(
        model="qwen-plus",
        input="hi",
        conversation=conv.id,
    )
    assert bodies["responses"]["conversation"] == _CONV_ID
    client.close()


def test_responses_conversation_mutex_previous_response_id():
    client = _make_client(_capture_handler({}, _CONVERSATION_BODY))
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.responses.create(
            model="qwen-plus",
            input="hi",
            conversation=_CONV_ID,
            previous_response_id="resp_1",
        )
    client.close()


# --------------------------------------------------------------------------
# async symmetry
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_conversation_crud():
    captured = {}
    client = _make_async_client(
        _capture_handler(captured, _CONVERSATION_BODY),
    )
    conv = await client.conversations.create(metadata={"topic": "demo"})
    assert captured["sdk"].endswith("/conversations2")
    assert conv.id == _CONV_ID
    await client.conversations.get(conv)
    assert captured["path"] == f"/api/v1/conversations/{_CONV_ID}"
    await client.conversations.update(_CONV_ID, metadata={"topic": "x"})
    assert captured["body"] == {"metadata": {"topic": "x"}}
    await client.aclose()


@pytest.mark.asyncio
async def test_async_delete_conversation():
    captured = {}
    client = _make_async_client(
        _capture_handler(captured, _CONV_DELETED_BODY),
    )
    result = await client.conversations.delete(_CONV_ID)
    assert captured["method"] == "DELETE"
    assert result.deleted is True
    await client.aclose()


@pytest.mark.asyncio
async def test_async_items_create_list_get_delete():
    captured = {}
    client = _make_async_client(
        _capture_handler(captured, _ITEMS_CREATE_BODY),
    )
    page = await client.conversations.items.create(
        _CONV_ID,
        items=[{"type": "message", "role": "user", "content": "hi"}],
    )
    assert captured["path"] == f"/api/v1/conversations/{_CONV_ID}/items"
    assert page.items[0].id == _ITEM_ID
    await client.aclose()

    client = _make_async_client(_capture_handler(captured, _ITEMS_LIST_BODY))
    listed = await client.conversations.items.list(_CONV_ID, limit=5)
    assert captured["params"]["limit"] == "5"
    assert listed.items[0].text == "hi"
    await client.aclose()

    client = _make_async_client(_capture_handler(captured, _ITEM_BODY))
    item = await client.conversations.items.get(_CONV_ID, _ITEM_ID)
    assert item.text == "hi"
    await client.aclose()

    client = _make_async_client(
        _capture_handler(captured, _ITEM_DELETED_BODY),
    )
    deleted = await client.conversations.items.delete(_CONV_ID, _ITEM_ID)
    assert deleted.deleted is True
    await client.aclose()


@pytest.mark.asyncio
async def test_async_items_cursor_traversal():
    pages = {
        None: {
            "data": [{**_ITEM_BODY, "id": "msg_1"}],
            "first_id": "msg_1",
            "last_id": "msg_1",
            "has_more": True,
            "object": "list",
        },
        "msg_1": {
            "data": [{**_ITEM_BODY, "id": "msg_2"}],
            "first_id": "msg_2",
            "last_id": "msg_2",
            "has_more": False,
            "object": "list",
        },
    }

    def handler(request):
        return httpx.Response(200, json=pages[request.url.params.get("after")])

    client = _make_async_client(handler)
    first = await client.conversations.items.list(_CONV_ID)
    ids = [item.id async for page in first.iter_pages_async() for item in page]
    assert ids == ["msg_1", "msg_2"]
    await client.aclose()


@pytest.mark.asyncio
async def test_async_routing_default_gateway(clean_env):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_CONVERSATION_BODY)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=_WS_HOST + "/api/v1",
    )
    client = AsyncClient(
        api_key="sk-test",
        workspace="ws-123",
        http_client=http_client,
    )
    await client.conversations.get(_CONV_ID)
    assert urls == [_WS_HOST + _COMPAT_CONVERSATIONS + f"/{_CONV_ID}"]
    await client.aclose()


@pytest.mark.asyncio
async def test_async_error_envelope_when_not_raising():
    def handler(request):
        return httpx.Response(
            404,
            json={"error": {"code": "NotFound", "message": "gone"}},
        )

    client = _make_async_client(handler, raise_on_error=False)
    conv = await client.conversations.get(_CONV_ID)
    assert conv.is_error is True
    assert conv.code == "NotFound"
    await client.aclose()


@pytest.mark.asyncio
async def test_async_raw():
    client = _make_async_client(
        _capture_handler({}, _CONVERSATION_BODY),
    )
    raw = await client.conversations.create(raw=True)
    assert isinstance(raw, APIResponse)
    assert raw.json()["id"] == _CONV_ID
    await client.aclose()


# --------------------------------------------------------------------------
# path-id guards (percent-encoding, rejection before the wire)
# --------------------------------------------------------------------------

_TRICKY_CONV = "conv/a?b#c"
_TRICKY_ITEM = "msg/x?y#z"
_ENCODED_CONV = b"/api/v1/conversations/conv%2Fa%3Fb%23c"
_ENCODED_ITEM = b"/items/msg%2Fx%3Fy%23z"
_VALID_ITEMS = [
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hi"}],
    },
]


def _id_guard_handler(seen):
    def handler(request):
        seen.append(request.url.raw_path)
        path = request.url.path
        if path.endswith("/items"):
            body = (
                _ITEMS_CREATE_BODY
                if request.method == "POST"
                else _ITEMS_LIST_BODY
            )
        elif "/items/" in path:
            body = (
                _ITEM_DELETED_BODY
                if request.method == "DELETE"
                else _ITEM_BODY
            )
        elif request.method == "DELETE":
            body = _CONV_DELETED_BODY
        else:
            body = _CONVERSATION_BODY
        return httpx.Response(200, json=body)

    return handler


def test_conversation_and_item_ids_percent_encoded_in_path():
    seen = []
    client = _make_client(_id_guard_handler(seen))
    client.conversations.get(_TRICKY_CONV)
    client.conversations.update(_TRICKY_CONV, metadata={"topic": "x"})
    client.conversations.delete(_TRICKY_CONV)
    client.conversations.items.create(_TRICKY_CONV, items=_VALID_ITEMS)
    client.conversations.items.list(_TRICKY_CONV)
    client.conversations.items.get(_TRICKY_CONV, _TRICKY_ITEM)
    client.conversations.items.delete(_TRICKY_CONV, _TRICKY_ITEM)
    # '/', '?' and '#' stay inside one path segment instead of retargeting
    # the request to another path or query.
    assert seen == [
        _ENCODED_CONV,
        _ENCODED_CONV,
        _ENCODED_CONV,
        _ENCODED_CONV + b"/items",
        _ENCODED_CONV + b"/items",
        _ENCODED_CONV + _ENCODED_ITEM,
        _ENCODED_CONV + _ENCODED_ITEM,
    ]
    client.close()


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_conversation_id_rejected_before_wire(bad):
    def handler(request):
        raise AssertionError("request must not reach the wire")

    client = _make_client(handler)
    with pytest.raises(ValueError, match="conversation_id"):
        client.conversations.get(bad)
    with pytest.raises(ValueError, match="conversation_id"):
        client.conversations.update(bad, metadata={"k": "v"})
    with pytest.raises(ValueError, match="conversation_id"):
        client.conversations.delete(bad)
    with pytest.raises(ValueError, match="conversation_id"):
        client.conversations.items.create(bad, items=_VALID_ITEMS)
    with pytest.raises(ValueError, match="conversation_id"):
        client.conversations.items.list(bad)
    with pytest.raises(ValueError, match="conversation_id"):
        client.conversations.items.get(bad, _ITEM_ID)
    with pytest.raises(ValueError, match="conversation_id"):
        client.conversations.items.delete(bad, _ITEM_ID)
    client.close()


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_item_id_rejected_before_wire(bad):
    def handler(request):
        raise AssertionError("request must not reach the wire")

    client = _make_client(handler)
    with pytest.raises(ValueError, match="item_id"):
        client.conversations.items.get(_CONV_ID, bad)
    with pytest.raises(ValueError, match="item_id"):
        client.conversations.items.delete(_CONV_ID, bad)
    client.close()


@pytest.mark.asyncio
async def test_async_ids_percent_encoded_in_path():
    seen = []
    client = _make_async_client(_id_guard_handler(seen))
    await client.conversations.get(_TRICKY_CONV)
    await client.conversations.items.get(_TRICKY_CONV, _TRICKY_ITEM)
    await client.conversations.items.delete(_TRICKY_CONV, _TRICKY_ITEM)
    assert seen == [
        _ENCODED_CONV,
        _ENCODED_CONV + _ENCODED_ITEM,
        _ENCODED_CONV + _ENCODED_ITEM,
    ]
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["", "   "])
async def test_async_empty_ids_rejected_before_wire(bad):
    def handler(request):
        raise AssertionError("request must not reach the wire")

    client = _make_async_client(handler)
    with pytest.raises(ValueError, match="conversation_id"):
        await client.conversations.get(bad)
    with pytest.raises(ValueError, match="conversation_id"):
        await client.conversations.update(bad, metadata={"k": "v"})
    with pytest.raises(ValueError, match="conversation_id"):
        await client.conversations.delete(bad)
    with pytest.raises(ValueError, match="conversation_id"):
        await client.conversations.items.list(bad)
    with pytest.raises(ValueError, match="item_id"):
        await client.conversations.items.get(_CONV_ID, bad)
    with pytest.raises(ValueError, match="item_id"):
        await client.conversations.items.delete(_CONV_ID, bad)
    await client.aclose()
