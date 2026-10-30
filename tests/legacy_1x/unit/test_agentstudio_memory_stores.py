# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Memory store resource shape tests.

A client with a recording transport that routes (method, path) to
canned payloads; the memories list endpoint serves a two-page queue
so auto-pagination is exercised.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Dict, List, Tuple

import pytest

from dashscope.agentstudio import (
    AsyncClient,
    Client,
    ConflictError,
)
from dashscope.agentstudio.exceptions import from_response
from dashscope.agentstudio.pagination import CursorPage, build_page
from dashscope.agentstudio.transport import APIResponse
from dashscope.agentstudio.types import Memory, MemoryStore, MemoryVersion


def _store_payload() -> Dict[str, Any]:
    return {
        "id": "memstore_01Hx",
        "type": "memory_store",
        "name": "User Preferences",
        "description": "Per-user preferences.",
        "metadata": {"owner": "demo"},
        "created_at": "2026-10-01T00:00:00Z",
        "updated_at": "2026-10-01T00:00:00Z",
        "archived_at": None,
    }


def _archived_store_payload() -> Dict[str, Any]:
    return {**_store_payload(), "archived_at": "2026-10-03T00:00:00Z"}


def _memory_payload() -> Dict[str, Any]:
    return {
        "id": "mem_01Hy",
        "type": "memory",
        "memory_store_id": "memstore_01Hx",
        "path": "/formatting.md",
        "content": "Use GAAP formatting.",
        "content_sha256": "abc123",
        "content_size_bytes": 21,
        "memory_version_id": "memver_01Hz",
        "created_at": "2026-10-01T00:00:00Z",
        "updated_at": "2026-10-02T00:00:00Z",
    }


def _version_payload() -> Dict[str, Any]:
    return {
        "id": "memver_01Hz",
        "type": "memory_version",
        "memory_store_id": "memstore_01Hx",
        "memory_id": "mem_01Hy",
        "operation": "modified",
        "path": "/formatting.md",
        "content": "Use GAAP formatting.",
        "content_sha256": "abc123",
        "content_size_bytes": 21,
        "created_at": "2026-10-02T00:00:00Z",
        "redacted_at": None,
    }


def _redacted_version_payload() -> Dict[str, Any]:
    return {
        **_version_payload(),
        "path": None,
        "content": None,
        "content_sha256": None,
        "content_size_bytes": None,
        "redacted_at": "2026-10-04T00:00:00Z",
    }


_EMPTY_PAGE: Dict[str, Any] = {"data": [], "next_page": None}

_REDACT_PATH = (
    "/memory_stores/memstore_01Hx/memory_versions/memver_01Hz/redact"
)

_RESPONSES: Dict[Tuple[str, str], Callable[[], Any]] = {
    ("POST", "/memory_stores"): _store_payload,
    ("GET", "/memory_stores"): lambda: {
        "data": [_store_payload()],
        "next_page": None,
    },
    ("GET", "/memory_stores/memstore_01Hx"): _store_payload,
    ("POST", "/memory_stores/memstore_01Hx"): _store_payload,
    ("DELETE", "/memory_stores/memstore_01Hx"): lambda: {
        "id": "memstore_01Hx",
        "type": "memory_store_deleted",
    },
    ("POST", "/memory_stores/memstore_01Hx/archive"): (
        _archived_store_payload
    ),
    ("POST", "/memory_stores/memstore_01Hx/memories"): _memory_payload,
    ("GET", "/memory_stores/memstore_01Hx/memories/mem_01Hy"): (
        _memory_payload
    ),
    ("POST", "/memory_stores/memstore_01Hx/memories/mem_01Hy"): (
        _memory_payload
    ),
    ("DELETE", "/memory_stores/memstore_01Hx/memories/mem_01Hy"): (
        lambda: {"id": "mem_01Hy", "type": "memory_deleted"}
    ),
    ("GET", "/memory_stores/memstore_01Hx/memory_versions"): lambda: {
        "data": [_version_payload()],
        "next_page": None,
    },
    ("GET", "/memory_stores/memstore_01Hx/memory_versions/memver_01Hz"): (
        _version_payload
    ),
    ("POST", _REDACT_PATH): _redacted_version_payload,
}


class _Tx:
    """Recording transport routing (method, path) to canned payloads.

    The memories list serves a two-page queue: the first call returns
    a page with a ``next_page`` cursor, the second the (empty) last
    page — exercising auto-pagination.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._pages: Deque[Dict[str, Any]] = deque(
            [
                {
                    "data": [_memory_payload()],
                    "prefixes": ["/notes/"],
                    "next_page": "tok_2",
                },
                {"data": [], "prefixes": [], "next_page": None},
            ],
        )

    def request(self, method: str, path: str, **kwargs: Any):
        self.calls.append({"method": method, "path": path, **kwargs})
        if method == "GET" and path == (
            "/memory_stores/memstore_01Hx/memories"
        ):
            data = self._pages.popleft() if self._pages else dict(_EMPTY_PAGE)
            return APIResponse(data=data, request_id="req_test")
        factory = _RESPONSES.get((method, path))
        data = factory() if factory is not None else {}
        return APIResponse(data=data, request_id="req_test")


class _AsyncTx:
    """Async recording transport delegating to the sync routing."""

    def __init__(self) -> None:
        self._tx = _Tx()
        self.calls = self._tx.calls

    async def request(self, method: str, path: str, **kwargs: Any):
        return self._tx.request(method, path, **kwargs)


@pytest.fixture(name="client")
def _client_fixture():
    c = Client(api_key="test-key", base_url="http://test")
    c.transport = _Tx()
    return c


@pytest.fixture(name="async_client")
def _async_client_fixture():
    c = AsyncClient(api_key="test-key", base_url="http://test")
    c.transport = _AsyncTx()
    return c


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


def test_store_create_posts_body(client):
    store = client.memory_stores.create(
        name="User Preferences",
        description="Per-user preferences.",
        metadata={"owner": "demo"},
    )
    call = client.transport.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/memory_stores"
    assert call["json"] == {
        "name": "User Preferences",
        "description": "Per-user preferences.",
        "metadata": {"owner": "demo"},
    }
    assert isinstance(store, MemoryStore)
    assert store.id == "memstore_01Hx"
    assert store.type == "memory_store"


def test_store_list_serializes_bool_query(client):
    page = client.memory_stores.list(include_archived=True, limit=10)
    call = client.transport.calls[0]
    assert call["method"] == "GET"
    assert call["params"]["include_archived"] == "true"
    assert call["params"]["limit"] == 10
    assert len(page.data) == 1
    assert isinstance(page.data[0], MemoryStore)


def test_store_retrieve_and_get_alias(client):
    store = client.memory_stores.retrieve("memstore_01Hx")
    assert client.memory_stores.get("memstore_01Hx") == store
    assert client.transport.calls[0]["path"] == (
        "/memory_stores/memstore_01Hx"
    )


def test_store_update_posts_partial_body(client):
    client.memory_stores.update("memstore_01Hx", name="Renamed")
    assert client.transport.calls[0]["json"] == {"name": "Renamed"}


def test_store_update_requires_at_least_one_field(client):
    with pytest.raises(TypeError):
        client.memory_stores.update("memstore_01Hx")


def test_store_archive_returns_archived_store(client):
    store = client.memory_stores.archive("memstore_01Hx")
    assert client.transport.calls[0]["path"] == (
        "/memory_stores/memstore_01Hx/archive"
    )
    assert store.archived_at is not None


def test_store_delete_returns_deleted_marker(client):
    resp = client.memory_stores.delete("memstore_01Hx")
    assert client.transport.calls[0]["method"] == "DELETE"
    assert resp.id == "memstore_01Hx"
    assert resp.type == "memory_store_deleted"


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------


def test_memory_create_posts_path_and_content(client):
    mem = client.memory_stores.memories.create(
        "memstore_01Hx",
        path="/formatting.md",
        content="Use GAAP formatting.",
    )
    call = client.transport.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/memory_stores/memstore_01Hx/memories"
    assert call["json"] == {
        "path": "/formatting.md",
        "content": "Use GAAP formatting.",
    }
    assert isinstance(mem, Memory)
    assert mem.memory_store_id == "memstore_01Hx"


def test_memory_list_returns_prefixes_and_pages(client):
    page = client.memory_stores.memories.list(
        "memstore_01Hx",
        path_prefix="/",
        depth=1,
        view="basic",
    )
    call = client.transport.calls[0]
    assert call["params"] == {
        "path_prefix": "/",
        "depth": 1,
        "view": "basic",
    }
    assert isinstance(page, CursorPage)
    assert page.prefixes == ["/notes/"]
    assert len(page.data) == 1
    assert isinstance(page.data[0], Memory)

    # Iterating auto-paginates through the queued second page.
    assert len(list(page)) == 1


def test_memory_list_depth_one_passes_through_wire(client):
    client.memory_stores.memories.list("memstore_01Hx", depth=1)
    assert client.transport.calls[0]["params"] == {"depth": 1}


def test_memory_list_depth_zero_passes_through_wire(client):
    # depth=0 is the native flat mode (same as omitting it).
    client.memory_stores.memories.list("memstore_01Hx", depth=0)
    assert client.transport.calls[0]["params"] == {"depth": 0}


_FLAT_LISTING = {
    "data": [
        {"id": "mem_1", "type": "memory", "path": "/root.md"},
        {"id": "mem_2", "type": "memory", "path": "/notes/top.md"},
        {"id": "mem_3", "type": "memory", "path": "/notes/deep/file.md"},
    ],
    "next_page": None,
}


class _FlatTx:
    """Transport returning a flat (no-depth) memories listing."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def request(self, method, path, **kwargs):
        self.calls.append({"method": method, "path": path, **kwargs})
        return APIResponse(data=dict(_FLAT_LISTING), request_id="req_d")


def _flat_client() -> Client:
    c = Client(api_key="test-key", base_url="http://test")
    c.transport = _FlatTx()
    return c


def test_memory_list_depth_two_emulated_client_side():
    # The backend rejects wire depth >= 2; the SDK omits it and derives
    # the bounded view locally.
    c = _flat_client()
    page = c.memory_stores.memories.list("memstore_01Hx", depth=2)
    assert "depth" not in c.transport.calls[0]["params"]
    # The 3-segment file is cut but leaves its frontier directory.
    assert [m.path for m in page.data] == ["/root.md", "/notes/top.md"]
    assert page.prefixes == ["/notes/", "/notes/deep/"]


def test_memory_list_depth_three_emulated():
    c = _flat_client()
    page = c.memory_stores.memories.list("memstore_01Hx", depth=3)
    assert "depth" not in c.transport.calls[0]["params"]
    assert [m.path for m in page.data] == [
        "/root.md",
        "/notes/top.md",
        "/notes/deep/file.md",
    ]
    assert page.extra["prefixes"] == ["/notes/", "/notes/deep/"]


def test_page_extra_bag_preserves_unknown_envelope_fields():
    payload = {
        "data": [{"id": "memstore_1", "type": "memory_store"}],
        "next_page": None,
        "prefixes": ["/a/"],
        "future_field": {"x": 1},
    }
    page = build_page(
        payload=payload,
        item_factory=lambda p: MemoryStore(**p),
        request_id="req_x",
    )
    assert isinstance(page.data[0], MemoryStore)
    assert page.extra == {
        "prefixes": ["/a/"],
        "future_field": {"x": 1},
    }


def test_store_list_page_extra_defaults_empty(client):
    page = client.memory_stores.list()
    assert page.extra == {}
    # Non-memory pages: the typed accessor returns an empty list.
    assert page.prefixes == []


def test_memory_update_sends_flattened_precondition(client):
    mem = client.memory_stores.memories.update(
        "memstore_01Hx",
        "mem_01Hy",
        content="CORRECTED",
        content_sha256="abc123",
    )
    assert client.transport.calls[0]["json"] == {
        "content": "CORRECTED",
        "content_sha256": "abc123",
    }
    assert isinstance(mem, Memory)


def test_memory_update_requires_path_or_content(client):
    with pytest.raises(TypeError):
        client.memory_stores.memories.update(
            "memstore_01Hx",
            "mem_01Hy",
            content_sha256="abc123",
        )


def test_memory_delete_returns_marker(client):
    resp = client.memory_stores.memories.delete(
        "memstore_01Hx",
        "mem_01Hy",
    )
    assert client.transport.calls[0]["method"] == "DELETE"
    assert resp.type == "memory_deleted"


# ---------------------------------------------------------------------------
# Memory versions
# ---------------------------------------------------------------------------


def test_version_list_uses_dotted_time_query_keys(client):
    page = client.memory_stores.memory_versions.list(
        "memstore_01Hx",
        memory_id="mem_01Hy",
        operation="modified",
        created_at_gte="2026-10-01T00:00:00Z",
        created_at_lte="2026-10-31T00:00:00Z",
        view="full",
    )
    call = client.transport.calls[0]
    assert call["path"] == "/memory_stores/memstore_01Hx/memory_versions"
    assert call["params"] == {
        "memory_id": "mem_01Hy",
        "operation": "modified",
        "created_at.gte": "2026-10-01T00:00:00Z",
        "created_at.lte": "2026-10-31T00:00:00Z",
        "view": "full",
    }
    assert isinstance(page.data[0], MemoryVersion)
    assert page.data[0].operation == "modified"


def test_version_retrieve_and_redact(client):
    version = client.memory_stores.memory_versions.retrieve(
        "memstore_01Hx",
        "memver_01Hz",
    )
    assert version.id == "memver_01Hz"

    redacted = client.memory_stores.memory_versions.redact(
        "memstore_01Hx",
        "memver_01Hz",
    )
    call = client.transport.calls[1]
    assert call["method"] == "POST"
    assert call["path"] == (
        "/memory_stores/memstore_01Hx/memory_versions/memver_01Hz/redact"
    )
    assert redacted.content is None
    assert redacted.redacted_at == "2026-10-04T00:00:00Z"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "memory_path_conflict_error",
        "memory_precondition_failed_error",
        "memory_store_archived_error",
    ],
)
def test_memory_error_codes_raise_conflict_with_code(code: str) -> None:
    err = from_response(
        status_code=409,
        body={"error": {"code": code, "message": "boom"}},
    )
    assert type(err) is ConflictError
    assert err.code == code
    assert err.status_code == 409


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_memory_resources(async_client):
    store = await async_client.memory_stores.create(name="Prefs")
    assert isinstance(store, MemoryStore)

    page = await async_client.memory_stores.memories.list(
        "memstore_01Hx",
    )
    assert page.prefixes == ["/notes/"]

    mem = await async_client.memory_stores.memories.create(
        "memstore_01Hx",
        path="/formatting.md",
        content="Use GAAP formatting.",
    )
    assert isinstance(mem, Memory)

    version = await async_client.memory_stores.memory_versions.redact(
        "memstore_01Hx",
        "memver_01Hz",
    )
    assert version.redacted_at is not None
