# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Memory stores resource classes.

A memory store is a workspace-scoped collection of small text
documents that persists across sessions.  Stores are attached to
sessions via the session ``resources[]`` array and mounted into the
session container at ``/mnt/memory/<name>/``.

Object model::

    memory_stores                (memstore_...)   workspace container
      .memories                  (mem_...)        one text file per path
      .memory_versions           (memver_...)     immutable snapshots

Follows the Claude Managed Agents ``client.beta.memory_stores`` SDK
surface, with these verified divergences (the backend is
bailian-app-control ``MemoryStoreController`` over bailian-app-context
OTS storage):

* updates use POST; the optimistic-lock hash rides the top-level
  ``content_sha256`` body field (CMA uses PATCH with a
  ``precondition`` object);
* store ``metadata`` on update is replaced wholesale; CMA merge-patches
  it;
* ``view="full"`` caps ``limit`` at 20 and rejects higher values with
  400; CMA silently clamps;
* list filter keys use dotted wire names (``created_at.gte``), and
  directory nodes come back as a separate ``prefixes`` page field
  (``page.prefixes``, mirrored by ``page.getPrefixes()`` in the Java
  SDK) instead of inline ``memory_prefix`` items;
* depth accepts 0/1 natively; larger values are derived client-side
  (see ``_MAX_WIRE_DEPTH``);
* create and update responses never include ``content``;
* no store-list time filters, no conditional memory delete, no version
  attribution fields or filters — CMA has all of them (unknown filters
  are silently ignored).
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

from dashscope.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
    build_page,
)
from dashscope.agentstudio.resources._helpers import (
    _coerce_memory,
    _coerce_memory_store,
    _coerce_memory_version,
    _memories_path,
    _memory_item_path,
    _memory_stores_path,
    _memory_store_archive_path,
    _memory_store_item_path,
    _memory_version_item_path,
    _memory_version_redact_path,
    _memory_versions_path,
)
from dashscope.agentstudio.types import (
    DeleteResponse,
    Memory,
    MemoryStore,
    MemoryVersion,
)
from dashscope.agentstudio.types.params import (
    MemoryCreateParams,
    MemoryListParams,
    MemoryStoreCreateParams,
    MemoryStoreListParams,
    MemoryStoreUpdateParams,
    MemoryUpdateParams,
    MemoryVersionListParams,
)


# ---------------------------------------------------------------------------
# Memories (sub-resource of MemoryStores)
# ---------------------------------------------------------------------------

# The backend accepts only depth null/0 (flat) or 1 (single-level tree);
# anything else returns 400 (CMA likewise defines only 0/1). depth >= 2 is
# this SDK's own generalization over the flat listing.
_MAX_WIRE_DEPTH = 1


def _apply_depth_emulation(
    page: Any,
    depth: int,
    path_prefix: Optional[str],
) -> Any:
    """Filter a flat page into a depth-bounded tree view.

    Drops files more than ``depth`` segments below ``path_prefix``
    and synthesizes directory prefixes (cut files still contribute
    their frontier directory). Both are per page.
    """

    base = path_prefix or "/"
    if not base.endswith("/"):
        base += "/"
    kept: List[Any] = []
    dirs: set = set()
    for item in page.data:
        path = item.path or ""
        rel = path[len(base) :] if path.startswith(base) else path.lstrip("/")
        segs = [s for s in rel.split("/") if s]
        if len(segs) <= depth:
            kept.append(item)
        for lvl in range(1, min(len(segs) - 1, depth) + 1):
            dirs.add(base + "/".join(segs[:lvl]) + "/")
    page.data = kept
    page.extra["prefixes"] = sorted(dirs)
    return page


class Memories:
    def __init__(self, client) -> None:
        self._client = client

    def create(
        self,
        store_id: str,
        *,
        path: str,
        content: str,
    ) -> Memory:
        """Create a memory at ``path`` (``POST .../memories``).

        A memory already existing at ``path`` returns 409
        ``memory_path_conflict_error``.
        """
        body = MemoryCreateParams(path=path, content=content).to_dict()
        resp = self._client.transport.request(
            "POST",
            _memories_path(store_id),
            json=body,
        )
        return _coerce_memory(resp.data)

    def list(
        self,
        store_id: str,
        *,
        path_prefix: Optional[str] = None,
        depth: Optional[int] = None,
        view: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> CursorPage[Memory]:
        """List memories (``GET .../memories``).

        ``path_prefix`` scopes the listing (server-validated: must
        start and end with ``/`` — segment-aligned, so ``/notes/``
        matches ``/notes/a.md`` but not ``/notes_backup/old.md``; a
        bare ``"/"`` lists the whole store). File items land in
        ``data``; directory nodes come back as ``page.prefixes``
        (per page; merge across pages manually for a full tree).
        ``view="full"`` includes ``content`` per item but caps
        ``limit`` at 20 — the backend rejects higher values with
        400 (CMA silently clamps); the default ``"basic"`` returns
        metadata only with ``limit`` up to 100.

        ``depth`` bounds the tree walk below ``path_prefix``; the
        backend implements 0/1 natively and larger values are derived
        client-side (see ``_MAX_WIRE_DEPTH``). With ``view="full"`` the
        derivation still transfers the content of files the depth cut
        discards — prefer ``view="basic"`` for deep walks.
        """
        depth_emulated = depth is not None and depth > _MAX_WIRE_DEPTH
        wire_depth = None if depth_emulated else depth
        params = MemoryListParams(
            path_prefix=path_prefix,
            depth=wire_depth,
            view=view,
            limit=limit,
            page=page,
        ).to_dict()
        resp = self._client.transport.request(
            "GET",
            _memories_path(store_id),
            params=params,
        )
        result = build_page(
            payload=resp.data,
            item_factory=_coerce_memory,
            request_id=resp.request_id,
            fetch_next=lambda nxt: self.list(
                store_id,
                path_prefix=path_prefix,
                depth=depth,
                view=view,
                limit=limit,
                page=nxt,
            ),
        )
        if depth_emulated:
            result = _apply_depth_emulation(result, depth, path_prefix)
        return result

    def retrieve(
        self,
        store_id: str,
        memory_id: str,
    ) -> Memory:
        """Retrieve a memory with content (``GET .../memories/{id}``)."""
        resp = self._client.transport.request(
            "GET",
            _memory_item_path(store_id, memory_id),
        )
        return _coerce_memory(resp.data)

    get = retrieve  # type: ignore[assignment]

    def update(
        self,
        store_id: str,
        memory_id: str,
        *,
        path: Optional[str] = None,
        content: Optional[str] = None,
        content_sha256: Optional[str] = None,
    ) -> Memory:
        """Update a memory (``POST .../memories/{id}``).

        Mutate ``content``, rename via ``path``, or both — at least one
        must be given.  Pass the sha from the previously retrieved
        memory as ``content_sha256`` for optimistic concurrency; a
        mismatch returns 409 ``memory_precondition_failed_error``.
        """
        if path is None and content is None:
            raise TypeError(
                "memories.update() requires at least one of "
                "'path' or 'content'",
            )
        body = MemoryUpdateParams(
            path=path,
            content=content,
            content_sha256=content_sha256,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            _memory_item_path(store_id, memory_id),
            json=body,
        )
        return _coerce_memory(resp.data)

    def delete(
        self,
        store_id: str,
        memory_id: str,
    ) -> DeleteResponse:
        """Delete a memory (``DELETE .../memories/{id}``)."""
        resp = self._client.transport.request(
            "DELETE",
            _memory_item_path(store_id, memory_id),
        )
        return DeleteResponse(**resp.data)


# ---------------------------------------------------------------------------
# MemoryVersions (sub-resource of MemoryStores)
# ---------------------------------------------------------------------------


class MemoryVersions:
    def __init__(self, client) -> None:
        self._client = client

    def list(
        self,
        store_id: str,
        *,
        memory_id: Optional[str] = None,
        operation: Optional[str] = None,
        created_at_gte: Optional[str] = None,
        created_at_lte: Optional[str] = None,
        view: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> CursorPage[MemoryVersion]:
        """List versions, newest-first (``GET .../memory_versions``).

        Filters: ``memory_id``, ``operation`` (``created`` / ``modified``
        / ``deleted``), ``created_at_gte`` / ``created_at_lte`` (ISO-8601
        timestamps; sent on the wire as ``created_at.gte`` /
        ``created_at.lte``), and ``view`` (``basic`` / ``full``).
        """
        params = MemoryVersionListParams(
            memory_id=memory_id,
            operation=operation,
            created_at_gte=created_at_gte,
            created_at_lte=created_at_lte,
            view=view,
            limit=limit,
            page=page,
        ).to_dict()
        resp = self._client.transport.request(
            "GET",
            _memory_versions_path(store_id),
            params=params,
        )
        return build_page(
            payload=resp.data,
            item_factory=_coerce_memory_version,
            request_id=resp.request_id,
            fetch_next=lambda nxt: self.list(
                store_id,
                memory_id=memory_id,
                operation=operation,
                created_at_gte=created_at_gte,
                created_at_lte=created_at_lte,
                view=view,
                limit=limit,
                page=nxt,
            ),
        )

    def retrieve(
        self,
        store_id: str,
        version_id: str,
    ) -> MemoryVersion:
        """Retrieve a version snapshot (``GET .../memory_versions/{id}``)."""
        resp = self._client.transport.request(
            "GET",
            _memory_version_item_path(store_id, version_id),
        )
        return _coerce_memory_version(resp.data)

    get = retrieve  # type: ignore[assignment]

    def redact(
        self,
        store_id: str,
        version_id: str,
    ) -> MemoryVersion:
        """Redact a version (``POST .../memory_versions/{id}/redact``).

        Scrubs ``path`` / ``content`` / ``content_sha256`` /
        ``content_size_bytes`` while preserving the audit trail
        (``created_at``, ids).  Idempotent and irreversible.
        """
        resp = self._client.transport.request(
            "POST",
            _memory_version_redact_path(store_id, version_id),
        )
        return _coerce_memory_version(resp.data)


# ---------------------------------------------------------------------------
# MemoryStores (parent resource)
# ---------------------------------------------------------------------------


class MemoryStores:
    def __init__(self, client) -> None:
        self._client = client
        self.memories = Memories(client)
        self.memory_versions = MemoryVersions(client)

    def create(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryStore:
        """Create a memory store (``POST /memory_stores``).

        ``description`` is passed to the agent so it knows what the
        store contains — write it for the model, not for humans.
        """
        body = MemoryStoreCreateParams(
            name=name,
            description=description,
            metadata=dict(metadata) if metadata is not None else None,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            _memory_stores_path(),
            json=body,
        )
        return _coerce_memory_store(resp.data)

    def retrieve(self, store_id: str) -> MemoryStore:
        resp = self._client.transport.request(
            "GET",
            _memory_store_item_path(store_id),
        )
        return _coerce_memory_store(resp.data)

    get = retrieve  # type: ignore[assignment]

    def update(
        self,
        store_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryStore:
        """Update a store (``POST /memory_stores/{store_id}``).

        Mutate ``name`` / ``description`` / ``metadata`` — at least one
        must be given. ``metadata`` is **replaced wholesale** (unlike
        CMA's merge-patch): keys not included in the request are
        dropped; a key set to ``null`` deletes it.
        """
        if name is None and description is None and metadata is None:
            raise TypeError(
                "memory_stores.update() requires at least one of "
                "'name', 'description' or 'metadata'",
            )
        body = MemoryStoreUpdateParams(
            name=name,
            description=description,
            metadata=dict(metadata) if metadata is not None else None,
        ).to_dict()
        resp = self._client.transport.request(
            "POST",
            _memory_store_item_path(store_id),
            json=body,
        )
        return _coerce_memory_store(resp.data)

    def list(
        self,
        *,
        include_archived: Optional[bool] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> CursorPage[MemoryStore]:
        """List stores (``GET /memory_stores``).

        The public API has no ``keyword`` filter (that is a
        console-only capability).
        """
        params = MemoryStoreListParams(
            include_archived=include_archived,
            limit=limit,
            page=page,
        ).to_dict()
        resp = self._client.transport.request(
            "GET",
            _memory_stores_path(),
            params=params,
        )
        return build_page(
            payload=resp.data,
            item_factory=_coerce_memory_store,
            request_id=resp.request_id,
            fetch_next=lambda nxt: self.list(
                include_archived=include_archived,
                limit=limit,
                page=nxt,
            ),
        )

    def delete(self, store_id: str) -> DeleteResponse:
        """Delete a store (``DELETE /memory_stores/{id}``).

        Accepted immediately; items are removed asynchronously.  The
        response is a marker (``type="memory_store_deleted"``).
        """
        resp = self._client.transport.request(
            "DELETE",
            _memory_store_item_path(store_id),
        )
        return DeleteResponse(**resp.data)

    def archive(self, store_id: str) -> MemoryStore:
        """Archive a store (``POST /memory_stores/{id}/archive``).

        Read-only afterwards: existing session attachments continue,
        new sessions cannot reference it.  Irreversible.
        """
        resp = self._client.transport.request(
            "POST",
            _memory_store_archive_path(store_id),
        )
        return _coerce_memory_store(resp.data)


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


class AsyncMemories:
    def __init__(self, client) -> None:
        self._client = client

    async def create(
        self,
        store_id: str,
        *,
        path: str,
        content: str,
    ) -> Memory:
        body = MemoryCreateParams(path=path, content=content).to_dict()
        resp = await self._client.transport.request(
            "POST",
            _memories_path(store_id),
            json=body,
        )
        return _coerce_memory(resp.data)

    async def list(
        self,
        store_id: str,
        *,
        path_prefix: Optional[str] = None,
        depth: Optional[int] = None,
        view: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[Memory]:
        """Async variant of :meth:`Memories.list` (depth >= 2 is
        emulated client-side the same way)."""
        depth_emulated = depth is not None and depth > _MAX_WIRE_DEPTH
        wire_depth = None if depth_emulated else depth
        params = MemoryListParams(
            path_prefix=path_prefix,
            depth=wire_depth,
            view=view,
            limit=limit,
            page=page,
        ).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _memories_path(store_id),
            params=params,
        )

        async def fetch_next(nxt: str) -> AsyncCursorPage[Memory]:
            return await self.list(
                store_id,
                path_prefix=path_prefix,
                depth=depth,
                view=view,
                limit=limit,
                page=nxt,
            )

        result = build_page(
            payload=resp.data,
            item_factory=_coerce_memory,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )
        if depth_emulated:
            result = _apply_depth_emulation(result, depth, path_prefix)
        return result

    async def retrieve(
        self,
        store_id: str,
        memory_id: str,
    ) -> Memory:
        resp = await self._client.transport.request(
            "GET",
            _memory_item_path(store_id, memory_id),
        )
        return _coerce_memory(resp.data)

    get = retrieve  # type: ignore[assignment]

    async def update(
        self,
        store_id: str,
        memory_id: str,
        *,
        path: Optional[str] = None,
        content: Optional[str] = None,
        content_sha256: Optional[str] = None,
    ) -> Memory:
        if path is None and content is None:
            raise TypeError(
                "memories.update() requires at least one of "
                "'path' or 'content'",
            )
        body = MemoryUpdateParams(
            path=path,
            content=content,
            content_sha256=content_sha256,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            _memory_item_path(store_id, memory_id),
            json=body,
        )
        return _coerce_memory(resp.data)

    async def delete(
        self,
        store_id: str,
        memory_id: str,
    ) -> DeleteResponse:
        resp = await self._client.transport.request(
            "DELETE",
            _memory_item_path(store_id, memory_id),
        )
        return DeleteResponse(**resp.data)


class AsyncMemoryVersions:
    def __init__(self, client) -> None:
        self._client = client

    async def list(
        self,
        store_id: str,
        *,
        memory_id: Optional[str] = None,
        operation: Optional[str] = None,
        created_at_gte: Optional[str] = None,
        created_at_lte: Optional[str] = None,
        view: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[MemoryVersion]:
        params = MemoryVersionListParams(
            memory_id=memory_id,
            operation=operation,
            created_at_gte=created_at_gte,
            created_at_lte=created_at_lte,
            view=view,
            limit=limit,
            page=page,
        ).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _memory_versions_path(store_id),
            params=params,
        )

        async def fetch_next(nxt: str) -> AsyncCursorPage[MemoryVersion]:
            return await self.list(
                store_id,
                memory_id=memory_id,
                operation=operation,
                created_at_gte=created_at_gte,
                created_at_lte=created_at_lte,
                view=view,
                limit=limit,
                page=nxt,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_memory_version,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def retrieve(
        self,
        store_id: str,
        version_id: str,
    ) -> MemoryVersion:
        resp = await self._client.transport.request(
            "GET",
            _memory_version_item_path(store_id, version_id),
        )
        return _coerce_memory_version(resp.data)

    get = retrieve  # type: ignore[assignment]

    async def redact(
        self,
        store_id: str,
        version_id: str,
    ) -> MemoryVersion:
        resp = await self._client.transport.request(
            "POST",
            _memory_version_redact_path(store_id, version_id),
        )
        return _coerce_memory_version(resp.data)


class AsyncMemoryStores:
    def __init__(self, client) -> None:
        self._client = client
        self.memories = AsyncMemories(client)
        self.memory_versions = AsyncMemoryVersions(client)

    async def create(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryStore:
        body = MemoryStoreCreateParams(
            name=name,
            description=description,
            metadata=dict(metadata) if metadata is not None else None,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            _memory_stores_path(),
            json=body,
        )
        return _coerce_memory_store(resp.data)

    async def retrieve(self, store_id: str) -> MemoryStore:
        resp = await self._client.transport.request(
            "GET",
            _memory_store_item_path(store_id),
        )
        return _coerce_memory_store(resp.data)

    get = retrieve  # type: ignore[assignment]

    async def update(
        self,
        store_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MemoryStore:
        if name is None and description is None and metadata is None:
            raise TypeError(
                "memory_stores.update() requires at least one of "
                "'name', 'description' or 'metadata'",
            )
        body = MemoryStoreUpdateParams(
            name=name,
            description=description,
            metadata=dict(metadata) if metadata is not None else None,
        ).to_dict()
        resp = await self._client.transport.request(
            "POST",
            _memory_store_item_path(store_id),
            json=body,
        )
        return _coerce_memory_store(resp.data)

    async def list(
        self,
        *,
        include_archived: Optional[bool] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[MemoryStore]:
        params = MemoryStoreListParams(
            include_archived=include_archived,
            limit=limit,
            page=page,
        ).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _memory_stores_path(),
            params=params,
        )

        async def fetch_next(nxt: str) -> AsyncCursorPage[MemoryStore]:
            return await self.list(
                include_archived=include_archived,
                limit=limit,
                page=nxt,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_memory_store,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def delete(self, store_id: str) -> DeleteResponse:
        resp = await self._client.transport.request(
            "DELETE",
            _memory_store_item_path(store_id),
        )
        return DeleteResponse(**resp.data)

    async def archive(self, store_id: str) -> MemoryStore:
        resp = await self._client.transport.request(
            "POST",
            _memory_store_archive_path(store_id),
        )
        return _coerce_memory_store(resp.data)
