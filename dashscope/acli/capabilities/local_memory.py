# -*- coding: utf-8 -*-
"""Local Memory Capability Implementation.

Local, privacy-first implementation of the MemoryCapability interface,
backed by LocalMemoryClient — the same store that serves /profile and
/memory. One store, one file (.acli/memory/default.json): facts stored
through the agent's memory_store tool are searchable from /profile and
vice versa.
"""

from __future__ import annotations

from typing import Any

from dashscope.acli.capabilities import MemoryCapability
from dashscope.acli.platforms.local.memory import LocalMemoryClient


class LocalMemoryCapability(MemoryCapability):
    """Local file-based memory implementation."""

    def __init__(self, client: LocalMemoryClient | None = None):
        self._client = client or LocalMemoryClient()
        self._initialized = False

    @property
    def name(self) -> str:
        return "local.memory"

    @property
    def provider(self) -> str:
        return "local"

    def is_available(self) -> bool:
        return True  # Always available (local)

    async def initialize(self) -> None:
        """Local client reads lazily per operation; nothing to warm up."""
        self._initialized = True

    async def shutdown(self) -> None:
        """Writes are persisted atomically as they happen; nothing to flush."""

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search memories by keyword matching."""
        if not query or not query.strip():
            return []
        nodes = await self._client.search(query, top_k=top_k, min_score=0.1)
        return [
            {
                "id": n.id,
                "content": n.content,
                "score": n.score,
                "created_at": n.created_at,
            }
            for n in nodes
        ]

    async def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a memory entry. Returns entry ID."""
        nodes = await self._client.add(
            [],
            custom_content=content,
            metadata=metadata,
        )
        if not nodes:
            return ""
        return nodes[0].id

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        return await self._client.delete(entry_id)

    async def list(self, limit: int = 20) -> list[dict[str, Any]]:
        """List the most recent memory entries."""
        # Client paginates from the oldest entry; pull all and take the tail
        # so callers see the most recent items (local store is file-backed,
        # so a single wide page is fine).
        nodes = await self._client.list(page_num=1, page_size=10**6)
        return [
            {
                "id": n.id,
                "content": n.content,
                "created_at": n.created_at,
                "metadata": n.metadata,
            }
            for n in nodes[-limit:]
        ]
