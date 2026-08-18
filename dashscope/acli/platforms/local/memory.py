# -*- coding: utf-8 -*-
"""Local file-based memory provider.

Storage layout:
    .acli/memory/<user_id>.json   (workspace-local)

Each file is a JSON array of memory nodes:
    [
        {
            "id": "uuid",
            "content": "...",
            "created_at": "ISO timestamp",
            "updated_at": "ISO timestamp",
            "score": 0.0
        },
        ...
    ]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dashscope.acli.platforms.base import MemoryNode
from dashscope.acli.utils.ids import now_iso, short_uuid
from dashscope.acli.utils.keywords import extract_keywords
from dashscope.acli.utils.paths import atomic_write_text


def _workspace_memory_dir() -> Path:
    """Return workspace .acli/memory/ directory."""
    from dashscope.acli.config import WORKSPACE_DIR

    return WORKSPACE_DIR / "memory"


class LocalMemoryClient:
    """File-based memory provider. Implements MemoryProvider Protocol."""

    def __init__(self, user_id: str = "default", memory_library_id: str = ""):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", user_id) or ".." in user_id:
            raise ValueError(f"invalid memory user_id: {user_id}")
        self.user_id = user_id
        self.memory_library_id = memory_library_id  # accepted but unused
        self._dir = _workspace_memory_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / f"{user_id}.json"

    # ----- internal helpers -----

    def _load(self) -> list[dict[str, Any]]:
        if not self._file.exists():
            return []
        try:
            return json.loads(self._file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, nodes: list[dict[str, Any]]) -> None:
        atomic_write_text(
            self._file,
            json.dumps(nodes, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def _new_node(
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": short_uuid(),
            "content": content,
            "created_at": now_iso(),
            "updated_at": "",
            "score": 0.0,
            "metadata": metadata or {},
        }

    # ----- MemoryProvider Protocol -----

    async def add(
        self,
        messages: list[dict[str, str]],
        custom_content: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[MemoryNode]:
        """Extract key facts from messages and store as memory nodes.

        For local mode we do simple extraction: pull out lines that look
        like user preferences / facts (contains "I", "my", "prefer", etc.)
        or just store custom_content directly.
        """
        from dashscope.acli.utils import sanitize_text

        nodes = self._load()
        created: list[MemoryNode] = []

        if custom_content:
            clean = sanitize_text(custom_content.strip())
            if not clean or any(n["content"] == clean for n in nodes):
                return []
            node = self._new_node(clean, metadata)
            nodes.append(node)
            created.append(
                MemoryNode(
                    id=node["id"],
                    content=node["content"],
                    metadata=node.get("metadata", {}),
                ),
            )
        else:
            # Extract meaningful lines from conversation
            keywords = re.compile(
                r"(我是|我用|我的|我喜欢|我不|我需要|我的项目|偏好|习惯|"
                r"I am|I use|I prefer|I like|I need|my project|my stack)",
                re.IGNORECASE,
            )
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", "")
                for line in content.splitlines():
                    line = sanitize_text(line.strip())
                    if len(line) < 5 or len(line) > 200:
                        continue
                    if keywords.search(line):
                        # De-duplicate
                        if any(n["content"] == line for n in nodes):
                            continue
                        node = self._new_node(line)
                        nodes.append(node)
                        created.append(MemoryNode(id=node["id"], content=line))

        if created:
            self._save(nodes)
        return created

    async def search(
        self,
        query: str | list[dict[str, str]],
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[MemoryNode]:
        """Simple keyword-based search (no embedding model available
        locally)."""
        nodes = self._load()
        if not nodes:
            return []

        # Build query text
        if isinstance(query, str):
            query_text = query.lower()
        else:
            query_text = " ".join(m.get("content", "") for m in query).lower()

        keywords = extract_keywords(query_text)
        if not keywords:
            return []

        def _weight(kw: str) -> float:
            return 2.0 if len(kw) >= 2 else 0.5

        max_weight = sum(_weight(kw) for kw in keywords)
        scored: list[tuple[float, int, dict]] = []
        for index, node in enumerate(nodes):
            content_lower = node["content"].lower()
            hit_weight = sum(
                _weight(kw) for kw in keywords if kw in content_lower
            )
            if hit_weight > 0:
                score = min(hit_weight / max_weight, 1.0)
                if score >= min_score:
                    scored.append((score, index, node))

        # Highest score first; ties favor the most recently added node.
        scored.sort(key=lambda item: (-item[0], -item[1]))
        results = []
        for score, _, node in scored[:top_k]:
            results.append(
                MemoryNode(
                    id=node["id"],
                    content=node["content"],
                    created_at=node.get("created_at", ""),
                    updated_at=node.get("updated_at", ""),
                    score=score,
                    metadata=node.get("metadata", {}),
                ),
            )
        return results

    async def list(
        self,
        page_num: int = 1,
        page_size: int = 10,
    ) -> list[MemoryNode]:
        """List all memory nodes with pagination."""
        nodes = self._load()
        start = (page_num - 1) * page_size
        end = start + page_size
        results = []
        for node in nodes[start:end]:
            results.append(
                MemoryNode(
                    id=node["id"],
                    content=node["content"],
                    created_at=node.get("created_at", ""),
                    updated_at=node.get("updated_at", ""),
                    metadata=node.get("metadata", {}),
                ),
            )
        return results

    async def delete(self, memory_node_id: str) -> bool:
        """Delete a specific memory node."""
        nodes = self._load()
        original_len = len(nodes)
        nodes = [n for n in nodes if n["id"] != memory_node_id]
        if len(nodes) < original_len:
            self._save(nodes)
            return True
        return False

    async def update(self, memory_node_id: str, content: str) -> bool:
        """Update a memory node's content."""
        nodes = self._load()
        for node in nodes:
            if node["id"] == memory_node_id:
                node["content"] = content
                node["updated_at"] = now_iso()
                self._save(nodes)
                return True
        return False
