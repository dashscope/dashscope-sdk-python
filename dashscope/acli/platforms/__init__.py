# -*- coding: utf-8 -*-
from __future__ import annotations

from dashscope.acli.platforms.base import (
    AgentInfo,
    Category,
    ContextChatResponse,
    ContextInfo,
    ContextProvider,
    DataProvider,
    FileInfo,
    IndexDocument,
    IndexInfo,
    KBProvider,
    MemoryLibrary,
    MemoryNode,
    MemoryProvider,
    PromptProvider,
    PromptTemplate,
    RetrieveNode,
    SearchProvider,
    SearchResult,
)

__all__ = [
    "MemoryNode",
    "FileInfo",
    "Category",
    "IndexInfo",
    "IndexDocument",
    "RetrieveNode",
    "PromptTemplate",
    "SearchResult",
    "AgentInfo",
    "MemoryLibrary",
    "ContextInfo",
    "ContextChatResponse",
    "MemoryProvider",
    "KBProvider",
    "DataProvider",
    "PromptProvider",
    "SearchProvider",
    "ContextProvider",
    "get_memory_provider",
    "get_cli_provider",
]


def get_memory_provider(config) -> MemoryProvider | None:
    """Always use local file-based memory storage."""
    user_id = config.memory_user_id
    if not user_id:
        from dashscope.acli.utils.ids import stable_memory_user_id

        user_id = stable_memory_user_id()
    from dashscope.acli.platforms.local import LocalMemoryClient

    return LocalMemoryClient(
        user_id=user_id,
        memory_library_id=config.memory_library_id,
    )


def get_cli_provider(config):
    """Return a BailianCLIClient when the `bl` binary is on PATH, else None.

    `bl` handles its own auth state (`bl auth login` or env), but we forward
    config.tongyi_api_key as the --api-key fallback so a one-config setup just
    works.
    """
    from dashscope.acli.platforms.bailian.cli import BailianCLIClient

    client = BailianCLIClient(api_key=config.tongyi_api_key)
    return client if client.available() else None
