# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

# ===== Data Classes =====


@dataclass
class MemoryNode:
    id: str
    content: str
    created_at: str = ""
    updated_at: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class FileInfo:
    file_id: str
    file_name: str = ""
    status: str = ""
    size: int = 0
    category_id: str = ""


@dataclass
class Category:
    category_id: str
    category_name: str
    category_type: str = ""
    parent_category_id: str = ""


@dataclass
class IndexInfo:
    id: str
    name: str = ""
    structure_type: str = ""
    source_type: str = ""
    chunk_size: int = 0
    overlap_size: int = 0


@dataclass
class IndexDocument:
    id: str
    name: str = ""
    status: str = ""
    size: int = 0
    document_type: str = ""
    code: str = ""
    message: str = ""


@dataclass
class RetrieveNode:
    text: str
    score: float = 0.0
    file_name: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class PromptTemplate:
    id: str
    name: str
    content: str
    type: str = "Custom"
    variables: list[str] | None = None


@dataclass
class SearchResult:
    title: str
    content: str
    link: str
    media: str = ""
    publish_date: str = ""


@dataclass
class AgentInfo:
    code: str
    name: str = ""
    instructions: str = ""
    model_id: str = ""


@dataclass
class MemoryLibrary:
    memory_id: str
    description: str = ""


# ===== Provider Protocols =====


class MemoryProvider(Protocol):
    user_id: str

    async def add(
        self,
        messages: list[dict[str, str]],
        custom_content: str = "",
        metadata: dict | None = None,
    ) -> list[MemoryNode]:
        ...

    async def search(
        self,
        query: str | list[dict[str, str]],
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[MemoryNode]:
        ...

    async def list(
        self,
        page_num: int = 1,
        page_size: int = 10,
    ) -> list[MemoryNode]:
        ...

    async def delete(self, node_id: str) -> bool:
        ...

    async def update(self, node_id: str, content: str) -> bool:
        ...


class KBProvider(Protocol):
    def list_indices(
        self,
        page_number: int = 1,
        page_size: int = 20,
    ) -> list[IndexInfo]:
        ...

    def retrieve(
        self,
        index_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[RetrieveNode]:
        ...

    def create_index(
        self,
        name: str,
        category_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
        chunk_size: int = 500,
        overlap_size: int = 100,
    ) -> str:
        ...

    def delete_index(self, index_id: str) -> bool:
        ...

    def submit_index_job(self, index_id: str) -> str:
        ...

    def get_index_job_status(self, index_id: str, job_id: str) -> dict:
        ...

    def add_documents_to_index(
        self,
        index_id: str,
        category_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        ...

    def list_index_documents(
        self,
        index_id: str,
        page_number: int = 1,
        page_size: int = 20,
    ) -> list[IndexDocument]:
        ...

    def delete_index_documents(
        self,
        index_id: str,
        document_ids: list[str],
    ) -> bool:
        ...


class DataProvider(Protocol):
    def upload_file(
        self,
        file_path: str,
        category_id: str = "default",
    ) -> FileInfo:
        ...

    def list_files(
        self,
        category_id: str = "default",
        max_results: int = 20,
    ) -> list[FileInfo]:
        ...

    def delete_file(self, file_id: str, category_id: str = "default") -> bool:
        ...

    def list_categories(
        self,
        category_type: str = "UNSTRUCTURED",
    ) -> list[Category]:
        ...

    def add_category(
        self,
        name: str,
        category_type: str = "UNSTRUCTURED",
    ) -> Category:
        ...


class PromptProvider(Protocol):
    def list(
        self,
        name: str | None = None,
        type: str | None = None,  # pylint: disable=redefined-builtin
        max_results: int = 20,
    ) -> list[PromptTemplate]:
        ...

    def get(self, template_id: str) -> PromptTemplate:
        ...

    def create(self, name: str, content: str) -> PromptTemplate:
        ...

    def update(
        self,
        template_id: str,
        name: str | None = None,
        content: str | None = None,
    ) -> bool:
        ...

    def delete(self, template_id: str) -> bool:
        ...

    def render(self, template_id: str, variables: dict[str, str]) -> str:
        ...


class SearchProvider(Protocol):
    async def search(
        self,
        query: str,
        count: int = 10,
        recency: str = "noLimit",
    ) -> list[SearchResult]:
        ...


@dataclass
class ContextInfo:
    id: str
    model: str
    mode: str = "session"
    ttl: int = 3600
    truncation_strategy: dict = field(default_factory=dict)
    cached_tokens: int = 0


@dataclass
class ContextChatResponse:
    content: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


class ContextProvider(Protocol):
    async def create_context(
        self,
        messages: list[dict[str, str]],
        mode: str = "session",
        ttl: int = 3600,
        truncation_strategy: dict | None = None,
    ) -> ContextInfo:
        ...

    async def chat(
        self,
        context_id: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> ContextChatResponse:
        ...
