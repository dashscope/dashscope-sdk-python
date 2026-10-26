# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Skills and SkillVersions resource classes."""

from __future__ import annotations

import os  # pylint: disable=unused-import  # noqa: F401
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union

from dashscope.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
    build_page,
)
from dashscope.agentstudio.resources._helpers import (
    _coerce_skill,
    _coerce_skill_version,
)
from dashscope.agentstudio.types import DeleteResponse, Skill, SkillVersion
from dashscope.agentstudio.types.params import (
    SkillCreateParams,
    SkillListParams,
    SkillVersionCreateParams,
    SkillVersionListParams,
)


_PATH_SKILLS = "/skills"

FileSource = Union[str, "os.PathLike[str]", BinaryIO, Tuple[str, BinaryIO]]


class Skills:
    """Skill CRUD — instance methods backed by a shared client."""

    def __init__(self, client) -> None:
        self._client = client
        self.versions = SkillVersions(client)

    def create(
        self,
        *,
        file_id: Optional[str] = None,
        file: Optional[FileSource] = None,
        mime_type: Optional[str] = None,
    ) -> Skill:
        resolved = _resolve_file_id_sync(
            client=self._client,
            file_id=file_id,
            file=file,
            mime_type=mime_type,
        )
        body = SkillCreateParams(file_id=resolved).to_dict()
        resp = self._client.transport.request("POST", _PATH_SKILLS, json=body)
        return _coerce_skill(resp.data)

    def retrieve(self, skill_id: str) -> Skill:
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_SKILLS}/{skill_id}",
        )
        return _coerce_skill(resp.data)

    # Alias: get() delegates to retrieve()
    get = retrieve  # type: ignore[assignment]

    # Alias: call() delegates to create()
    call = create  # type: ignore[assignment]

    def list(
        self,
        *,
        source: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> CursorPage[Skill]:
        params = SkillListParams(
            source=source,
            limit=limit,
            page=page,
        ).to_dict()
        resp = self._client.transport.request(
            "GET",
            _PATH_SKILLS,
            params=params,
        )
        return build_page(
            payload=resp.data,
            item_factory=_coerce_skill,
            request_id=resp.request_id,
            fetch_next=lambda nxt: self.list(
                source=source,
                limit=limit,
                page=nxt,
            ),
        )

    def delete(self, skill_id: str) -> DeleteResponse:
        resp = self._client.transport.request(
            "DELETE",
            f"{_PATH_SKILLS}/{skill_id}",
        )
        return DeleteResponse(**resp.data)


class SkillVersions:
    """Skill version sub-resource."""

    def __init__(self, client) -> None:
        self._client = client

    def create(
        self,
        skill_id: str,
        *,
        file_id: Optional[str] = None,
        file: Optional[FileSource] = None,
        mime_type: Optional[str] = None,
    ) -> SkillVersion:
        resolved = _resolve_file_id_sync(
            client=self._client,
            file_id=file_id,
            file=file,
            mime_type=mime_type,
        )
        body = SkillVersionCreateParams(file_id=resolved).to_dict()
        resp = self._client.transport.request(
            "POST",
            f"{_PATH_SKILLS}/{skill_id}/versions",
            json=body,
        )
        return _coerce_skill_version(resp.data)

    def list(
        self,
        skill_id: str,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> CursorPage[SkillVersion]:
        params = SkillVersionListParams(limit=limit, page=page).to_dict()
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_SKILLS}/{skill_id}/versions",
            params=params,
        )
        return build_page(
            payload=resp.data,
            item_factory=_coerce_skill_version,
            request_id=resp.request_id,
            fetch_next=lambda nxt: self.list(
                skill_id,
                limit=limit,
                page=nxt,
            ),
        )

    def retrieve(self, skill_id: str, version: str) -> SkillVersion:
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_SKILLS}/{skill_id}/versions/{version}",
        )
        return _coerce_skill_version(resp.data)

    # Alias: get() delegates to retrieve()
    get = retrieve  # type: ignore[assignment]

    def download(self, skill_id: str, version: str) -> Dict[str, Any]:
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_SKILLS}/{skill_id}/versions/{version}/content",
        )
        return dict(resp.data)


class AsyncSkills:
    """Async Skill CRUD — instance methods backed by a shared client."""

    def __init__(self, client) -> None:
        self._client = client
        self.versions = AsyncSkillVersions(client)

    async def create(
        self,
        *,
        file_id: Optional[str] = None,
        file: Optional[FileSource] = None,
        mime_type: Optional[str] = None,
    ) -> Skill:
        resolved = await _resolve_file_id_async(
            client=self._client,
            file_id=file_id,
            file=file,
            mime_type=mime_type,
        )
        body = SkillCreateParams(file_id=resolved).to_dict()
        resp = await self._client.transport.request(
            "POST",
            _PATH_SKILLS,
            json=body,
        )
        return _coerce_skill(resp.data)

    async def retrieve(self, skill_id: str) -> Skill:
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_SKILLS}/{skill_id}",
        )
        return _coerce_skill(resp.data)

    # Alias: get() delegates to retrieve()
    get = retrieve  # type: ignore[assignment]

    # Alias: call() delegates to create()
    call = create  # type: ignore[assignment]

    async def list(
        self,
        *,
        source: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[Skill]:
        params = SkillListParams(
            source=source,
            limit=limit,
            page=page,
        ).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _PATH_SKILLS,
            params=params,
        )

        async def fetch_next(nxt: str) -> AsyncCursorPage[Skill]:
            return await self.list(
                source=source,
                limit=limit,
                page=nxt,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_skill,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def delete(self, skill_id: str) -> DeleteResponse:
        resp = await self._client.transport.request(
            "DELETE",
            f"{_PATH_SKILLS}/{skill_id}",
        )
        return DeleteResponse(**resp.data)


class AsyncSkillVersions:
    """Async skill version sub-resource."""

    def __init__(self, client) -> None:
        self._client = client

    async def create(
        self,
        skill_id: str,
        *,
        file_id: Optional[str] = None,
        file: Optional[FileSource] = None,
        mime_type: Optional[str] = None,
    ) -> SkillVersion:
        resolved = await _resolve_file_id_async(
            client=self._client,
            file_id=file_id,
            file=file,
            mime_type=mime_type,
        )
        body = SkillVersionCreateParams(file_id=resolved).to_dict()
        resp = await self._client.transport.request(
            "POST",
            f"{_PATH_SKILLS}/{skill_id}/versions",
            json=body,
        )
        return _coerce_skill_version(resp.data)

    async def list(
        self,
        skill_id: str,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
    ) -> AsyncCursorPage[SkillVersion]:
        params = SkillVersionListParams(limit=limit, page=page).to_dict()
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_SKILLS}/{skill_id}/versions",
            params=params,
        )

        async def fetch_next(nxt: str) -> AsyncCursorPage[SkillVersion]:
            return await self.list(
                skill_id,
                limit=limit,
                page=nxt,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_skill_version,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def retrieve(self, skill_id: str, version: str) -> SkillVersion:
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_SKILLS}/{skill_id}/versions/{version}",
        )
        return _coerce_skill_version(resp.data)

    # Alias: get() delegates to retrieve()
    get = retrieve  # type: ignore[assignment]

    async def download(self, skill_id: str, version: str) -> Dict[str, Any]:
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_SKILLS}/{skill_id}/versions/{version}/content",
        )
        return dict(resp.data)


# ---------------------------------------------------------------------------
# File-ID resolution helpers (internal)
# ---------------------------------------------------------------------------


def _resolve_file_id_sync(
    *,
    client: Any,
    file_id: Optional[str],
    file: Optional[FileSource],
    mime_type: Optional[str],
) -> str:
    if (file_id is None) == (file is None):
        raise TypeError(
            "skills upload requires exactly one of file_id=... or file=...",
        )
    if file_id is not None:
        return file_id
    # Use the client's files resource to upload
    uploaded = client.files.upload(
        file,
        mime_type=mime_type or "application/zip",
    )
    return uploaded.id


async def _resolve_file_id_async(
    *,
    client: Any,
    file_id: Optional[str],
    file: Optional[FileSource],
    mime_type: Optional[str],
) -> str:
    if (file_id is None) == (file is None):
        raise TypeError(
            "skills upload requires exactly one of file_id=... or file=...",
        )
    if file_id is not None:
        return file_id
    uploaded = await client.files.upload(
        file,
        mime_type=mime_type or "application/zip",
    )
    return uploaded.id
