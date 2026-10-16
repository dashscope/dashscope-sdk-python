# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.files resource (upload/list/get/delete)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, Field

from dashscope.v2.errors import (
    BadRequestError,
    NotFoundError,
    check_response,
    response_error,
)
from dashscope.v2.types.common import (
    ResponseBase,
    response_envelope,
    with_headers,
)

_BASE_PATH = "/files"

MODULE = "files"

# DashScope reports a file that does not exist as a 400-family error (wire
# code ``InvalidParameter``, message "File not found.") rather than a 404, so
# the catalog-driven mapper in ``errors`` yields ``BadRequestError``. The
# message is the only field that states the file is missing; ``get``/``delete``
# match it narrowly to surface ``NotFoundError`` instead. English-only by
# design -- the v2 tree stays ASCII and the gateway's documented text is
# English.
_FILE_NOT_FOUND = re.compile(
    r"not\s+found|not\s+exist|no\s+such\s+file",
    re.IGNORECASE,
)

__all__ = [
    "AsyncFiles",
    "DeletionStatus",
    "File",
    "FileList",
    "Files",
]


class File(ResponseBase):
    """One uploaded file.

    Returned on its own by ``upload`` / ``get`` and nested in
    ``FileList.files``; both roles share one shape. The envelope is filled in
    on the standalone result and stays empty on the nested copies, because
    the server reports it once per response rather than once per item.
    """

    file_id: Optional[str] = None
    name: Optional[str] = None
    purpose: Optional[str] = None
    size: Optional[int] = None
    uploaded_at: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("uploaded_at", "upload_time"),
    )


class FileList(ResponseBase):
    page_no: int = 1
    page_size: int = 10
    total_count: int = 0
    files: List[File] = []


class DeletionStatus(ResponseBase):
    deleted: bool = False


def _parse_upload(body: Dict[str, Any]) -> File:
    data = body.get("data") or {}
    if not isinstance(data, dict):
        return File(**response_envelope(body))
    if "uploaded_files" in data:
        # The endpoint always answers with a list even though one file went up.
        uploaded = data["uploaded_files"] or []
        data = uploaded[0] if uploaded else {}
    return File(**data, **response_envelope(body))


def _parse_list(body: Dict[str, Any]) -> FileList:
    data = body.get("data") or {}
    if isinstance(data, dict):
        files = data.get("files") or []
        page_no = data.get("page_no", 1)
        page_size = data.get("page_size", 10)
        total = data.get("total", len(files))
    else:
        files, page_no, page_size, total = data, 1, 10, len(data)
    return FileList(
        page_no=page_no,
        page_size=page_size,
        total_count=total,
        files=files,
        **response_envelope(body),
    )


def _parse_file(body: Dict[str, Any]) -> File:
    data = body.get("data") or {}
    if not isinstance(data, dict):
        return File(**response_envelope(body))
    return File(**data, **response_envelope(body))


def _parse_delete(body: Dict[str, Any]) -> DeletionStatus:
    data = body.get("data") or {}
    deleted = True
    if isinstance(data, dict) and "deleted" in data:
        deleted = bool(data["deleted"])
    return DeletionStatus(deleted=deleted, **response_envelope(body))


def _is_valid_jsonl(file_path: str) -> bool:
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            try:
                json.loads(line)
            except json.decoder.JSONDecodeError:
                return False
    return True


def _validate_upload(file_path: str, purpose: str) -> None:
    if purpose in ("fine-tune", "fine_tune"):
        if not _is_valid_jsonl(file_path):
            raise ValueError(
                f"file {file_path} is not in valid jsonl format",
            )
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"file not found: {file_path}")


def _upload_form(file_path: str, description: Optional[str]):
    data = {}
    if description is not None:
        data["descriptions"] = description
    with open(file_path, "rb") as f:
        # Bytes, not a live handle: transport retries must be able
        # to re-send the full body.
        content = f.read()
    return data, {"files": (os.path.basename(file_path), content)}


def _require_file_id(file_id: str) -> None:
    # An empty id would collapse the path to "/files/", silently hitting
    # the list endpoint (or a confusing gateway error); fail locally.
    if not file_id or not file_id.strip():
        raise ValueError("file_id must be a non-empty string")


def _check_by_id_response(response, *, raise_on_error: bool) -> None:
    """``check_response`` for a by-id ``get``/``delete``.

    A missing file arrives as a 400 ``InvalidParameter`` ("File not
    found."), which the catalog mapper turns into ``BadRequestError``.
    Because these two calls address exactly one ``file_id``, that response
    can only mean "no such file", so it is re-raised as ``NotFoundError``.
    This is a deliberate refinement *outside* the catalog: the crosswalk
    maps ``InvalidParameter`` to ``BadRequestError``, and the registry has
    no file-not-found entry to route through (it is generated, DO-NOT-EDIT).
    So unlike ``models.get`` -- catalog-legal because it re-uses the
    ``Model.NotExist`` wire code -- the class is picked here directly. The
    server's real ``status_code`` is preserved (never an invented 404) and
    every other failure propagates unchanged.
    """
    error = response_error(response)
    if error is None or not raise_on_error:
        return
    if isinstance(error, BadRequestError) and _FILE_NOT_FOUND.search(
        error.message or "",
    ):
        raise NotFoundError(
            error.message,
            code=error.code,
            status_code=error.status_code,
            request_id=error.request_id,
            body=error.body,
            headers=error.headers,
        )
    raise error


class Files:
    def __init__(self, client) -> None:
        self._client = client

    def upload(
        self,
        *,
        file_path: str,
        purpose: str = "fine-tune",
        description: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> File:
        _validate_upload(file_path, purpose)
        data, files = _upload_form(file_path, description)
        response = self._client.transport.request(
            "POST",
            _BASE_PATH,
            data=data,
            files=files,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_upload(response.json()), response)

    def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> FileList:
        response = self._client.transport.request(
            "GET",
            _BASE_PATH,
            params={"page_no": page_no, "page_size": page_size},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_list(response.json()), response)

    def get(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> File:
        _require_file_id(file_id)
        response = self._client.transport.request(
            "GET",
            f"{_BASE_PATH}/{file_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        _check_by_id_response(
            response,
            raise_on_error=self._client.raise_on_error,
        )
        return with_headers(_parse_file(response.json()), response)

    def delete(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> DeletionStatus:
        _require_file_id(file_id)
        response = self._client.transport.request(
            "DELETE",
            f"{_BASE_PATH}/{file_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        _check_by_id_response(
            response,
            raise_on_error=self._client.raise_on_error,
        )
        return with_headers(_parse_delete(response.json()), response)


class AsyncFiles:
    def __init__(self, client) -> None:
        self._client = client

    async def upload(
        self,
        *,
        file_path: str,
        purpose: str = "fine-tune",
        description: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> File:
        _validate_upload(file_path, purpose)
        data, files = _upload_form(file_path, description)
        response = await self._client.transport.request(
            "POST",
            _BASE_PATH,
            data=data,
            files=files,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_upload(response.json()), response)

    async def list(
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> FileList:
        response = await self._client.transport.request(
            "GET",
            _BASE_PATH,
            params={"page_no": page_no, "page_size": page_size},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        check_response(response, raise_on_error=self._client.raise_on_error)
        return with_headers(_parse_list(response.json()), response)

    async def get(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> File:
        _require_file_id(file_id)
        response = await self._client.transport.request(
            "GET",
            f"{_BASE_PATH}/{file_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        _check_by_id_response(
            response,
            raise_on_error=self._client.raise_on_error,
        )
        return with_headers(_parse_file(response.json()), response)

    async def delete(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> DeletionStatus:
        _require_file_id(file_id)
        response = await self._client.transport.request(
            "DELETE",
            f"{_BASE_PATH}/{file_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
        )
        _check_by_id_response(
            response,
            raise_on_error=self._client.raise_on_error,
        )
        return with_headers(_parse_delete(response.json()), response)
