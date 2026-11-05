# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.files resource (upload/list/get/delete)."""

from __future__ import annotations

import json
import os
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    overload,
)

from pydantic import AliasChoices, Field

from dashscope.errors import (
    BadRequestError,
    NotFoundError,
    response_error,
    sdk_error,
)
from dashscope.transport._options import request_options
from dashscope.types._parse import parse_model
from dashscope.types.common import (
    ResponseBase,
    response_envelope,
    with_headers,
)
from dashscope.types.pagination import BasePage

if TYPE_CHECKING:
    import httpx

    from dashscope.transport._response import APIResponse

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


class FileList(BasePage[File]):
    """One page of files, from ``files.list``.

    Pagination fields come from
    :class:`~dashscope.types.pagination.BasePage`; iterate the page directly
    or cross pages with ``iter_pages()`` / ``next_page()``.
    """

    files: List[File] = []

    @property
    def items(self) -> List[File]:
        return self.files


class DeletionStatus(ResponseBase):
    deleted: bool = False


def _parse_upload(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional[httpx.Response] = None,
) -> File:
    data = body.get("data") or {}
    if not isinstance(data, dict):
        return parse_model(
            File,
            response_envelope(body),
            strict=strict,
            response=response,
        )
    if "uploaded_files" in data:
        # The endpoint always answers with a list even though one file went up.
        uploaded = data["uploaded_files"] or []
        data = uploaded[0] if uploaded else {}
    # Dict merge, not ``**`` twice: a record carrying an envelope key of
    # its own would otherwise raise a duplicate-keyword TypeError.
    return parse_model(
        File,
        {**data, **response_envelope(body)},
        strict=strict,
        response=response,
    )


def _parse_list(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional[httpx.Response] = None,
) -> FileList:
    data = body.get("data") or {}
    if isinstance(data, dict):
        files = data.get("files")
        # A retyped ``files`` would both poison the ``len()`` below (a
        # string's length is not a page total) and blow up in the model
        # layer's repair of a list field. Empty is the honest reading of
        # "not a list", matching ``JobList.from_payload``.
        if not isinstance(files, list):
            files = []
        page_no = data.get("page_no", 1)
        page_size = data.get("page_size", 10)
        total = data.get("total", len(files))
    else:
        # Only a genuine list can be read as a whole page; ``len()`` on a
        # drifted scalar would escape as a bare TypeError from here.
        files = data if isinstance(data, list) else []
        page_no, page_size, total = 1, 10, len(files)
    return parse_model(
        FileList,
        {
            "page_no": page_no,
            "page_size": page_size,
            "total_count": total,
            "files": files,
            **response_envelope(body),
        },
        strict=strict,
        response=response,
    )


def _parse_file(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional[httpx.Response] = None,
) -> File:
    data = body.get("data") or {}
    if not isinstance(data, dict):
        return parse_model(
            File,
            response_envelope(body),
            strict=strict,
            response=response,
        )
    return parse_model(
        File,
        {**data, **response_envelope(body)},
        strict=strict,
        response=response,
    )


def _parse_delete(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional[httpx.Response] = None,
) -> DeletionStatus:
    data = body.get("data") or {}
    deleted = True
    if isinstance(data, dict) and "deleted" in data:
        deleted = bool(data["deleted"])
    return parse_model(
        DeletionStatus,
        {"deleted": deleted, **response_envelope(body)},
        strict=strict,
        response=response,
    )


def _is_valid_jsonl(file_path: str) -> bool:
    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                try:
                    json.loads(line)
                except json.decoder.JSONDecodeError:
                    return False
    except OSError as exc:
        raise _io_error(exc) from exc
    return True


def _io_error(exc: BaseException) -> Exception:
    # Local filesystem failures are SDK-side, not gateway errors: the
    # catalog's sdk.IoError entry supplies message/code/solution.
    return sdk_error("sdk.IoError", {"detail": str(exc)})


def _validate_upload(file_path: str, purpose: str) -> None:
    if purpose in ("fine-tune", "fine_tune"):
        if not _is_valid_jsonl(file_path):
            raise ValueError(
                f"file {file_path} is not in valid jsonl format",
            )
    if not os.path.isfile(file_path):
        raise _io_error(FileNotFoundError(f"file not found: {file_path}"))


def _upload_form(file_path: str, description: Optional[str]):
    data = {}
    if description is not None:
        data["descriptions"] = description
    try:
        with open(file_path, "rb") as f:
            # Bytes, not a live handle: transport retries must be able
            # to re-send the full body.
            content = f.read()
    except OSError as exc:
        raise _io_error(exc) from exc
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

    @overload
    def upload(
        self,
        *,
        file_path: str,
        purpose: str = "fine-tune",
        description: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> File:
        ...

    @overload
    def upload(
        self,
        *,
        file_path: str,
        purpose: str = "fine-tune",
        description: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def upload(
        self,
        *,
        file_path: str,
        purpose: str = "fine-tune",
        description: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[File, "APIResponse"]:
        """Upload a JSONL file for fine-tuning.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        _validate_upload(file_path, purpose)
        data, files = _upload_form(file_path, description)
        resp = self._client.transport.request_api(
            "POST",
            _BASE_PATH,
            data=data,
            files=files,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_upload(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    def list(  # pylint: disable=redefined-builtin
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> FileList:
        ...

    @overload
    def list(  # pylint: disable=redefined-builtin
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def list(  # pylint: disable=redefined-builtin
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[FileList, "APIResponse"]:
        """List uploaded files, one page at a time.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself. A raw response
        carries no pagination callback -- use ``FileList.iter_pages()`` on the
        parsed result to cross pages.
        """
        resp = self._client.transport.request_api(
            "GET",
            _BASE_PATH,
            params={"page_no": page_no, "page_size": page_size},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        result = _parse_list(
            resp.json(),
            strict=self._client.config.strict_validation,
            response=resp.http_response,
        )
        result.bind_pagination(
            lambda next_no, next_size: self.list(
                page_no=next_no,
                page_size=next_size,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            ),
            requested_page_no=page_no,
            requested_page_size=page_size,
        )
        return with_headers(result, resp.http_response)

    @overload
    def get(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> File:
        ...

    @overload
    def get(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def get(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[File, "APIResponse"]:
        """Fetch one uploaded file's metadata.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself. It also skips the
        ``NotFoundError`` refinement documented on
        :func:`_check_by_id_response`: a missing file comes back as the
        gateway's own 400 body rather than raising.
        """
        _require_file_id(file_id)
        resp = self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/{file_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        _check_by_id_response(
            resp.http_response,
            raise_on_error=self._client.raise_on_error,
        )
        return with_headers(
            _parse_file(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    def delete(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> DeletionStatus:
        ...

    @overload
    def delete(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    def delete(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[DeletionStatus, "APIResponse"]:
        """Delete one uploaded file.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself. It also skips the
        ``NotFoundError`` refinement documented on
        :func:`_check_by_id_response`: deleting a file that is already gone
        comes back as the gateway's own 400 body rather than raising.
        """
        _require_file_id(file_id)
        resp = self._client.transport.request_api(
            "DELETE",
            f"{_BASE_PATH}/{file_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        _check_by_id_response(
            resp.http_response,
            raise_on_error=self._client.raise_on_error,
        )
        return with_headers(
            _parse_delete(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )


class AsyncFiles:
    def __init__(self, client) -> None:
        self._client = client

    @overload
    async def upload(
        self,
        *,
        file_path: str,
        purpose: str = "fine-tune",
        description: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> File:
        ...

    @overload
    async def upload(
        self,
        *,
        file_path: str,
        purpose: str = "fine-tune",
        description: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def upload(
        self,
        *,
        file_path: str,
        purpose: str = "fine-tune",
        description: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[File, "APIResponse"]:
        """Upload a JSONL file for fine-tuning.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself.
        """
        _validate_upload(file_path, purpose)
        data, files = _upload_form(file_path, description)
        resp = await self._client.transport.request_api(
            "POST",
            _BASE_PATH,
            data=data,
            files=files,
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        return with_headers(
            _parse_upload(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    async def list(  # pylint: disable=redefined-builtin
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> FileList:
        ...

    @overload
    async def list(  # pylint: disable=redefined-builtin
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def list(  # pylint: disable=redefined-builtin
        self,
        *,
        page_no: int = 1,
        page_size: int = 10,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[FileList, "APIResponse"]:
        """List uploaded files, one page at a time.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself. A raw response
        carries no pagination callback -- use ``FileList.iter_pages()`` on the
        parsed result to cross pages.
        """
        resp = await self._client.transport.request_api(
            "GET",
            _BASE_PATH,
            params={"page_no": page_no, "page_size": page_size},
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        resp.raise_for_status()
        result = _parse_list(
            resp.json(),
            strict=self._client.config.strict_validation,
            response=resp.http_response,
        )
        result.bind_pagination(
            lambda next_no, next_size: self.list(
                page_no=next_no,
                page_size=next_size,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            ),
            requested_page_no=page_no,
            requested_page_size=page_size,
        )
        return with_headers(result, resp.http_response)

    @overload
    async def get(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> File:
        ...

    @overload
    async def get(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def get(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[File, "APIResponse"]:
        """Fetch one uploaded file's metadata.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself. It also skips the
        ``NotFoundError`` refinement documented on
        :func:`_check_by_id_response`: a missing file comes back as the
        gateway's own 400 body rather than raising.
        """
        _require_file_id(file_id)
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/{file_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        _check_by_id_response(
            resp.http_response,
            raise_on_error=self._client.raise_on_error,
        )
        return with_headers(
            _parse_file(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )

    @overload
    async def delete(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> DeletionStatus:
        ...

    @overload
    async def delete(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[True],
    ) -> "APIResponse":
        ...

    async def delete(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[DeletionStatus, "APIResponse"]:
        """Delete one uploaded file.

        ``raw=True`` returns the :class:`~dashscope.transport.APIResponse`
        before any error check, so a caller can read the status, headers and
        ``elapsed`` of the response or re-parse it itself. It also skips the
        ``NotFoundError`` refinement documented on
        :func:`_check_by_id_response`: deleting a file that is already gone
        comes back as the gateway's own 400 body rather than raising.
        """
        _require_file_id(file_id)
        resp = await self._client.transport.request_api(
            "DELETE",
            f"{_BASE_PATH}/{file_id}",
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        _check_by_id_response(
            resp.http_response,
            raise_on_error=self._client.raise_on_error,
        )
        return with_headers(
            _parse_delete(
                resp.json(),
                strict=self._client.config.strict_validation,
                response=resp.http_response,
            ),
            resp.http_response,
        )
