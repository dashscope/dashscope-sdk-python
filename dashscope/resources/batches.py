# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.batches -- OpenAI-compatible Batch workflow (file input).

B1 mounts the compatible-mode File sub-resource (``client.batches.files``):
``upload`` (purpose=batch), ``content`` download, cursor-paginated ``list``,
``get`` and ``delete``. The batch job plane (create/get/list/cancel) lands
in B2 on the same ``Batches`` class.

Wire: every call sits under ``{gateway}/compatible-mode/v1`` -- the
OpenAI-compatible File API, NOT the native ``/api/v1/files`` that
``client.files`` uses for fine-tune. The two differ in multipart field name
(``file`` vs ``files``), response shape (a flat OpenAI object vs
``data.uploaded_files``) and pagination (an ``after``/``has_more`` cursor vs
``page_no``/``total``). Like ``client.decision``, this resource derives its
own base URL via ``compatible_mode_base_url`` so the chat ``protocol``
setting cannot move it.

Contract:
https://help.aliyun.com/zh/model-studio/openai-file-interface and
https://help.aliyun.com/zh/model-studio/batch-interfaces-compatible-with-openai
(the ``GET /files/{id}/content`` download).
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    overload,
)

from pydantic import AliasChoices, Field, PrivateAttr

from dashscope._config import Config, compatible_mode_base_url
from dashscope.errors import response_error
from dashscope.resources.files import _io_error, _is_valid_jsonl
from dashscope.transport._options import request_options
from dashscope.types._parse import parse_model
from dashscope.types.common import (
    ResponseBase,
    response_envelope,
    with_headers,
)

if TYPE_CHECKING:
    import httpx

    from dashscope.transport._response import APIResponse

_BASE_PATH = "/files"

MODULE = "batches"

# The list endpoint caps ``limit`` at 2000 (default 2000 when omitted).
_MAX_LIMIT = 2000

__all__ = [
    "AsyncBatchFiles",
    "AsyncBatches",
    "BatchFile",
    "BatchFileList",
    "BatchFiles",
    "Batches",
    "FileContent",
    "FileDeletion",
]


class BatchFile(ResponseBase):
    """One file in the OpenAI-compatible store.

    The flat OpenAI shape comes back from ``upload`` / ``get`` and nested in
    ``BatchFileList.data``. ``id`` is the wire name (``file-batch-xxx`` /
    ``file-fe-xxx``); :attr:`file_id` is a convenience alias so callers can
    pass the same value to ``content`` / ``get`` / ``delete``.
    """

    id: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("id", "file_id"),
    )
    object: Optional[str] = None
    bytes: Optional[int] = None
    # Unix seconds (Integer on the wire), not a formatted string.
    created_at: Optional[int] = None
    filename: Optional[str] = None
    purpose: Optional[str] = None
    status: Optional[str] = None
    status_details: Optional[Any] = None

    @property
    def file_id(self) -> Optional[str]:
        return self.id


class BatchFileList(ResponseBase):
    """One cursor page of files, from ``files.list``.

    The compatible-mode list paginates by cursor: ``has_more`` says whether
    another page exists and ``after`` (the last ``id`` on this page) fetches
    it. This mirrors :class:`~dashscope.types.pagination.BasePage`'s
    traversal ergonomics (``iter_pages`` / ``next_page``) but is driven by
    the cursor rather than ``page_no`` arithmetic, because the wire carries
    no total count.
    """

    data: List[BatchFile] = []
    object: Optional[str] = None
    has_more: bool = False

    # Fetches the next page as ``(after_cursor) -> page`` (a page or an
    # awaitable of one). A PrivateAttr so it stays out of ``model_dump()``.
    _fetch_next: Optional[Callable[[str], Any]] = PrivateAttr(default=None)

    @property
    def items(self) -> List[BatchFile]:
        return self.data

    def bind_pagination(
        self,
        callback: Optional[Callable[[str], Any]],
    ) -> "BatchFileList":
        self._fetch_next = callback
        return self

    @property
    def _next_cursor(self) -> Optional[str]:
        return self.data[-1].id if self.data else None

    def has_next_page(self) -> bool:
        return (
            self.has_more
            and self._fetch_next is not None
            and self._next_cursor is not None
        )

    def next_page(self) -> Optional["BatchFileList"]:
        if not self.has_next_page():
            return None
        assert self._fetch_next is not None and self._next_cursor is not None
        result = self._fetch_next(self._next_cursor)
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if callable(close):
                close()
            raise TypeError(
                "this page was fetched through AsyncClient, so advancing it "
                "is a coroutine; use next_page_async() or iter_pages_async() "
                "instead of the synchronous traversal",
            )
        return result

    async def next_page_async(self) -> Optional["BatchFileList"]:
        if not self.has_next_page():
            return None
        assert self._fetch_next is not None and self._next_cursor is not None
        result = self._fetch_next(self._next_cursor)
        if inspect.isawaitable(result):
            result = await result
        return result

    def __iter__(self) -> Iterator[BatchFile]:
        return iter(self.data)

    def __aiter__(self) -> AsyncIterator[BatchFile]:
        async def _gen() -> AsyncIterator[BatchFile]:
            for item in self.data:
                yield item

        return _gen()

    def iter_pages(self) -> Iterator["BatchFileList"]:
        page: Optional["BatchFileList"] = self
        while page is not None:
            yield page
            page = page.next_page()

    async def iter_pages_async(self) -> AsyncIterator["BatchFileList"]:
        page: Optional["BatchFileList"] = self
        while page is not None:
            yield page
            page = await page.next_page_async()


class FileDeletion(ResponseBase):
    """Result of ``files.delete`` (``{object, deleted, id}``)."""

    id: Optional[str] = None
    object: Optional[str] = None
    deleted: bool = False


class FileContent(bytes):
    """Binary content of a downloaded file (``files.content``).

    A ``bytes`` subclass so it drops into any code expecting raw bytes, with
    a :attr:`text` view (UTF-8) for the JSONL batch results and a
    :meth:`write_to_file` helper. ``status_code`` / :attr:`is_error` let a
    ``raise_on_error=False`` caller inspect a failed download, whose body is
    the gateway's error bytes rather than a parsed envelope.
    """

    request_id: Optional[str]
    headers: Dict[str, str]
    status_code: Optional[int]

    def __new__(
        cls,
        content: bytes = b"",
        *,
        request_id: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        status_code: Optional[int] = None,
    ) -> "FileContent":
        obj = super().__new__(cls, content)
        obj.request_id = request_id
        obj.headers = dict(headers or {})
        obj.status_code = status_code
        return obj

    @property
    def text(self) -> str:
        return self.decode("utf-8")

    @property
    def is_error(self) -> bool:
        return self.status_code is not None and self.status_code >= 400

    def write_to_file(self, path: Union[str, "os.PathLike[str]"]) -> Path:
        dest = Path(os.fspath(path))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self)
        return dest


def _endpoint_base(config: Config) -> str:
    # The compatible-mode File API sits under /compatible-mode/v1 on every
    # gateway; the chat protocol setting must not move it.
    return compatible_mode_base_url(
        config.workspace,
        config.region,
        config.configured_base_url,
    )


def _require_file_id(file_id: str) -> None:
    if not file_id or not file_id.strip():
        raise ValueError("file_id must be a non-empty string")


def _validate_upload(file_path: str, purpose: str) -> None:
    if not os.path.isfile(file_path):
        raise _io_error(FileNotFoundError(f"file not found: {file_path}"))
    # Batch input must be JSONL; file-extract accepts binary documents.
    if purpose == "batch" and not _is_valid_jsonl(file_path):
        raise ValueError(f"file {file_path} is not in valid jsonl format")


def _upload_form(file_path: str, purpose: str):
    try:
        with open(file_path, "rb") as f:
            # Bytes, not a live handle: transport retries must re-send the
            # full body.
            content = f.read()
    except OSError as exc:
        raise _io_error(exc) from exc
    return {"purpose": purpose}, {
        "file": (os.path.basename(file_path), content),
    }


def _list_params(
    *,
    purpose: Optional[str],
    after: Optional[str],
    limit: Optional[int],
    create_before: Optional[str],
    create_after: Optional[str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if purpose is not None:
        params["purpose"] = purpose
    if after is not None:
        params["after"] = after
    if limit is not None:
        if not 1 <= limit <= _MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {_MAX_LIMIT}")
        params["limit"] = limit
    if create_before is not None:
        params["create_before"] = create_before
    if create_after is not None:
        params["create_after"] = create_after
    return params


def _parse_file(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> BatchFile:
    payload = dict(body) if isinstance(body, dict) else {}
    return parse_model(
        BatchFile,
        {**payload, **response_envelope(payload)},
        strict=strict,
        response=response,
    )


def _parse_list(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> BatchFileList:
    payload = dict(body) if isinstance(body, dict) else {}
    data = payload.get("data")
    if not isinstance(data, list):
        data = []
    return parse_model(
        BatchFileList,
        {
            "data": data,
            "object": payload.get("object"),
            "has_more": bool(payload.get("has_more", False)),
            **response_envelope(payload),
        },
        strict=strict,
        response=response,
    )


def _parse_delete(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> FileDeletion:
    payload = dict(body) if isinstance(body, dict) else {}
    return parse_model(
        FileDeletion,
        {**payload, **response_envelope(payload)},
        strict=strict,
        response=response,
    )


def _checked(client, resp, model_cls, parser):
    """Raise on failure, or fill the error envelope when not raising.

    Compatible-mode failures nest ``code`` / ``message`` under ``error``
    (OpenAI shape), which the flat success parser cannot see. Routing the
    reply through ``response_error`` keeps ``raise_on_error=False``
    reporting the failure via ``is_error`` / ``code`` / ``message`` instead
    of surfacing it as an empty success.
    """
    http_resp = resp.http_response
    error = response_error(http_resp)
    if error is None:
        return with_headers(
            parser(
                resp.json(),
                strict=client.config.strict_validation,
                response=http_resp,
            ),
            http_resp,
        )
    if client.raise_on_error:
        raise error
    return with_headers(
        model_cls(
            request_id=error.request_id,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            is_error=True,
        ),
        http_resp,
    )


def _content_result(client, resp) -> FileContent:
    http_resp = resp.http_response
    # Status-first: a successful download is raw bytes (JSONL), not a JSON
    # envelope, so the business-code layer must not parse it as one.
    if http_resp.status_code >= 400:
        error = response_error(http_resp)
        if error is not None and client.raise_on_error:
            raise error
    return FileContent(
        http_resp.content,
        request_id=resp.request_id,
        headers=dict(http_resp.headers),
        status_code=http_resp.status_code,
    )


class BatchFiles:
    """Synchronous compatible-mode File calls (``client.batches.files``)."""

    def __init__(self, client) -> None:
        self._client = client

    @overload
    def upload(
        self,
        *,
        file_path: str,
        purpose: str = "batch",
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> BatchFile:
        ...

    @overload
    def upload(
        self,
        *,
        file_path: str,
        purpose: str = "batch",
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
        purpose: str = "batch",
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[BatchFile, "APIResponse"]:
        """Upload a file to the compatible-mode store.

        ``purpose="batch"`` (the default) requires a JSONL input file;
        ``file-extract`` accepts document/binary files. The multipart field
        is ``file`` (singular), unlike the native ``client.files`` upload.
        """
        _validate_upload(file_path, purpose)
        data, files = _upload_form(file_path, purpose)
        resp = self._client.transport.request_api(
            "POST",
            _BASE_PATH,
            data=data,
            files=files,
            base_url=_endpoint_base(self._client.config),
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
        return _checked(self._client, resp, BatchFile, _parse_file)

    @overload
    def content(
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
    ) -> FileContent:
        ...

    @overload
    def content(
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

    def content(
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
    ) -> Union[FileContent, "APIResponse"]:
        """Download a file's raw content (``GET /files/{id}/content``).

        Used for a batch's ``output_file_id`` / ``error_file_id`` JSONL
        results. Returns :class:`FileContent` (bytes) with a ``text`` view
        and ``write_to_file``.
        """
        _require_file_id(file_id)
        resp = self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/{file_id}/content",
            base_url=_endpoint_base(self._client.config),
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
        return _content_result(self._client, resp)

    @overload
    def list(  # pylint: disable=redefined-builtin
        self,
        *,
        purpose: Optional[str] = None,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        create_before: Optional[str] = None,
        create_after: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> BatchFileList:
        ...

    @overload
    def list(  # pylint: disable=redefined-builtin
        self,
        *,
        purpose: Optional[str] = None,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        create_before: Optional[str] = None,
        create_after: Optional[str] = None,
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
        purpose: Optional[str] = None,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        create_before: Optional[str] = None,
        create_after: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[BatchFileList, "APIResponse"]:
        """List files, one cursor page at a time.

        ``purpose="batch"`` narrows to batch files. ``after`` is the last
        ``id`` of the previous page; ``limit`` is in [1, 2000]. Cross pages
        with ``iter_pages()`` / ``next_page()`` on the parsed result.
        """
        params = _list_params(
            purpose=purpose,
            after=after,
            limit=limit,
            create_before=create_before,
            create_after=create_after,
        )
        resp = self._client.transport.request_api(
            "GET",
            _BASE_PATH,
            params=params,
            base_url=_endpoint_base(self._client.config),
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
        result = _checked(self._client, resp, BatchFileList, _parse_list)
        if not result.is_error:
            result.bind_pagination(
                lambda cursor: self.list(
                    purpose=purpose,
                    after=cursor,
                    limit=limit,
                    create_before=create_before,
                    create_after=create_after,
                    timeout=timeout,
                    api_key=api_key,
                    workspace=workspace,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    max_retries=max_retries,
                ),
            )
        return result

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
    ) -> BatchFile:
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
    ) -> Union[BatchFile, "APIResponse"]:
        """Fetch one file's metadata (``GET /files/{id}``)."""
        _require_file_id(file_id)
        resp = self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/{file_id}",
            base_url=_endpoint_base(self._client.config),
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
        return _checked(self._client, resp, BatchFile, _parse_file)

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
    ) -> FileDeletion:
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
    ) -> Union[FileDeletion, "APIResponse"]:
        """Delete one file (``DELETE /files/{id}``)."""
        _require_file_id(file_id)
        resp = self._client.transport.request_api(
            "DELETE",
            f"{_BASE_PATH}/{file_id}",
            base_url=_endpoint_base(self._client.config),
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
        return _checked(self._client, resp, FileDeletion, _parse_delete)


class AsyncBatchFiles:
    """Asynchronous compatible-mode File calls (``client.batches.files``)."""

    def __init__(self, client) -> None:
        self._client = client

    @overload
    async def upload(
        self,
        *,
        file_path: str,
        purpose: str = "batch",
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> BatchFile:
        ...

    @overload
    async def upload(
        self,
        *,
        file_path: str,
        purpose: str = "batch",
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
        purpose: str = "batch",
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[BatchFile, "APIResponse"]:
        """Async twin of :meth:`BatchFiles.upload`."""
        _validate_upload(file_path, purpose)
        data, files = _upload_form(file_path, purpose)
        resp = await self._client.transport.request_api(
            "POST",
            _BASE_PATH,
            data=data,
            files=files,
            base_url=_endpoint_base(self._client.config),
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
        return _checked(self._client, resp, BatchFile, _parse_file)

    @overload
    async def content(
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
    ) -> FileContent:
        ...

    @overload
    async def content(
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

    async def content(
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
    ) -> Union[FileContent, "APIResponse"]:
        """Async twin of :meth:`BatchFiles.content`."""
        _require_file_id(file_id)
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/{file_id}/content",
            base_url=_endpoint_base(self._client.config),
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
        return _content_result(self._client, resp)

    @overload
    async def list(  # pylint: disable=redefined-builtin
        self,
        *,
        purpose: Optional[str] = None,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        create_before: Optional[str] = None,
        create_after: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> BatchFileList:
        ...

    @overload
    async def list(  # pylint: disable=redefined-builtin
        self,
        *,
        purpose: Optional[str] = None,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        create_before: Optional[str] = None,
        create_after: Optional[str] = None,
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
        purpose: Optional[str] = None,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        create_before: Optional[str] = None,
        create_after: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[BatchFileList, "APIResponse"]:
        """Async twin of :meth:`BatchFiles.list`."""
        params = _list_params(
            purpose=purpose,
            after=after,
            limit=limit,
            create_before=create_before,
            create_after=create_after,
        )
        resp = await self._client.transport.request_api(
            "GET",
            _BASE_PATH,
            params=params,
            base_url=_endpoint_base(self._client.config),
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
        result = _checked(self._client, resp, BatchFileList, _parse_list)
        if not result.is_error:
            result.bind_pagination(
                lambda cursor: self.list(
                    purpose=purpose,
                    after=cursor,
                    limit=limit,
                    create_before=create_before,
                    create_after=create_after,
                    timeout=timeout,
                    api_key=api_key,
                    workspace=workspace,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    max_retries=max_retries,
                ),
            )
        return result

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
    ) -> BatchFile:
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
    ) -> Union[BatchFile, "APIResponse"]:
        """Async twin of :meth:`BatchFiles.get`."""
        _require_file_id(file_id)
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BASE_PATH}/{file_id}",
            base_url=_endpoint_base(self._client.config),
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
        return _checked(self._client, resp, BatchFile, _parse_file)

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
    ) -> FileDeletion:
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
    ) -> Union[FileDeletion, "APIResponse"]:
        """Async twin of :meth:`BatchFiles.delete`."""
        _require_file_id(file_id)
        resp = await self._client.transport.request_api(
            "DELETE",
            f"{_BASE_PATH}/{file_id}",
            base_url=_endpoint_base(self._client.config),
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
        return _checked(self._client, resp, FileDeletion, _parse_delete)


class Batches:
    """OpenAI-compatible Batch workflow (``client.batches``).

    B1 mounts the File sub-resource; the batch job plane
    (create/get/list/cancel) lands in B2.
    """

    def __init__(self, client) -> None:
        self._client = client
        self.files = BatchFiles(client)


class AsyncBatches:
    """Async twin of :class:`Batches`."""

    def __init__(self, client) -> None:
        self._client = client
        self.files = AsyncBatchFiles(client)