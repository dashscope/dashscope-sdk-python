# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""client.batches -- OpenAI-compatible Batch workflow.

Two planes share the ``Batches`` resource:

* ``client.batches.files`` (B1) -- the compatible-mode File sub-resource:
  ``upload`` (purpose=batch), ``content`` download, cursor-paginated
  ``list``, ``get`` and ``delete``.
* ``client.batches`` (B2) -- the batch job plane: ``create`` / ``get`` /
  ``list`` / ``cancel`` / ``wait`` plus a pre-upload ``validate_input``
  row-level JSONL check.
* ``client.batches.download_output`` / ``download_error`` (B3) -- resolve a
  batch's ``output_file_id`` / ``error_file_id`` and download the results
  JSONL through ``files.content`` (``None`` when the file is absent). An
  uploaded input file id is reusable across creates (no re-upload needed);
  discover existing batch files with ``files.list(purpose="batch")``.

Wire: every call sits under ``{gateway}/compatible-mode/v1`` -- the
OpenAI-compatible File/Batch API, NOT the native ``/api/v1/files`` that
``client.files`` uses for fine-tune. The two differ in multipart field name
(``file`` vs ``files``), response shape (a flat OpenAI object vs
``data.uploaded_files``) and pagination (an ``after``/``has_more`` cursor vs
``page_no``/``total``). Like ``client.decision``, this resource derives its
own base URL via ``compatible_mode_base_url`` so the chat ``protocol``
setting cannot move it.

``create`` has no server-side idempotency key on the wire, so it defaults
to ``max_retries=0`` (no automatic retry) to avoid creating duplicate
batches; a caller who passes ``max_retries`` explicitly owns that risk.

Contract:
https://help.aliyun.com/zh/model-studio/openai-file-interface and
https://help.aliyun.com/zh/model-studio/batch-interfaces-compatible-with-openai
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
    overload,
)

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr

from dashscope._config import Config, compatible_mode_base_url
from dashscope.errors import (
    DashScopeError,
    error_from_business_code,
    response_error,
)
from dashscope.resources._jobs import (
    DEFAULT_MAX_INTERVAL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_WAIT_TIMEOUT,
)
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
_BATCH_PATH = "/batches"

MODULE = "batches"

# The File list endpoint caps ``limit`` at 2000; the Batch list endpoint
# caps it at 100 (default 20). Different wires -- do not share the bound.
_MAX_LIMIT = 2000
_BATCH_MAX_LIMIT = 100
_MAX_INPUT_FILE_IDS = 20
_MAX_BATCH_ROWS = 50000

# completion_window: integer + h/d, 24h..336h (official Batch contract).
_WINDOW_RE = re.compile(r"^(\d+)([hd])$")
_MIN_WINDOW_HOURS = 24
_MAX_WINDOW_HOURS = 336

# Batch lifecycle statuses (lowercase on the wire, unlike the uppercase
# native Job statuses in ``_jobs``).
VALIDATING = "validating"
IN_PROGRESS = "in_progress"
FINALIZING = "finalizing"
COMPLETED = "completed"
FAILED = "failed"
EXPIRED = "expired"
CANCELLING = "cancelling"
CANCELLED = "cancelled"

_BATCH_TERMINAL = frozenset({COMPLETED, FAILED, EXPIRED, CANCELLED})

__all__ = [
    "AsyncBatchFiles",
    "AsyncBatches",
    "Batch",
    "BatchErrorItem",
    "BatchErrors",
    "BatchFile",
    "BatchFileList",
    "BatchFiles",
    "BatchList",
    "Batches",
    "CANCELLED",
    "CANCELLING",
    "COMPLETED",
    "EXPIRED",
    "FAILED",
    "FINALIZING",
    "FileContent",
    "FileDeletion",
    "IN_PROGRESS",
    "RequestCounts",
    "VALIDATING",
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


class RequestCounts(BaseModel):
    """Per-status request tallies on a batch (``request_counts``)."""

    model_config = ConfigDict(extra="allow")

    total: int = 0
    completed: int = 0
    failed: int = 0


class BatchErrorItem(BaseModel):
    """One row-level error in a batch's ``errors.data`` (OpenAI shape)."""

    model_config = ConfigDict(extra="allow")

    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None
    line: Optional[int] = None


class BatchErrors(BaseModel):
    """The ``errors`` object on a failed batch (``{object, data}``)."""

    model_config = ConfigDict(extra="allow")

    object: Optional[str] = None
    data: List[BatchErrorItem] = []


class Batch(ResponseBase):
    """One batch job (``client.batches`` create/get/list/cancel/wait).

    The flat OpenAI Batch object: ``status`` drives the lifecycle
    (validating -> in_progress -> finalizing -> completed, or
    failed/expired/cancelling/cancelled). Timestamps are Unix seconds
    (``null`` until the stage is reached). :attr:`finished` is true at any
    terminal status; :attr:`succeeded` only at ``completed``.
    """

    id: Optional[str] = None
    object: Optional[str] = None
    endpoint: Optional[str] = None
    errors: Optional[BatchErrors] = None
    input_file_id: Optional[str] = None
    completion_window: Optional[str] = None
    status: Optional[str] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    created_at: Optional[int] = None
    in_progress_at: Optional[int] = None
    expires_at: Optional[int] = None
    finalizing_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    expired_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    request_counts: Optional[RequestCounts] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def batch_id(self) -> Optional[str]:
        return self.id

    @property
    def finished(self) -> bool:
        return self.status in _BATCH_TERMINAL

    @property
    def succeeded(self) -> bool:
        return self.status == COMPLETED


_PageT = TypeVar("_PageT", bound="_CursorPage")


class _CursorPage(ResponseBase):
    """Shared cursor-page traversal for compatible-mode list endpoints.

    The compatible-mode wire paginates by cursor (``has_more`` + ``after``
    = the last id), not by ``page_no``/``total`` like
    :class:`~dashscope.types.pagination.BasePage`. This base holds the
    traversal ergonomics (``iter_pages`` / ``next_page`` / async twins);
    subclasses supply the typed ``data`` list, ``items`` and ``_next_cursor``.
    The ``self: _PageT`` annotations keep ``next_page`` returning the
    concrete subclass type.
    """

    data: List[Any] = []
    has_more: bool = False

    # Fetches the next page as ``(after_cursor) -> page`` (a page or an
    # awaitable of one). A PrivateAttr so it stays out of ``model_dump()``.
    _fetch_next: Optional[Callable[[str], Any]] = PrivateAttr(default=None)

    @property
    def items(self) -> List[Any]:
        return self.data

    def bind_pagination(
        self: _PageT,
        callback: Optional[Callable[[str], Any]],
    ) -> _PageT:
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

    def next_page(self: _PageT) -> Optional[_PageT]:
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

    async def next_page_async(self: _PageT) -> Optional[_PageT]:
        if not self.has_next_page():
            return None
        assert self._fetch_next is not None and self._next_cursor is not None
        result = self._fetch_next(self._next_cursor)
        if inspect.isawaitable(result):
            result = await result
        return result

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data)

    def __aiter__(self) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            for item in self.data:
                yield item

        return _gen()

    def iter_pages(self: _PageT) -> Iterator[_PageT]:
        page: Optional[_PageT] = self
        while page is not None:
            yield page
            page = page.next_page()

    async def iter_pages_async(self: _PageT) -> AsyncIterator[_PageT]:
        page: Optional[_PageT] = self
        while page is not None:
            yield page
            page = await page.next_page_async()


class BatchFileList(_CursorPage):
    """One cursor page of files, from ``files.list``.

    ``has_more`` says whether another page exists and ``after`` (the last
    ``id`` on this page) fetches it; see :class:`_CursorPage` for traversal.
    """

    data: List[BatchFile] = []
    object: Optional[str] = None

    @property
    def items(self) -> List[BatchFile]:
        return self.data

    def __iter__(self) -> Iterator[BatchFile]:
        return iter(self.data)

    def __aiter__(self) -> AsyncIterator[BatchFile]:
        async def _gen() -> AsyncIterator[BatchFile]:
            for item in self.data:
                yield item

        return _gen()


class BatchList(_CursorPage):
    """One cursor page of batch jobs, from ``batches.list``.

    Like :class:`BatchFileList` but the wire also carries ``first_id`` /
    ``last_id``; the next-page cursor prefers ``last_id`` (the doc's
    ``after`` = "last task id of the previous page").
    """

    data: List[Batch] = []
    object: Optional[str] = None
    first_id: Optional[str] = None
    last_id: Optional[str] = None

    @property
    def items(self) -> List[Batch]:
        return self.data

    @property
    def _next_cursor(self) -> Optional[str]:
        if self.last_id:
            return self.last_id
        return self.data[-1].id if self.data else None

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.data)

    def __aiter__(self) -> AsyncIterator[Batch]:
        async def _gen() -> AsyncIterator[Batch]:
            for item in self.data:
                yield item

        return _gen()


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


def _parse_batch(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> Batch:
    payload = dict(body) if isinstance(body, dict) else {}
    return parse_model(
        Batch,
        {**payload, **response_envelope(payload)},
        strict=strict,
        response=response,
    )


def _parse_batch_list(
    body: Dict[str, Any],
    *,
    strict: bool = False,
    response: Optional["httpx.Response"] = None,
) -> BatchList:
    payload = dict(body) if isinstance(body, dict) else {}
    data = payload.get("data")
    if not isinstance(data, list):
        data = []
    return parse_model(
        BatchList,
        {
            "data": data,
            "object": payload.get("object"),
            "first_id": payload.get("first_id"),
            "last_id": payload.get("last_id"),
            "has_more": bool(payload.get("has_more", False)),
            **response_envelope(payload),
        },
        strict=strict,
        response=response,
    )


def _require_batch_id(batch_id: str) -> None:
    if not batch_id or not batch_id.strip():
        raise ValueError("batch_id must be a non-empty string")


def _batch_id(batch: Union[Batch, str]) -> str:
    batch_id = batch.batch_id if isinstance(batch, Batch) else batch
    if not batch_id:
        raise DashScopeError("batch id required")
    return batch_id


def _result_file_id(batch: Batch, attr: str) -> Optional[str]:
    # output_file_id / error_file_id stay null until the batch produces
    # results (error_file_id only when some row failed); treat blank as
    # absent so the download helpers return None instead of GETting "".
    value = getattr(batch, attr, None)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _validate_completion_window(window: str) -> None:
    match = _WINDOW_RE.match(window or "")
    if match is None:
        raise ValueError(
            "completion_window must be an integer followed by 'h' or 'd' "
            "(e.g. '24h', '7d')",
        )
    amount, unit = int(match.group(1)), match.group(2)
    hours = amount * 24 if unit == "d" else amount
    if not _MIN_WINDOW_HOURS <= hours <= _MAX_WINDOW_HOURS:
        raise ValueError(
            f"completion_window must be between {_MIN_WINDOW_HOURS}h and "
            f"{_MAX_WINDOW_HOURS}h (got {window!r})",
        )


def _build_batch_create_body(
    *,
    input_file_id: str,
    endpoint: str,
    completion_window: str,
    metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not input_file_id or not input_file_id.strip():
        raise ValueError("input_file_id must be a non-empty string")
    if not endpoint or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty string")
    _validate_completion_window(completion_window)
    body: Dict[str, Any] = {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    if metadata is not None:
        body["metadata"] = dict(metadata)
    return body


def _csv_param(value: Union[str, Sequence[str]]) -> str:
    if isinstance(value, str):
        return value
    return ",".join(value)


def _batch_list_params(
    *,
    after: Optional[str],
    limit: Optional[int],
    ds_name: Optional[str],
    input_file_ids: Union[str, Sequence[str], None],
    status: Union[str, Sequence[str], None],
    create_after: Optional[str],
    create_before: Optional[str],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if after is not None:
        params["after"] = after
    if limit is not None:
        if not 1 <= limit <= _BATCH_MAX_LIMIT:
            raise ValueError(
                f"limit must be between 1 and {_BATCH_MAX_LIMIT}",
            )
        params["limit"] = limit
    if ds_name is not None:
        params["ds_name"] = ds_name
    if input_file_ids is not None:
        ids = (
            input_file_ids.split(",")
            if isinstance(input_file_ids, str)
            else list(input_file_ids)
        )
        if len(ids) > _MAX_INPUT_FILE_IDS:
            raise ValueError(
                f"input_file_ids accepts at most {_MAX_INPUT_FILE_IDS} ids",
            )
        params["input_file_ids"] = ",".join(ids)
    if status is not None:
        params["status"] = _csv_param(status)
    if create_after is not None:
        params["create_after"] = create_after
    if create_before is not None:
        params["create_before"] = create_before
    return params


def _validate_input_rows(
    file_path: str,
    *,
    endpoint: Optional[str] = None,
) -> int:
    """Row-level validation of a batch input JSONL file (pre-upload).

    Each non-blank line must be a JSON object with a unique non-empty
    ``custom_id``, ``method="POST"``, a non-empty ``url`` (equal to
    ``endpoint`` when given) and a ``body`` object carrying a non-empty
    ``model``; all rows must use the same model. Returns the row count.
    Local file I/O only -- no network call.
    """
    if not os.path.isfile(file_path):
        raise _io_error(FileNotFoundError(f"file not found: {file_path}"))
    seen: Set[str] = set()
    model: Optional[str] = None
    count = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                count += 1
                if count > _MAX_BATCH_ROWS:
                    raise ValueError(
                        f"batch input exceeds {_MAX_BATCH_ROWS} rows",
                    )
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(
                        f"line {lineno}: not valid JSON ({exc})",
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"line {lineno}: each row must be a JSON object",
                    )
                custom_id = row.get("custom_id")
                if not isinstance(custom_id, str) or not custom_id.strip():
                    raise ValueError(
                        f"line {lineno}: custom_id must be a non-empty string",
                    )
                if custom_id in seen:
                    raise ValueError(
                        f"line {lineno}: duplicate custom_id {custom_id!r}",
                    )
                seen.add(custom_id)
                if row.get("method") != "POST":
                    raise ValueError(f"line {lineno}: method must be 'POST'")
                url = row.get("url")
                if not isinstance(url, str) or not url.strip():
                    raise ValueError(
                        f"line {lineno}: url must be a non-empty string",
                    )
                if endpoint is not None and url != endpoint:
                    raise ValueError(
                        f"line {lineno}: url {url!r} does not match "
                        f"endpoint {endpoint!r}",
                    )
                body = row.get("body")
                if not isinstance(body, dict):
                    raise ValueError(
                        f"line {lineno}: body must be a JSON object",
                    )
                row_model = body.get("model")
                if not isinstance(row_model, str) or not row_model.strip():
                    raise ValueError(
                        f"line {lineno}: body.model must be a non-empty string",
                    )
                if model is None:
                    model = row_model
                elif row_model != model:
                    raise ValueError(
                        f"line {lineno}: all rows must use the same model "
                        f"(expected {model!r}, got {row_model!r})",
                    )
    except OSError as exc:
        raise _io_error(exc) from exc
    if count == 0:
        raise ValueError(f"file {file_path} has no batch input rows")
    return count


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

    ``files`` is the compatible-mode File sub-resource (B1); the methods
    here are the batch job plane (B2): create/get/list/cancel/wait plus a
    pre-upload ``validate_input`` row-level JSONL check.
    """

    def __init__(self, client) -> None:
        self._client = client
        self.files = BatchFiles(client)

    @overload
    def create(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Batch:
        ...

    @overload
    def create(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: Optional[Mapping[str, Any]] = None,
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

    def create(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Batch, "APIResponse"]:
        """Create a batch job (``POST /batches``).

        ``input_file_id`` is an uploaded batch file id (or OSS URL);
        ``endpoint`` must match every row's ``url`` in that file
        (``/v1/chat/completions``, ``/v1/embeddings`` or
        ``/v1/chat/ds-test``); ``completion_window`` is 24h..336h.

        The wire has **no idempotency key**, so this defaults to
        ``max_retries=0`` (no automatic retry) -- a retried create could
        otherwise spawn a duplicate batch. Passing ``max_retries``
        explicitly re-enables retry and the caller owns that duplicate
        risk. Validate the input first with :meth:`validate_input`.

        ``input_file_id`` is reusable: an already-uploaded batch file id
        (``file-batch-xxx``) can drive many creates without re-uploading;
        find existing ones with ``files.list(purpose="batch")``.
        """
        body = _build_batch_create_body(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
        )
        resp = self._client.transport.request_api(
            "POST",
            _BATCH_PATH,
            json=body,
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=0 if max_retries is None else max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(self._client, resp, Batch, _parse_batch)

    @overload
    def get(
        self,
        batch_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Batch:
        ...

    @overload
    def get(
        self,
        batch_id: str,
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
        batch_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Batch, "APIResponse"]:
        """Fetch one batch job (``GET /batches/{id}``).

        Only tasks created within the last 30 days are retrievable.
        """
        _require_batch_id(batch_id)
        resp = self._client.transport.request_api(
            "GET",
            f"{_BATCH_PATH}/{batch_id}",
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
        return _checked(self._client, resp, Batch, _parse_batch)

    @overload
    def list(  # pylint: disable=redefined-builtin
        self,
        *,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        ds_name: Optional[str] = None,
        input_file_ids: Union[str, Sequence[str], None] = None,
        status: Union[str, Sequence[str], None] = None,
        create_after: Optional[str] = None,
        create_before: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> BatchList:
        ...

    @overload
    def list(  # pylint: disable=redefined-builtin
        self,
        *,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        ds_name: Optional[str] = None,
        input_file_ids: Union[str, Sequence[str], None] = None,
        status: Union[str, Sequence[str], None] = None,
        create_after: Optional[str] = None,
        create_before: Optional[str] = None,
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
        after: Optional[str] = None,
        limit: Optional[int] = None,
        ds_name: Optional[str] = None,
        input_file_ids: Union[str, Sequence[str], None] = None,
        status: Union[str, Sequence[str], None] = None,
        create_after: Optional[str] = None,
        create_before: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[BatchList, "APIResponse"]:
        """List batch jobs, one cursor page at a time (``GET /batches``).

        ``after`` is the last id of the previous page; ``limit`` is in
        [1, 100] (default 20). ``input_file_ids`` / ``status`` accept a
        comma-separated string or a sequence (joined for the wire);
        ``input_file_ids`` is capped at 20. ``create_after`` /
        ``create_before`` are ``yyyyMMddHHmmss``. Cross pages with
        ``iter_pages()`` / ``next_page()`` on the parsed result.
        """
        params = _batch_list_params(
            after=after,
            limit=limit,
            ds_name=ds_name,
            input_file_ids=input_file_ids,
            status=status,
            create_after=create_after,
            create_before=create_before,
        )
        resp = self._client.transport.request_api(
            "GET",
            _BATCH_PATH,
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
        result = _checked(self._client, resp, BatchList, _parse_batch_list)
        if not result.is_error:
            result.bind_pagination(
                lambda cursor: self.list(
                    after=cursor,
                    limit=limit,
                    ds_name=ds_name,
                    input_file_ids=input_file_ids,
                    status=status,
                    create_after=create_after,
                    create_before=create_before,
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
    def cancel(
        self,
        batch_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Batch:
        ...

    @overload
    def cancel(
        self,
        batch_id: str,
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

    def cancel(
        self,
        batch_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Batch, "APIResponse"]:
        """Cancel a batch job (``POST /batches/{id}/cancel``).

        Moves the job to ``cancelling`` then ``cancelled``; work already
        done is still billed. Returns the updated :class:`Batch`.
        """
        _require_batch_id(batch_id)
        resp = self._client.transport.request_api(
            "POST",
            f"{_BATCH_PATH}/{batch_id}/cancel",
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
        return _checked(self._client, resp, Batch, _parse_batch)

    def wait(
        self,
        batch: Union[Batch, str],
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        _sleep: Optional[Callable[[float], Any]] = None,
    ) -> Batch:
        """Poll until the batch reaches a terminal status.

        Terminal: completed/failed/expired/cancelled. With
        ``raise_on_error=True`` a ``failed`` batch raises (carrying its
        ``errors`` detail) and ``cancelled``/``expired`` raise a
        :class:`DashScopeError`; ``completed`` returns the batch. With
        ``raise_on_error=False`` a transport-level failure comes back as
        the :class:`Batch` envelope instead of raising. ``_sleep`` is a
        test seam -- it replaces the pause between polls.
        """
        batch_id = _batch_id(batch)
        sleep_fn = _sleep or time.sleep
        deadline = time.monotonic() + timeout
        current = (
            batch
            if isinstance(batch, Batch)
            else self.get(
                batch_id,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            )
        )
        interval = poll_interval
        while not current.finished:
            if current.is_error:
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for batch {batch_id}",
                )
            sleep_fn(min(interval, remaining))
            interval = min(interval * 2, max_interval)
            current = self.get(
                batch_id,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            )
        return self._check_terminal(current, batch_id)

    def _check_terminal(self, batch: Batch, batch_id: str) -> Batch:
        if batch.status == FAILED and self._client.raise_on_error:
            detail = None
            if batch.errors and batch.errors.data:
                detail = batch.errors.data[0].message
            raise error_from_business_code(
                message=batch.message or detail or f"batch {batch_id} failed",
                code=batch.code,
                # ``headers`` is transport metadata, not wire body.
                body=batch.model_dump(exclude={"headers"}),
                headers=batch.headers,
            )
        if (
            batch.status in (CANCELLED, EXPIRED)
            and self._client.raise_on_error
        ):
            raise DashScopeError(
                f"batch {batch_id} was {batch.status} while waiting",
                request_id=batch.request_id,
            )
        return batch

    def validate_input(
        self,
        file_path: str,
        *,
        endpoint: Optional[str] = None,
    ) -> int:
        """Validate a batch input JSONL file row by row (pre-upload).

        Local file I/O only -- no network call, so this is a plain method
        on both the sync and async clients. Checks each row's
        ``custom_id`` (unique, non-empty), ``method="POST"``, ``url``
        (matching ``endpoint`` when given) and ``body.model`` (non-empty,
        same across all rows). Returns the row count; raises
        ``ValueError`` with the offending line number on the first
        violation. Run this before ``files.upload`` + ``create``.
        """
        return _validate_input_rows(file_path, endpoint=endpoint)

    def download_output(
        self,
        batch: Union[Batch, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> Optional[FileContent]:
        """Download a batch's successful-results JSONL (``output_file_id``).

        Accepts a :class:`Batch` (used as-is) or a batch id (fetched via
        :meth:`get`). Resolves ``output_file_id`` and downloads it through
        ``files.content`` (``GET /files/{id}/content``), returning
        :class:`FileContent` (bytes with ``text`` / ``write_to_file``).
        Returns ``None`` when the batch has no output file yet (not
        ``completed``, or an error envelope under ``raise_on_error=False``).
        Each output row is ``{id, custom_id, response:{status_code,
        request_id, body}, error}``. Call :meth:`wait` first for a terminal
        batch. High-level helper -- no ``raw`` overload; use
        ``files.content(file_id, raw=True)`` for the raw response.
        """
        return self._download_result(
            batch,
            "output_file_id",
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
        )

    def download_error(
        self,
        batch: Union[Batch, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> Optional[FileContent]:
        """Download a batch's failed-requests JSONL (``error_file_id``).

        Like :meth:`download_output` but for ``error_file_id``; returns
        ``None`` when no row failed (the common case for a clean batch).
        """
        return self._download_result(
            batch,
            "error_file_id",
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
        )

    def _download_result(
        self,
        batch: Union[Batch, str],
        attr: str,
        *,
        timeout: Optional[float],
        api_key: Optional[str],
        workspace: Optional[str],
        extra_headers: Optional[Mapping[str, str]],
        extra_query: Optional[Mapping[str, Any]],
        extra_body: Optional[Mapping[str, Any]],
        max_retries: Optional[int],
    ) -> Optional[FileContent]:
        batch_id = _batch_id(batch)
        current = (
            batch
            if isinstance(batch, Batch)
            else self.get(
                batch_id,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            )
        )
        file_id = _result_file_id(current, attr)
        if file_id is None:
            return None
        return self.files.content(
            file_id,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
        )


class AsyncBatches:
    """Async twin of :class:`Batches`."""

    def __init__(self, client) -> None:
        self._client = client
        self.files = AsyncBatchFiles(client)

    @overload
    async def create(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Batch:
        ...

    @overload
    async def create(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: Optional[Mapping[str, Any]] = None,
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

    async def create(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Batch, "APIResponse"]:
        """Async twin of :meth:`Batches.create` (defaults ``max_retries=0``)."""
        body = _build_batch_create_body(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
        )
        resp = await self._client.transport.request_api(
            "POST",
            _BATCH_PATH,
            json=body,
            base_url=_endpoint_base(self._client.config),
            sdk_module=MODULE,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            options=request_options(
                extra_headers,
                extra_query,
                extra_body,
                max_retries=0 if max_retries is None else max_retries,
            ),
            client=self._client,
        )
        if raw:
            return resp
        return _checked(self._client, resp, Batch, _parse_batch)

    @overload
    async def get(
        self,
        batch_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Batch:
        ...

    @overload
    async def get(
        self,
        batch_id: str,
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
        batch_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Batch, "APIResponse"]:
        """Async twin of :meth:`Batches.get`."""
        _require_batch_id(batch_id)
        resp = await self._client.transport.request_api(
            "GET",
            f"{_BATCH_PATH}/{batch_id}",
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
        return _checked(self._client, resp, Batch, _parse_batch)

    @overload
    async def list(  # pylint: disable=redefined-builtin
        self,
        *,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        ds_name: Optional[str] = None,
        input_file_ids: Union[str, Sequence[str], None] = None,
        status: Union[str, Sequence[str], None] = None,
        create_after: Optional[str] = None,
        create_before: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> BatchList:
        ...

    @overload
    async def list(  # pylint: disable=redefined-builtin
        self,
        *,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        ds_name: Optional[str] = None,
        input_file_ids: Union[str, Sequence[str], None] = None,
        status: Union[str, Sequence[str], None] = None,
        create_after: Optional[str] = None,
        create_before: Optional[str] = None,
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
        after: Optional[str] = None,
        limit: Optional[int] = None,
        ds_name: Optional[str] = None,
        input_file_ids: Union[str, Sequence[str], None] = None,
        status: Union[str, Sequence[str], None] = None,
        create_after: Optional[str] = None,
        create_before: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[BatchList, "APIResponse"]:
        """Async twin of :meth:`Batches.list`."""
        params = _batch_list_params(
            after=after,
            limit=limit,
            ds_name=ds_name,
            input_file_ids=input_file_ids,
            status=status,
            create_after=create_after,
            create_before=create_before,
        )
        resp = await self._client.transport.request_api(
            "GET",
            _BATCH_PATH,
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
        result = _checked(self._client, resp, BatchList, _parse_batch_list)
        if not result.is_error:
            result.bind_pagination(
                lambda cursor: self.list(
                    after=cursor,
                    limit=limit,
                    ds_name=ds_name,
                    input_file_ids=input_file_ids,
                    status=status,
                    create_after=create_after,
                    create_before=create_before,
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
    async def cancel(
        self,
        batch_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: Literal[False] = False,
    ) -> Batch:
        ...

    @overload
    async def cancel(
        self,
        batch_id: str,
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

    async def cancel(
        self,
        batch_id: str,
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        raw: bool = False,
    ) -> Union[Batch, "APIResponse"]:
        """Async twin of :meth:`Batches.cancel`."""
        _require_batch_id(batch_id)
        resp = await self._client.transport.request_api(
            "POST",
            f"{_BATCH_PATH}/{batch_id}/cancel",
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
        return _checked(self._client, resp, Batch, _parse_batch)

    async def wait(
        self,
        batch: Union[Batch, str],
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        max_interval: float = DEFAULT_MAX_INTERVAL,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
        _sleep: Optional[Callable[[float], Awaitable[None]]] = None,
    ) -> Batch:
        """Async twin of :meth:`Batches.wait`."""
        batch_id = _batch_id(batch)
        sleep_fn = _sleep or asyncio.sleep
        deadline = time.monotonic() + timeout
        current = (
            batch
            if isinstance(batch, Batch)
            else await self.get(
                batch_id,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            )
        )
        interval = poll_interval
        while not current.finished:
            if current.is_error:
                return current
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DashScopeError(
                    f"timed out after {timeout}s waiting for batch {batch_id}",
                )
            await sleep_fn(min(interval, remaining))
            interval = min(interval * 2, max_interval)
            current = await self.get(
                batch_id,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            )
        return self._check_terminal(current, batch_id)

    def _check_terminal(self, batch: Batch, batch_id: str) -> Batch:
        if batch.status == FAILED and self._client.raise_on_error:
            detail = None
            if batch.errors and batch.errors.data:
                detail = batch.errors.data[0].message
            raise error_from_business_code(
                message=batch.message or detail or f"batch {batch_id} failed",
                code=batch.code,
                body=batch.model_dump(exclude={"headers"}),
                headers=batch.headers,
            )
        if (
            batch.status in (CANCELLED, EXPIRED)
            and self._client.raise_on_error
        ):
            raise DashScopeError(
                f"batch {batch_id} was {batch.status} while waiting",
                request_id=batch.request_id,
            )
        return batch

    def validate_input(
        self,
        file_path: str,
        *,
        endpoint: Optional[str] = None,
    ) -> int:
        """Local row-level JSONL validation; see :meth:`Batches.validate_input`.

        A plain (non-async) method: it does local file I/O, no network.
        """
        return _validate_input_rows(file_path, endpoint=endpoint)

    async def download_output(
        self,
        batch: Union[Batch, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> Optional[FileContent]:
        """Async twin of :meth:`Batches.download_output`."""
        return await self._download_result(
            batch,
            "output_file_id",
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
        )

    async def download_error(
        self,
        batch: Union[Batch, str],
        *,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_query: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        max_retries: Optional[int] = None,
    ) -> Optional[FileContent]:
        """Async twin of :meth:`Batches.download_error`."""
        return await self._download_result(
            batch,
            "error_file_id",
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
        )

    async def _download_result(
        self,
        batch: Union[Batch, str],
        attr: str,
        *,
        timeout: Optional[float],
        api_key: Optional[str],
        workspace: Optional[str],
        extra_headers: Optional[Mapping[str, str]],
        extra_query: Optional[Mapping[str, Any]],
        extra_body: Optional[Mapping[str, Any]],
        max_retries: Optional[int],
    ) -> Optional[FileContent]:
        batch_id = _batch_id(batch)
        current = (
            batch
            if isinstance(batch, Batch)
            else await self.get(
                batch_id,
                timeout=timeout,
                api_key=api_key,
                workspace=workspace,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                max_retries=max_retries,
            )
        )
        file_id = _result_file_id(current, attr)
        if file_id is None:
            return None
        return await self.files.content(
            file_id,
            timeout=timeout,
            api_key=api_key,
            workspace=workspace,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            max_retries=max_retries,
        )