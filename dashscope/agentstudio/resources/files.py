# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Files resource class."""

from __future__ import annotations

import os
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    IO,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from dashscope.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
    build_page,
)
from dashscope.agentstudio.resources._helpers import (
    _coerce_file,
)
from dashscope.agentstudio.types import DeleteResponse, File
from dashscope.agentstudio.types.params import FileListParams


_PATH_FILES = "/files"

# (bytes_read, total_bytes_or_-1) -- called after every chunk.
ProgressCallback = Callable[[int, int], None]


class FileContent(bytes):
    """Binary content of a downloaded file.

    A ``bytes`` subclass so it drops into any code expecting raw bytes,
    with a :meth:`write_to_file` helper that mirrors the Anthropic SDK
    (``client.beta.files.download(...).write_to_file(path)``).
    """

    def write_to_file(
        self,
        path: Union[str, "os.PathLike[str]"],
    ) -> Path:
        """Write the content to ``path`` and return it.

        Missing parent directories are created.
        """
        dest = Path(os.fspath(path))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self)
        return dest


def _open_file(
    file: Union[str, "os.PathLike[str]", BinaryIO, Tuple[str, BinaryIO]],
) -> Tuple[str, BinaryIO, bool]:
    """Return ``(filename, fileobj, must_close)``."""
    if isinstance(file, tuple) and len(file) == 2:
        name, obj = file
        return str(name), obj, False
    if isinstance(file, (str, bytes, os.PathLike)):
        path = Path(os.fspath(file))
        return path.name, path.open("rb"), True
    name = getattr(file, "name", "upload.bin")
    return os.path.basename(str(name)), file, False


class _ProgressFile:
    """Wrap a file-like object so callers can observe upload progress."""

    def __init__(
        self,
        fileobj: IO[bytes],
        total: int,
        callback: ProgressCallback,
    ) -> None:
        self._fileobj = fileobj
        self._total = total
        self._read = 0
        self._callback = callback

    def __len__(self) -> int:
        return self._total

    def read(self, size: int = -1) -> bytes:
        chunk = self._fileobj.read(size)
        if chunk:
            self._read += len(chunk)
            try:
                self._callback(self._read, self._total)
            except Exception:
                pass
        return chunk

    def seek(self, offset: int, whence: int = 0) -> int:
        result = self._fileobj.seek(offset, whence)
        if whence == 0:
            self._read = offset
        elif whence == 1:
            self._read += offset
        elif whence == 2:
            self._read = self._total + offset
        return result

    def tell(self) -> int:
        return self._fileobj.tell()

    @property
    def name(self) -> str:
        return getattr(self._fileobj, "name", "upload.bin")


def _file_size(fileobj: IO[bytes]) -> int:
    try:
        cur = fileobj.tell()
    except (AttributeError, OSError):
        return -1
    try:
        fileobj.seek(0, os.SEEK_END)
        end = fileobj.tell()
        fileobj.seek(cur)
        return end - cur
    except (AttributeError, OSError):
        return -1


class Files:
    """File upload / download / list / delete."""

    def __init__(self, client) -> None:
        self._client = client

    def upload(
        self,
        file: Union[str, "os.PathLike[str]", BinaryIO, Tuple[str, BinaryIO]],
        *,
        mime_type: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> File:
        filename, fileobj, must_close = _open_file(file)
        try:
            payload_obj: Any
            if progress is not None:
                size = _file_size(fileobj)
                if size > 0:
                    payload_obj = _ProgressFile(fileobj, size, progress)
                else:
                    payload_obj = fileobj
            else:
                payload_obj = fileobj
            multipart: Dict[str, Any] = {
                "file": (
                    filename,
                    payload_obj,
                    mime_type or "application/octet-stream",
                ),
            }
            resp = self._client.transport.request(
                "POST",
                _PATH_FILES,
                files=multipart,
                extra_headers=extra_headers,
            )
            return _coerce_file(resp.data)
        finally:
            if must_close:
                try:
                    fileobj.close()
                except Exception:
                    pass

    def retrieve(self, file_id: str) -> File:
        resp = self._client.transport.request(
            "GET",
            f"{_PATH_FILES}/{file_id}",
        )
        return _coerce_file(resp.data)

    # Alias: get() delegates to retrieve()
    get = retrieve  # type: ignore[assignment]

    def _open_content(self, file_id: str, timeout: Optional[float]):
        """GET the content endpoint and return the streaming response."""
        return self._client.transport.request(
            "GET",
            f"{_PATH_FILES}/{file_id}/content",
            extra_headers={"Accept": "*/*"},
            stream=True,
            timeout=timeout,
        )

    def download(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
    ) -> FileContent:
        """Return the file content as a :class:`FileContent`.

        Only files whose ``downloadable`` flag is true can be fetched; the
        service answers 403 otherwise. Use :meth:`FileContent.write_to_file`
        to persist the bytes to disk.

            content = client.files.download("file_xxx")
            content.write_to_file("output.txt")
        """
        resp = self._open_content(file_id, timeout)
        try:
            resp.read()
            return FileContent(resp.content)
        finally:
            resp.close()

    def list(
        self,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> CursorPage[File]:
        params = FileListParams(
            limit=limit,
            page=page,
            scope_id=scope_id,
        ).to_dict()
        resp = self._client.transport.request(
            "GET",
            _PATH_FILES,
            params=params,
        )

        def fetch_next(token: str) -> CursorPage[File]:
            return self.list(
                limit=limit,
                page=token,
                scope_id=scope_id,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_file,
            request_id=resp.request_id,
            page_cls=CursorPage,
            fetch_next=fetch_next,
        )

    def delete(self, file_id: str) -> DeleteResponse:
        resp = self._client.transport.request(
            "DELETE",
            f"{_PATH_FILES}/{file_id}",
        )
        return DeleteResponse(**resp.data)


class AsyncFiles:
    """Async file upload / download / list / delete."""

    def __init__(self, client) -> None:
        self._client = client

    async def upload(
        self,
        file: Union[str, "os.PathLike[str]", BinaryIO, Tuple[str, BinaryIO]],
        *,
        mime_type: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> File:
        filename, fileobj, must_close = _open_file(file)
        try:
            payload_obj: Any
            if progress is not None:
                size = _file_size(fileobj)
                if size > 0:
                    payload_obj = _ProgressFile(fileobj, size, progress)
                else:
                    payload_obj = fileobj
            else:
                payload_obj = fileobj
            multipart: Dict[str, Any] = {
                "file": (
                    filename,
                    payload_obj,
                    mime_type or "application/octet-stream",
                ),
            }
            resp = await self._client.transport.request(
                "POST",
                _PATH_FILES,
                files=multipart,
                extra_headers=extra_headers,
            )
            return _coerce_file(resp.data)
        finally:
            if must_close:
                try:
                    fileobj.close()
                except Exception:
                    pass

    async def retrieve(self, file_id: str) -> File:
        resp = await self._client.transport.request(
            "GET",
            f"{_PATH_FILES}/{file_id}",
        )
        return _coerce_file(resp.data)

    # Alias: get() delegates to retrieve()
    get = retrieve  # type: ignore[assignment]

    async def _open_content(self, file_id: str, timeout: Optional[float]):
        """GET the content endpoint and return the streaming response."""
        return await self._client.transport.request(
            "GET",
            f"{_PATH_FILES}/{file_id}/content",
            extra_headers={"Accept": "*/*"},
            stream=True,
            timeout=timeout,
        )

    async def download(
        self,
        file_id: str,
        *,
        timeout: Optional[float] = None,
    ) -> FileContent:
        """Return the file content as a :class:`FileContent`.

        Only files whose ``downloadable`` flag is true can be fetched; the
        service answers 403 otherwise. Use :meth:`FileContent.write_to_file`
        to persist the bytes to disk.

            content = await client.files.download("file_xxx")
            content.write_to_file("output.txt")
        """
        resp = await self._open_content(file_id, timeout)
        try:
            await resp.aread()
            return FileContent(resp.content)
        finally:
            await resp.aclose()

    async def list(
        self,
        *,
        limit: Optional[int] = None,
        page: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> AsyncCursorPage[File]:
        params = FileListParams(
            limit=limit,
            page=page,
            scope_id=scope_id,
        ).to_dict()
        resp = await self._client.transport.request(
            "GET",
            _PATH_FILES,
            params=params,
        )

        async def fetch_next(token: str) -> AsyncCursorPage[File]:
            return await self.list(
                limit=limit,
                page=token,
                scope_id=scope_id,
            )

        return build_page(
            payload=resp.data,
            item_factory=_coerce_file,
            request_id=resp.request_id,
            page_cls=AsyncCursorPage,
            fetch_next=fetch_next,
        )

    async def delete(self, file_id: str) -> DeleteResponse:
        resp = await self._client.transport.request(
            "DELETE",
            f"{_PATH_FILES}/{file_id}",
        )
        return DeleteResponse(**resp.data)
