# -*- coding: utf-8 -*-
"""File download: content shape, write_to_file, progress.

Download tests use a recording transport that hands back real
``httpx.Response`` objects, so the resource code exercises the same
``iter_bytes`` / ``close`` surface it sees against the service. The
service streams the bytes back directly (verified end-to-end), so the
fixtures return 200 with a body.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import httpx
import pytest

from dashscope.agentstudio import AsyncClient, Client
from dashscope.agentstudio.resources.files import FileContent

_CONTENT = b"0123456789" * 32


def _response(
    content: bytes = _CONTENT,
    *,
    disposition: Optional[str] = None,
) -> httpx.Response:
    headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(content)),
    }
    if disposition is not None:
        headers["Content-Disposition"] = disposition
    return httpx.Response(200, headers=headers, content=content)


class _Tx:
    """Recording transport that returns a fresh response per call."""

    def __init__(self, response_factory=_response):
        self.calls: List[Dict[str, Any]] = []
        self._factory = response_factory

    def request(self, method, path, **kwargs):
        self.calls.append({"method": method, "path": path, **kwargs})
        return self._factory()


class _AsyncTx:
    """Async recording transport that returns a fresh response per call."""

    def __init__(self, response_factory=_response):
        self.calls: List[Dict[str, Any]] = []
        self._factory = response_factory

    async def request(self, method, path, **kwargs):
        self.calls.append({"method": method, "path": path, **kwargs})
        return self._factory()


@pytest.fixture(name="client")
def _client_fixture():
    c = Client(api_key="test-key", base_url="http://test")
    c.transport = _Tx()
    return c


@pytest.fixture(name="async_client")
def _async_client_fixture():
    c = AsyncClient(api_key="test-key", base_url="http://test")
    c.transport = _AsyncTx()
    return c


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_download_requests_content_endpoint(client):
    client.files.download("file_1")

    call = client.transport.calls[0]
    assert call["method"] == "GET"
    assert call["path"] == "/files/file_1/content"
    # Streamed so large files never buffer at the transport.
    assert call["stream"] is True


# ---------------------------------------------------------------------------
# Content + write_to_file
# ---------------------------------------------------------------------------


def test_download_returns_file_content(client):
    content = client.files.download("file_1")

    assert isinstance(content, FileContent)
    # FileContent is a bytes subclass, so it drops into any code
    # expecting raw bytes.
    assert isinstance(content, bytes)
    assert content == _CONTENT


def test_download_write_to_file(client, tmp_path):
    content = client.files.download("file_1")

    written = content.write_to_file(tmp_path / "nested" / "out.bin")

    assert written == tmp_path / "nested" / "out.bin"
    assert written.read_bytes() == _CONTENT


def test_download_write_to_file_creates_missing_parents(client, tmp_path):
    content = client.files.download("file_1")

    written = content.write_to_file(tmp_path / "a" / "b" / "c.bin")

    assert written.read_bytes() == _CONTENT


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


def test_async_download(async_client):
    async def _run():
        return await async_client.files.download("file_1")

    content = asyncio.run(_run())
    assert isinstance(content, FileContent)
    assert content == _CONTENT


def test_async_download_write_to_file(async_client, tmp_path):
    async def _run():
        content = await async_client.files.download("file_1")
        return content.write_to_file(tmp_path / "out.bin")

    written = asyncio.run(_run())
    assert written.read_bytes() == _CONTENT
