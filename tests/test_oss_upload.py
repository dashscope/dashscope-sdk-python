# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the v2-native local-file → OSS uploader."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import asyncio
import os

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.errors import (
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
)
from dashscope.resources import _oss_upload
from dashscope.resources._blocking import run_blocking

BASE_URL = "https://api.test.local/api/v1"
OSS_HOST = "https://dashscope-oss.oss-cn-beijing.aliyuncs.com"

_CERT = {
    "oss_access_key_id": "STS.access-key-id",
    "signature": "sig==",
    "policy": "pol==",
    "upload_dir": "upload/2026/abc",
    "x_oss_object_acl": "private",
    "x_oss_forbid_overwrite": "true",
    "upload_host": OSS_HOST,
}


def _make_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def _make_async_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return AsyncClient(api_key="sk-test", http_client=http_client, **kwargs)


def _policy_handler(calls, status=200):
    def handler(request):
        calls.append(request)
        if status != 200:
            return httpx.Response(status, json={"message": "denied"})
        return httpx.Response(
            200,
            json={"request_id": "r-1", "data": _CERT},
        )

    return handler


def _policy_client(status=200):
    """Client whose transport answers ``GET /uploads`` with a policy."""
    calls = []
    return _make_client(_policy_handler(calls, status=status)), calls


def _capture_oss(monkeypatch, status=200):
    """Send the OSS POST through a MockTransport and record what it carried."""
    seen = {}
    real_post = _oss_upload._post_to_oss

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = request.content.decode("utf-8", "replace")
        if status == 200:
            return httpx.Response(200, text="")
        return httpx.Response(
            status,
            text="<Error><Code>AccessDenied</Code></Error>",
        )

    def patched(upload_host, form, file_name, file_path, transport=None):
        seen["form"] = form
        seen["file_name"] = file_name
        return real_post(
            upload_host,
            form,
            file_name,
            file_path,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(_oss_upload, "_post_to_oss", patched)
    return seen


def _png(tmp_path, name="cat.png"):
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return str(path)


def test_certificate_fetch_rides_the_client_transport():
    client, calls = _policy_client()

    cert = _oss_upload.get_upload_certificate(
        client,
        "qwen-vl-max",
        api_key="sk-override",
        workspace="ws-1",
    )

    assert cert == _CERT
    request = calls[0]
    assert request.url.path == "/api/v1/uploads"
    assert request.url.params["action"] == "getPolicy"
    assert request.url.params["model"] == "qwen-vl-max"
    # This hop talks to DashScope, so it does carry the per-call credential.
    assert request.headers.get("Authorization") == "Bearer sk-override"
    assert request.headers.get("X-DashScope-WorkSpace") == "ws-1"


def test_certificate_fetch_failure_raises():
    client, _ = _policy_client(status=401)
    with pytest.raises(AuthenticationError):
        _oss_upload.get_upload_certificate(client, "qwen-vl-max")


@pytest.mark.parametrize(
    "body,request_id",
    [
        ({"request_id": "r-1"}, "r-1"),
        # "output" is not this endpoint's envelope; accepting it would let
        # an unrelated body through as a policy and fail later on a missing
        # policy field instead of here.
        ({"request_id": "r-1", "output": _CERT}, "r-1"),
        ([], "header-id"),
        ("not-an-object", "header-id"),
    ],
)
def test_certificate_fetch_rejects_an_invalid_policy_body(body, request_id):
    client = _make_client(
        lambda request: httpx.Response(
            200,
            json=body,
            headers={"x-request-id": "header-id"},
        ),
    )

    with pytest.raises(InternalServerError) as caught:
        _oss_upload.get_upload_certificate(client, "m")
    assert caught.value.code == "InternalServerError"
    assert caught.value.status_code == 500
    assert caught.value.request_id == request_id
    assert caught.value.body == body
    assert caught.value.headers["x-request-id"] == "header-id"


def test_certificate_fetch_rejects_invalid_json_as_internal_error():
    client = _make_client(
        lambda request: httpx.Response(
            200,
            text="not-json",
            headers={"x-request-id": "header-id"},
        ),
    )

    with pytest.raises(InternalServerError) as caught:
        _oss_upload.get_upload_certificate(client, "m")
    assert caught.value.code == "InternalServerError"
    assert caught.value.status_code == 500
    assert caught.value.request_id == "header-id"
    assert caught.value.body == "not-json"
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_run_blocking_leaves_the_default_executor_for_async_transport(
    monkeypatch,
):
    real_to_thread = asyncio.to_thread
    default_executor_calls = []

    async def tracked_to_thread(fn, *args, **kwargs):
        default_executor_calls.append(fn)
        return await real_to_thread(fn, *args, **kwargs)

    class ExecutorUsingTransport:
        async def request(self):
            return await asyncio.to_thread(lambda: "ok")

    monkeypatch.setattr(asyncio, "to_thread", tracked_to_thread)
    client = type("Client", (), {"transport": ExecutorUsingTransport()})()

    result = await run_blocking(
        client,
        lambda bridged: bridged.transport.request(),
    )

    assert result == "ok"
    # Only the async transport used the loop default executor. The outer
    # blocking call ran on the isolated pool, so the two cannot starve each
    # other when concurrent uploads fill the bridge pool.
    assert len(default_executor_calls) == 1


@pytest.mark.asyncio
async def test_async_client_uploads_over_the_bridged_transport(
    tmp_path,
    monkeypatch,
):
    seen = _capture_oss(monkeypatch)
    calls = []
    client = _make_async_client(_policy_handler(calls))

    url, cert = await run_blocking(
        client,
        _oss_upload.upload_local_file,
        "qwen-vl-max",
        _png(tmp_path),
    )

    # The policy fetch must come back a response, not the un-awaited
    # coroutine an AsyncTransport hands a caller that forgets to await.
    assert url == f"oss://{_CERT['upload_dir']}/cat.png"
    assert cert == _CERT
    assert seen["url"] == OSS_HOST
    assert len(calls) == 1
    assert calls[0].headers.get("Authorization") == "Bearer sk-test"
    await client.aclose()


def test_upload_local_file_builds_the_oss_form(tmp_path, monkeypatch):
    seen = _capture_oss(monkeypatch)
    client, calls = _policy_client()
    file_path = _png(tmp_path)

    url, cert = _oss_upload.upload_local_file(
        client,
        "qwen-vl-max",
        file_path,
    )

    assert url == f"oss://{_CERT['upload_dir']}/cat.png"
    assert cert == _CERT
    assert seen["url"] == OSS_HOST
    assert seen["form"] == {
        "OSSAccessKeyId": _CERT["oss_access_key_id"],
        "Signature": _CERT["signature"],
        "policy": _CERT["policy"],
        "key": _CERT["upload_dir"] + "/cat.png",
        "x-oss-object-acl": "private",
        "x-oss-forbid-overwrite": "true",
        "success_action_status": "200",
        "x-oss-content-type": "image/png",
    }
    assert seen["file_name"] == "cat.png"
    # OSS rejects a post-object request whose "file" part is not last.
    assert seen["body"].rindex('name="key"') < seen["body"].rindex(
        'name="file"',
    )
    assert len(calls) == 1


def test_unknown_extension_falls_back_to_octet_stream(tmp_path, monkeypatch):
    seen = _capture_oss(monkeypatch)
    client, _ = _policy_client()
    file_path = str(tmp_path / "blob")
    with open(file_path, "wb") as handle:
        handle.write(b"x")

    _oss_upload.upload_local_file(client, "qwen-vl-max", file_path)

    assert seen["form"]["x-oss-content-type"] == "application/octet-stream"


def test_oss_post_carries_no_dashscope_credential(tmp_path, monkeypatch):
    seen = _capture_oss(monkeypatch)
    client, _ = _policy_client()

    _oss_upload.upload_local_file(client, "qwen-vl-max", _png(tmp_path))

    # The credential boundary: the storage host must never see the API key,
    # so this hop uses its own auth-free client rather than the transport.
    assert seen["headers"].get("Authorization") is None
    assert "sk-test" not in seen["body"]
    assert seen["headers"].get("Accept") == "application/json"
    user_agent = seen["headers"].get("User-Agent", "")
    assert user_agent.startswith("dashscope-v2-python/")


def test_oss_post_failure_maps_to_a_dashscope_error(tmp_path, monkeypatch):
    _capture_oss(monkeypatch, status=403)
    client, _ = _policy_client()

    with pytest.raises(PermissionDeniedError):
        _oss_upload.upload_local_file(client, "qwen-vl-max", _png(tmp_path))


def test_certificate_is_reused_across_uploads(tmp_path, monkeypatch):
    _capture_oss(monkeypatch)
    client, calls = _policy_client()
    first = _png(tmp_path, "a.png")
    second = _png(tmp_path, "b.png")

    url_a, cert = _oss_upload.upload_local_file(client, "m", first)
    url_b, _ = _oss_upload.upload_local_file(
        client,
        "m",
        second,
        certificate=cert,
    )

    assert url_a == f"oss://{_CERT['upload_dir']}/a.png"
    assert url_b == f"oss://{_CERT['upload_dir']}/b.png"
    # Only the first upload needed a policy.
    assert len(calls) == 1


def test_check_and_upload_local_keeps_remote_urls_untouched():
    client, calls = _policy_client()

    for content in (
        "https://example.com/cat.png",
        "http://example.com/cat.png",
    ):
        assert _oss_upload.check_and_upload_local(client, "m", content) == (
            False,
            content,
            None,
        )
    assert calls == []


def test_check_and_upload_local_flags_an_existing_oss_url():
    client, calls = _policy_client()

    result = _oss_upload.check_and_upload_local(
        client,
        "m",
        "oss://bucket/cat.png",
    )

    # Nothing was uploaded, but the caller still has to opt the request into
    # OSS resolution.
    assert result == (True, "oss://bucket/cat.png", None)
    assert calls == []


def test_check_and_upload_local_uploads_a_file_uri(tmp_path, monkeypatch):
    _capture_oss(monkeypatch)
    client, calls = _policy_client()
    file_path = _png(tmp_path)

    is_up, url, cert = _oss_upload.check_and_upload_local(
        client,
        "qwen-vl-max",
        f"file://{file_path}",
        api_key="sk-override",
    )

    assert is_up is True
    assert url == f"oss://{_CERT['upload_dir']}/cat.png"
    assert cert == _CERT
    assert len(calls) == 1
    assert calls[0].headers.get("Authorization") == "Bearer sk-override"


def test_check_and_upload_local_uploads_a_bare_path(tmp_path, monkeypatch):
    _capture_oss(monkeypatch)
    client, _ = _policy_client()

    is_up, url, _ = _oss_upload.check_and_upload_local(
        client,
        "m",
        _png(tmp_path),
    )

    assert is_up is True
    assert url == f"oss://{_CERT['upload_dir']}/cat.png"


def test_check_and_upload_local_expands_a_path_that_is_not_a_file():
    client, calls = _policy_client()

    is_up, value, cert = _oss_upload.check_and_upload_local(
        client,
        "m",
        "~/nope.png",
    )

    # Passed through, but with ~ expanded as in 1.x.
    assert (is_up, cert) == (False, None)
    assert value == os.path.expanduser("~/nope.png")
    assert calls == []


def test_missing_file_uri_raises_value_error(tmp_path):
    client, calls = _policy_client()
    missing = (tmp_path / "gone.png").as_uri()

    with pytest.raises(ValueError, match="is not exists"):
        _oss_upload.check_and_upload_local(client, "m", missing)
    assert calls == []


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("file:///tmp/a%20b.png", "/tmp/a b.png"),
        ("file://localhost/tmp/a.png", "localhost/tmp/a.png"),
        ("file:///C:/Users/me/a.png", "C:/Users/me/a.png"),
    ],
)
def test_resolve_file_uri_path(uri, expected):
    assert _oss_upload._resolve_file_uri_path(uri) == expected
