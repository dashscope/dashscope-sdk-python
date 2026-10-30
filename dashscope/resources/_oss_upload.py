# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Local-file → OSS upload, native to the 2.0 SDK.

DashScope serves media from URLs, so a local path or ``file://`` URI has to
be posted to OSS first and the resulting ``oss://`` URL sent instead. The
upload is two hops with opposite trust levels:

* the policy fetch (``GET /uploads?action=getPolicy``) goes to DashScope and
  **does** carry the credential, so it rides the client transport and picks
  up its connection pool, retries and per-call api_key/workspace;
* the post-object upload goes to the OSS host and must **not** carry it. The
  transport merges its default ``Authorization: Bearer`` into every request,
  and httpx only strips that header on cross-origin *redirects* -- posting an
  absolute OSS URL straight through the transport would hand the API key to
  the storage host. Hence the separate auth-free client in ``_post_to_oss``.
"""

from __future__ import annotations

import mimetypes
import os
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote_plus, urlparse

import httpx

from dashscope._logging import get_logger
from dashscope.errors import (
    check_response,
    error_from_status,
    internal_response_error,
    response_json,
)
from dashscope.transport.http import default_user_agent

__all__ = [
    "check_and_upload_local",
    "get_upload_certificate",
    "upload_local_file",
]

_FILE_URI_SCHEME = "file://"
_OSS_SCHEME = "oss://"
_CERT_PATH = "/uploads"
# The client default (300s) is far too short for a large video.
_OSS_POST_TIMEOUT = 3600.0

logger = get_logger("oss")


def _resolve_file_uri_path(file_uri: str) -> str:
    """Turn a ``file://`` URI into a filesystem path."""
    parsed = urlparse(file_uri)
    if parsed.netloc:
        file_path = parsed.netloc + unquote_plus(parsed.path)
    else:
        file_path = unquote_plus(parsed.path)

    # A Windows URI parses to "/C:/dir/file"; drop the leading slash.
    if (
        file_path.startswith("/")
        and len(file_path) > 2
        and file_path[2] == ":"
    ):
        file_path = file_path[1:]

    return os.path.expanduser(file_path)


def _policy_from_response(response: httpx.Response) -> Dict[str, Any]:
    """Pull the policy out of the getPolicy envelope.

    ``data`` is the key the gateway answers with -- and the one 1.x read
    first. Accepting ``output`` too would let an unrelated body (an
    embeddings response, say) through as a "policy", deferring the failure
    to a ``KeyError`` on some policy field far from here.
    """
    body = response_json(response)
    policy = body.get("data")
    if not isinstance(policy, dict):
        raise internal_response_error(response, body)
    return policy


def get_upload_certificate(
    client,
    model: str,
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch the OSS post-object policy for ``model``.

    Raises even when the client was built with ``raise_on_error=False``:
    without a policy there is no URL to upload to and nothing to hand back.

    Blocking, like the rest of this module: an ``AsyncClient`` reaches it
    through the bridge in :mod:`dashscope.resources._blocking`, which is
    what lets ``client.transport.request`` return a response here instead
    of a coroutine.
    """
    response = client.transport.request(
        "GET",
        _CERT_PATH,
        params={"action": "getPolicy", "model": model},
        api_key=api_key,
        workspace=workspace,
    )
    check_response(response)
    return _policy_from_response(response)


def _post_to_oss(
    upload_host: str,
    form: Dict[str, Any],
    file_name: str,
    file_path: str,
    transport: Optional[httpx.BaseTransport] = None,
) -> httpx.Response:
    """Post one file to OSS with a client that carries no DashScope auth.

    The file is streamed from a live handle rather than read into memory, so
    a large video does not have to fit in the process. ``transport`` exists
    for tests to inject an ``httpx.MockTransport``.
    """
    headers = {
        "Accept": "application/json",
        "User-Agent": default_user_agent(),
    }
    with httpx.Client(
        timeout=_OSS_POST_TIMEOUT,
        transport=transport,
    ) as oss_client:
        with open(file_path, "rb") as handle:
            return oss_client.post(
                upload_host,
                data=form,
                files={"file": (file_name, handle)},
                headers=headers,
            )


def upload_local_file(
    client,
    model: str,
    file_path: str,
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
    certificate: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Upload one local file and return ``(oss_url, certificate)``.

    Pass the returned certificate back in to reuse it across a batch instead
    of fetching a fresh policy per file.
    """
    cert = certificate
    if cert is None:
        cert = get_upload_certificate(
            client,
            model,
            api_key=api_key,
            workspace=workspace,
        )
    key = cert["upload_dir"] + "/" + os.path.basename(file_path)
    form = {
        "OSSAccessKeyId": cert["oss_access_key_id"],
        "Signature": cert["signature"],
        "policy": cert["policy"],
        "key": key,
        "x-oss-object-acl": cert["x_oss_object_acl"],
        "x-oss-forbid-overwrite": cert["x_oss_forbid_overwrite"],
        "success_action_status": "200",
        "x-oss-content-type": (
            mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        ),
    }
    response = _post_to_oss(
        cert["upload_host"],
        form,
        os.path.basename(file_path),
        file_path,
    )
    if not response.is_success:
        logger.error(
            "uploading file %s to oss failed, status %s",
            file_path,
            response.status_code,
        )
        # OSS answers with XML, so there is no business code to read and the
        # error is mapped from the status code alone.
        raise error_from_status(
            response.status_code,
            message=response.text,
            headers=response.headers,
        )
    return _OSS_SCHEME + key, cert


def check_and_upload_local(
    client,
    model: str,
    content: str,
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
    certificate: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Upload ``content`` to OSS when it names a local file.

    Returns ``(is_upload, url_or_content, certificate)``. An existing
    ``oss://`` URL passes through untouched but still reports
    ``is_upload=True``, so the caller opts the request into OSS resolution;
    ``http(s)://`` and anything unrecognized pass through as ``False``.
    """
    if content.startswith(_FILE_URI_SCHEME):
        file_path = _resolve_file_uri_path(content)
        if not os.path.isfile(file_path):
            raise ValueError(f"The file: {file_path} is not exists!")
        url, cert = upload_local_file(
            client,
            model,
            file_path,
            api_key=api_key,
            workspace=workspace,
            certificate=certificate,
        )
        return True, url, cert
    if content.startswith(_OSS_SCHEME):
        return True, content, certificate
    if not content.startswith("http"):
        content = os.path.expanduser(content)
        if os.path.isfile(content):
            url, cert = upload_local_file(
                client,
                model,
                content,
                api_key=api_key,
                workspace=workspace,
                certificate=certificate,
            )
            return True, url, cert
    return False, content, certificate
