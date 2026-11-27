# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared fixtures/helpers for the ManagementClient (module G) tests.

Not collected by pytest (no ``test_`` prefix). Every management test
module calls ``pytest.importorskip`` on the alibabacloud libs *before*
importing this module, so the optional ``[management]`` extra stays
optional for the rest of the suite.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

import httpx

from dashscope.management import AsyncManagementClient, ManagementClient

# Fake credentials -- never real keys (detect-private-key safe).
AK = "ak-test"
SK = "sk-test"
STS_TOKEN = "sts-token-test"

MS_ENVELOPE: Dict[str, Any] = {
    "code": "200",
    "success": True,
    "requestId": "req-ms",
    "httpStatusCode": 200,
}
TP_ENVELOPE: Dict[str, Any] = {
    "Success": True,
    "Code": "200",
    "Message": "OK",
    "RequestId": "req-tp",
    "HttpStatusCode": 200,
}


def ms_body(**kw: Any) -> Dict[str, Any]:
    """A successful modelstudio (lowercase) envelope plus payload fields."""
    return {**MS_ENVELOPE, **kw}


def tp_body(**kw: Any) -> Dict[str, Any]:
    """A successful tokenplan (PascalCase) envelope plus payload fields."""
    return {**TP_ENVELOPE, **kw}


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> ManagementClient:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs.setdefault("access_key_id", AK)
    kwargs.setdefault("access_key_secret", SK)
    return ManagementClient(http_client=http_client, **kwargs)


def make_async_client(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> AsyncManagementClient:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    kwargs.setdefault("access_key_id", AK)
    kwargs.setdefault("access_key_secret", SK)
    return AsyncManagementClient(http_client=http_client, **kwargs)


def action_handler(
    bodies: Mapping[str, Any],
    *,
    status: int = 200,
    seen: Optional[list] = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """Return ``bodies[x-acs-action]`` for each request, recording actions."""

    def handler(request: httpx.Request) -> httpx.Response:
        action = request.headers["x-acs-action"]
        if seen is not None:
            seen.append(action)
        return httpx.Response(status, json=bodies[action])

    return handler


def query_of(request: httpx.Request) -> Dict[str, str]:
    return dict(request.url.params)
