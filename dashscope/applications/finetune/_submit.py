# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The job-submit POST for ``AgenticRL``, self-contained inside 2.0.

``AgenticRL.submit_job`` used to inherit ``CreateMixin`` from
``dashscope.legacy_1x.client.base_api`` for one non-streaming POST. Keeping that
would have left the last import of the 1.x client inside ``dashscope``,
so the call is spelled out here instead: a few dozen lines, against
pulling the 1.x request/response stack (``default_headers``, ``join_url``,
``_handle_http_stream_response``) into the 2.0 package.

The wire format is deliberately byte-identical to what ``CreateMixin``
sent, header order included, because the backend keys its statistics off
``x-dashscope-sdk-client`` and ``user-agent``:

- The module tag stays the *unmarked* ``finetune``. Deriving it from this
  module's dotted name would now yield ``v2``, and the v2 transport's
  marked form (``finetune2``) belongs to ``client.rl.jobs`` rather than
  to the submit path. Both spellings already exist today; each keeps its
  own.
- The identity headers come from ``dashscope._shared._sdk_ident``, so the
  CLI's ``set_sdk_client("python-cli")`` override and the process-wide
  session id still apply, and ``DASHSCOPE_DISABLE_SDK_HEADERS`` still
  suppresses them.

Only the non-SSE branches of the 1.x response parser are mirrored.
``submit_job`` posts without streaming to ``/fine-tunes``, which answers
``application/json``; a ``text/event-stream`` reply could not be consumed
meaningfully on this path, so there is nothing to reproduce.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import requests

from dashscope._shared._sdk_ident import get_sdk_headers, get_user_agent

__all__ = ["SUBMIT_TIMEOUT", "post_submit"]

# Mirrors 1.x's DEFAULT_REQUEST_TIMEOUT_SECONDS.
SUBMIT_TIMEOUT = 300

_JSON = "application/json; charset=utf-8"

# Hard-coded rather than derived from __module__: see the module
# docstring. The backend already reports on the unmarked spelling.
_SDK_MODULE = "finetune"


def _headers(api_key: str) -> Dict[str, str]:
    """Build the request headers in 1.x's insertion order."""
    headers = {"Content-Type": _JSON, "user-agent": get_user_agent()}
    headers.update(get_sdk_headers(module=_SDK_MODULE))
    headers["Authorization"] = f"Bearer {api_key}"
    headers["Accept"] = _JSON
    return headers


def _success(response: requests.Response) -> Dict[str, Any]:
    body = response.json()
    output: Any = body.get("data")
    if "output" in body:
        output = body["output"]
    request_id = body.pop("request_id", "")
    if "data" not in body and "output" not in body:
        output = body
    return {
        "request_id": request_id,
        "status_code": response.status_code,
        "code": body.get("code"),
        "message": body.get("message", ""),
        "output": output,
        "usage": body.get("usage"),
        "headers": dict(response.headers),
    }


def _failure(response: requests.Response) -> Dict[str, Any]:
    code: Any = "Unknown"
    message: Any = response.content.decode("utf-8")
    request_id = ""
    if "application/json" in response.headers.get("content-type", ""):
        error = response.json()
        code = error.get("code")
        message = error.get("message", "")
        if "msg" in error:
            message = error["msg"]
        request_id = error.get("request_id", "")
    return {
        "request_id": request_id,
        "status_code": response.status_code,
        "code": code,
        "message": message,
        "output": None,
        "usage": None,
        "headers": dict(response.headers),
    }


def post_submit(
    request: Dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    path: str,
    timeout: int = SUBMIT_TIMEOUT,
) -> Dict[str, Any]:
    """POST ``request`` and return the raw response envelope.

    The seven keys are the ones ``DashScopeAPIResponse`` carries, so the
    1.x surface can keep building ``FineTune(**envelope)`` and the 2.0
    surface can wrap the same dict in ``SubmitResult``.

    ``base_url`` is used verbatim. The RL constants hold no
    ``{workspace_id}`` placeholder and ``submit_job`` never sends a
    workspace, so 1.x's ``resolve_base_url`` and ``_workspace_header``
    were both no-ops on this path and are not reproduced.

    Args:
        request: The job payload, serialized with ``ensure_ascii=False``
            so non-ASCII job names reach the server intact.
        api_key: Bearer token; the caller resolves it.
        base_url: Management-plane host, without a trailing slash.
        path: Endpoint under ``base_url``.
        timeout: Seconds for the whole request.
    """
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    with requests.Session() as session:
        response = session.post(
            url,
            data=json.dumps(request, ensure_ascii=False).encode("utf-8"),
            headers=_headers(api_key),
            timeout=timeout,
        )
    if response.status_code in (200, 201):
        return _success(response)
    return _failure(response)
