# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""SSE framing shared by the job-event streams (fine-tunes, deployments).

Wire format mirrors 1.x ``StreamEventMixin`` (client/base_api.py): an
``event:error`` line flips the error flag, ``status:`` carries the mapped
HTTP status (optional), and ``data:`` lines carry the payload; an error
frame ends the stream.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterator, Mapping, Optional

from dashscope._logging import (
    get_logger,
    request_id_for_log,
    request_id_from_headers,
)
from dashscope.errors import (
    DashScopeError,
    error_from_business_code,
    error_from_status,
)

__all__ = []


def _try_json(data: str) -> Any:
    try:
        return json.loads(data)
    except (TypeError, ValueError):
        return data


def _log_stream_error(
    data: str,
    headers: Optional[Mapping[str, str]],
) -> None:
    if headers is None:
        return
    body = _try_json(data)
    body = body if isinstance(body, dict) else {}
    code = body.get("code")
    get_logger("transport.sse").error(
        "SSE business error: code=%s, request_id=%s",
        code if isinstance(code, str) else "unknown",
        request_id_for_log(
            body.get("request_id"),
            request_id_from_headers(headers),
        ),
    )


def _stream_error(
    status_code: Optional[int],
    data: str,
    headers: Optional[Mapping[str, str]] = None,
) -> DashScopeError:
    """Build the error for one ``event:error`` frame.

    ``status:`` is optional on the wire. Absent, the frame is a business
    failure carried on an HTTP 200 stream -- not a server 500 that nothing
    reported -- so the body's own ``code`` picks the class.
    """
    body = _try_json(data)
    message = data
    if isinstance(body, dict):
        message = body.get("message") or data
    if status_code is None:
        error = error_from_business_code(message=message, body=body)
    else:
        error = error_from_status(status_code, message=message, body=body)
    # Preserve the historical body-only ID contract on job-event exceptions.
    _log_stream_error(data, headers)
    return error


def _iter_sse(response) -> Iterator[Any]:
    is_error = False
    status_code: Optional[int] = None
    try:
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("event:error"):
                is_error = True
            elif line.startswith("status:"):
                status_code = int(line[len("status:") :].strip())
            elif line.startswith("data:"):
                yield is_error, status_code, line[len("data:") :].strip()
                if is_error:
                    break
    except Exception:
        get_logger("transport.sse").error(
            "Failed to read or parse job SSE stream, request_id=%s",
            request_id_for_log(request_id_from_headers(response.headers)),
        )
        raise


async def _aiter_sse(response) -> AsyncIterator[Any]:
    # Async twin of ``_iter_sse``; the framing rules are identical.
    is_error = False
    status_code: Optional[int] = None
    try:
        async for line in response.aiter_lines():
            if not line:
                continue
            if line.startswith("event:error"):
                is_error = True
            elif line.startswith("status:"):
                status_code = int(line[len("status:") :].strip())
            elif line.startswith("data:"):
                yield is_error, status_code, line[len("data:") :].strip()
                if is_error:
                    break
    except Exception:
        get_logger("transport.sse").error(
            "Failed to read or parse job SSE stream, request_id=%s",
            request_id_for_log(request_id_from_headers(response.headers)),
        )
        raise
